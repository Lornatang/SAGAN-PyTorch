# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time
from pathlib import Path
from typing import Dict, Union

import torch
import torch.utils.data
from torch import nn, optim, Tensor
from torch.cuda import amp
from torch.nn import functional as F_torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from sagan_pytorch.data.prefetcher import CPUPrefetcher, CUDAPrefetcher
from sagan_pytorch.models.discriminator import discriminator
from sagan_pytorch.models.generator import generator
from sagan_pytorch.models.utils import load_resume_state_dict
from sagan_pytorch.utils import AverageMeter, ProgressMeter


class Trainer:
    def __init__(
            self,
            config: Dict,
            scaler: amp.GradScaler,
            device: torch.device,
            save_weights_dir: Union[str, Path],
            save_visuals_dir: Union[str, Path],
            tblogger: SummaryWriter,
    ) -> None:
        self.config = config
        self.scaler = scaler
        self.device = device
        self.save_weights_dir = save_weights_dir
        self.save_visuals_dir = save_visuals_dir
        self.tblogger = tblogger

        self.start_epoch = 0

        self.g_model, self.ema_g_model, self.d_model = self.build_model()
        self.g_optim, self.d_optim = self.define_optim()
        self.dataloader = self.load_datasets()
        self.batches = len(self.dataloader)

        # For training visualization, select a fixed batch of data
        self.fixed_noise = torch.randn(self.config["MODEL"]["G"]["NUM_CLASSES"] * 5, self.config["MODEL"]["G"]["NOISE_DIM"], device=self.device)
        self.fixed_class = torch.arange(8, device=self.device).long().repeat(5)

        self.load_checkpoint()

    def build_model(self) -> tuple:
        """Build the generator, discriminator and exponential average generator models

        Returns:
            tuple: generator, exponential average generator and discriminator models
        """

        g_model = generator(noise_dim=self.config["MODEL"]["G"]["NOISE_DIM"], num_classes=self.config["MODEL"]["G"]["NUM_CLASSES"])
        d_model = discriminator(num_classes=self.config["MODEL"]["D"]["NUM_CLASSES"])

        g_model = g_model.to(self.device)
        d_model = d_model.to(self.device)

        # Generate an exponential average models based on the generator to stabilize models training
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.001 * averaged_model_parameter + 0.999 * model_parameter
        ema_g_model = AveragedModel(g_model, self.device, ema_avg_fn)

        # Compile model
        if self.config["MODEL"]["G"]["COMPILED"]:
            g_model = torch.compile(g_model)
            ema_g_model = torch.compile(ema_g_model)
        elif self.config["MODEL"]["D"]["COMPILED"]:
            d_model = torch.compile(d_model)

        return g_model, ema_g_model, d_model

    def define_optim(self) -> tuple:
        """Define the optimizer

        Returns:
            torch.optim.Optimizer: generator optimizer, discriminator optimizer
        """

        self.g_optim = optim.Adam(
            self.g_model.parameters(),
            self.config["TRAIN"]["OPTIM"]["G"]["LR"],
            (self.config["TRAIN"]["OPTIM"]["G"]["BETA1"], self.config["TRAIN"]["OPTIM"]["G"]["BETA2"]),
        )
        self.d_optim = optim.Adam(
            self.d_model.parameters(),
            self.config["TRAIN"]["OPTIM"]["D"]["LR"],
            (self.config["TRAIN"]["OPTIM"]["D"]["BETA1"], self.config["TRAIN"]["OPTIM"]["D"]["BETA2"]),
        )

        return self.g_optim, self.d_optim

    def load_datasets(self) -> CPUPrefetcher or CUDAPrefetcher:
        """Load the dataset and generate the dataset iterator

        Returns:
            DataLoader: dataset iterator
        """

        transform = transforms.Compose([
            transforms.Resize(self.config["DATASETS"]["IMG_SIZE"]),
            transforms.CenterCrop(self.config["DATASETS"]["IMG_SIZE"]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Load the train dataset
        datasets = ImageFolder(self.config["DATASETS"]["IMGS_DIR"], transform)
        # generate dataset iterator
        dataloader = torch.utils.data.DataLoader(datasets,
                                                 batch_size=self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                                 shuffle=True,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 drop_last=True,
                                                 persistent_workers=True)

        return dataloader

    @staticmethod
    def _requires_grad(model: nn.Module, flag: bool = True) -> None:
        for p in model.parameters():
            p.requires_grad = flag

    def update_g(self):
        # Disable discriminator backpropagation during generator training
        self._requires_grad(self.d_model, False)
        self._requires_grad(self.g_model, True)

        # Initialize the generator model gradient
        self.g_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            noise = torch.randn(self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"], self.config["MODEL"]["G"]["NOISE_DIM"], device=self.device)
            inputs = torch.ones(self.config["MODEL"]["G"]["NUM_CLASSES"], device=self.device)
            input_class = torch.multinomial(inputs, self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"], replacement=True)
            fake_imgs = self.g_model(noise, input_class)
            fake_output = self.d_model(fake_imgs, input_class)
            g_loss = - torch.mean(fake_output)

        # Backpropagation
        self.scaler.scale(g_loss).backward()
        # update generator weights
        self.scaler.step(self.g_optim)
        self.scaler.update()

        # update exponentially averaged model weights
        self.ema_g_model.update_parameters(self.g_model)

        return g_loss

    def update_d(self, imgs: Tensor, labels: Tensor):
        # During discriminator model training, enable discriminator model backpropagation
        self._requires_grad(self.d_model, True)
        self._requires_grad(self.g_model, False)

        # Initialize the discriminator model gradient
        self.d_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            noise = torch.randn(self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"], self.config["MODEL"]["G"]["NOISE_DIM"], device=self.device)
            fake_imgs = self.g_model(noise, labels)

            real_output = self.d_model(imgs, labels)
            fake_output = self.d_model(fake_imgs, labels)
            d_loss_real = F_torch.relu(1 + fake_output).mean()
            d_loss_fake = F_torch.relu(1 - real_output).mean()

        # Compute the discriminator total loss value
        d_loss = d_loss_real + d_loss_fake
        # Backpropagation
        self.scaler.scale(d_loss).backward()
        # update generator weights
        self.scaler.step(self.d_optim)
        self.scaler.update()

        return d_loss, d_loss_real, d_loss_fake

    def load_checkpoint(self) -> None:
        """Load the checkpoint"""

        def _load(model: nn.Module, weights_path: str) -> None:
            if weights_path != "" and os.path.exists(weights_path):
                print(f"Load checkpoint from '{weights_path}'")
                checkpoint = torch.load(weights_path, map_location=self.device)
                if checkpoint["state_dict"] is not None:
                    state_dict = checkpoint["state_dict"]
                elif checkpoint["ema_state_dict"] is not None:
                    state_dict = checkpoint["ema_state_dict"]
                else:
                    raise ValueError("The checkpoint does not contain the 'state_dict' or 'ema_state_dict' key")
                model.load_state_dict(state_dict)

        # Load pretrained model weights
        _load(self.g_model, self.config["TRAIN"]["CHECKPOINT"]["G"]["PRETRAINED_MODEL_WEIGHTS_PATH"])
        _load(self.d_model, self.config["TRAIN"]["CHECKPOINT"]["D"]["PRETRAINED_MODEL_WEIGHTS_PATH"])

        # Load resume model weights
        resume_g_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["G"]["RESUME_MODEL_WEIGHTS_PATH"]
        if resume_g_model_weights_path != "":
            print(f"Load resume checkpoint from '{resume_g_model_weights_path}'")
            self.start_epoch, self.g_model, self.ema_g_model, self.g_optim, _ = load_resume_state_dict(self.g_model,
                                                                                                       self.ema_g_model,
                                                                                                       resume_g_model_weights_path,
                                                                                                       self.config["MODEL"]["G"]["COMPILED"])
        resume_d_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["D"]["RESUME_MODEL_WEIGHTS_PATH"]
        if resume_d_model_weights_path != "":
            print(f"Load resume checkpoint from '{resume_d_model_weights_path}'")
            self.start_epoch, self.d_model, _, self.d_optim, _ = load_resume_state_dict(self.d_model,
                                                                                        None,
                                                                                        resume_d_model_weights_path,
                                                                                        self.config["MODEL"]["D"]["COMPILED"])

    def visual_on_iters(self, iters: int):
        with torch.no_grad():
            sample_imgs = self.g_model(self.fixed_noise, self.fixed_class)
            save_sample_path = os.path.join(self.save_visuals_dir, f"iter-{iters:06d}.jpg")
            save_image(sample_imgs.cpu().data, save_sample_path, nrow=1, padding=0, normalize=True)

    def save_checkpoint(self, epoch: int) -> None:
        # Automatically save models weights
        is_last = (epoch + 1) == self.config["TRAIN"]["HYP"]["EPOCHS"]

        g_state_dict = {
            "epoch": epoch + 1,
            "state_dict": self.g_model.state_dict(),
            "ema_state_dict": self.ema_g_model.state_dict(),
            "optimizer": self.g_optim,
            "scheduler": None,
        }
        d_state_dict = {
            "epoch": epoch + 1,
            "state_dict": self.d_model.state_dict(),
            "ema_state_dict": None,
            "optimizer": self.d_optim,
            "scheduler": None
        }

        if self.config["TRAIN"]["SAVE_EVERY_EPOCH"] and not is_last:
            g_weights_path = os.path.join(self.save_weights_dir, f"g_epoch_{epoch}.pth.tar")
            d_weights_path = os.path.join(self.save_weights_dir, f"d_epoch_{epoch}.pth.tar")
            torch.save(g_state_dict, g_weights_path)
            torch.save(d_state_dict, d_weights_path)
        else:
            g_weights_path = os.path.join(self.save_weights_dir, f"g_last.pth.tar")
            d_weights_path = os.path.join(self.save_weights_dir, f"d_last.pth.tar")
            torch.save(g_state_dict, g_weights_path)
            torch.save(d_state_dict, d_weights_path)

    def train_on_epoch(self, epoch: int):
        # The information printed by the progress bar
        global g_loss
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        g_losses = AverageMeter("G Loss", ":6.6f")
        d_losses = AverageMeter("D Loss", ":6.6f")
        progress = ProgressMeter(self.batches,
                                 [batch_time, data_time, g_losses, d_losses],
                                 prefix=f"Epoch: [{epoch}]")

        # Put the generator in training mode
        self.g_model.train()
        self.d_model.train()

        # Initialize data batches
        batch_index = 0

        # Record the start time of training a batch
        end = time.time()
        for i, (imgs, labels) in enumerate(self.dataloader):
            # Load batches of data
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Record the time to load a batch of data
            data_time.update(time.time() - end)

            # start training the discriminator model
            d_loss, d_loss_real, d_loss_fake = self.update_d(imgs, labels)

            # start training the generator model
            if batch_index % self.config["TRAIN"]["N_CRITIC"] == 0:
                g_loss = self.update_g()

            # record the loss value
            g_losses.update(g_loss.item(), self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])
            d_losses.update(d_loss.item(), self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])

            # Record the total time of training a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output training log information once
            iters = batch_index + epoch * self.batches
            if batch_index % self.config["TRAIN"]["PRINT_FREQ"] == 0:
                # write training log
                self.tblogger.add_scalar("Train/D_Loss", d_loss.item(), iters)
                self.tblogger.add_scalar("Train/D(GT)_Loss", d_loss_real.item(), iters)
                self.tblogger.add_scalar("Train/D(G(z))_Loss", d_loss_fake.item(), iters)
                self.tblogger.add_scalar("Train/G_Loss", g_loss.item(), iters)
                progress.display(batch_index + 1)

            # Save the generated samples
            if (iters + 1) % self.config["TRAIN"]["VISUAL_FREQ"] == 0:
                self.visual_on_iters(iters + 1)

            # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    def train(self):
        for epoch in range(self.start_epoch, self.config["TRAIN"]["HYP"]["EPOCHS"]):
            self.train_on_epoch(epoch)

            # Save weights
            self.save_checkpoint(epoch)
