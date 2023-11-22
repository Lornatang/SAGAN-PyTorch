# Copyright 2023 Lornatang Authors. All Rights Reserved.
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

import numpy as np
import torch
from scipy import linalg
from torch import Tensor
from torch.nn import functional as F_torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import find_classes
from torchvision.models import inception_v3, Inception3, Inception_V3_Weights
from tqdm import tqdm

from sagan_pytorch.data.img_data import ImageData
from sagan_pytorch.models.generator import Generator

__all__ = [
    "FID",
]


class FID:
    def __init__(
            self,
            root: str,
            transform: transforms.Compose = None,
            batch_size: int = 64,
            num_workers: int = 4,
            num_samples: int = 5000,
            model: Generator = None,
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            verbose: bool = False,
    ) -> None:
        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.device = device
        self.verbose = verbose

        # Load model
        self.model = model
        self.model.to(device)
        self.model.eval()

        self.fid_model = self.load_fid_model()
        self.fid_model.to(device)
        self.fid_model.eval()

        # Load dataset
        self.cls2id = find_classes(root)[1]

    def load_fid_model(self):
        def _forward(self, x: Tensor) -> Tensor:
            x = F_torch.upsample(x, size=(299, 299), mode="bilinear", align_corners=True)

            x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
            x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
            x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
            x = F_torch.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

            x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
            x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
            x = F_torch.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

            x = self.Mixed_5b(x)  # 35 x 35 x 192
            x = self.Mixed_5c(x)  # 35 x 35 x 256
            x = self.Mixed_5d(x)  # 35 x 35 x 288

            x = self.Mixed_6a(x)  # 35 x 35 x 288
            x = self.Mixed_6b(x)  # 17 x 17 x 768
            x = self.Mixed_6c(x)  # 17 x 17 x 768
            x = self.Mixed_6d(x)  # 17 x 17 x 768
            x = self.Mixed_6e(x)  # 17 x 17 x 768

            x = self.Mixed_7a(x)  # 17 x 17 x 768
            x = self.Mixed_7b(x)  # 8 x 8 x 1280
            x = self.Mixed_7c(x)  # 8 x 8 x 2048

            x = F_torch.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

            return x.squeeze()  # 1 x 1 x 2048

        fid_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        fid_model.eval()
        fid_model.forward = _forward.__get__(fid_model, Inception3)

        return fid_model

    def compute(self) -> float:
        fids = []

        for cls_name, index in self.cls2id.items():
            dataset = ImageData(os.path.join(self.root, cls_name), self.transform)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=True,
            )

            # Extract real features
            real_feature = []
            with torch.no_grad():
                for imgs in tqdm(dataloader, desc="Extract features of real images"):
                    imgs = imgs.to(self.device)
                    features = self.fid_model(imgs).detach().cpu().numpy()
                    real_feature.append(features)

            real_feature = np.concatenate(real_feature)
            real_feature_mean = np.mean(real_feature, 0)
            real_feature_cov = np.cov(real_feature, rowvar=False)
            del real_feature

            # Extract fake features
            fake_feature = []
            with torch.no_grad():
                n_iter = self.num_samples // self.batch_size
                resid = self.num_samples - n_iter * self.batch_size
                n_samples = [self.batch_size] * n_iter
                if resid != 0:
                    n_samples += [resid]
                for i in tqdm(n_samples, desc="Extract features of fake images"):
                    x_cls = torch.full([i], id, dtype=torch.long, device=self.device)
                    code = torch.randn(i, self.model.noise_dim, dtype=torch.long, device=self.device)
                    fake_imgs = self.model(code, x_cls)
                    features = self.fid_model(fake_imgs).detach().cpu().numpy()
                    fake_feature.append(features)

            fake_feature = np.concatenate(fake_feature)
            fake_feature_mean = np.mean(fake_feature, 0)
            fake_feature_cov = np.cov(fake_feature, rowvar=False)
            del fake_feature

            cov_sqrt, _ = linalg.sqrtm(fake_feature_cov @ real_feature_cov, disp=False)
            if not np.isfinite(cov_sqrt).all():
                if self.verbose:
                    print("Product of cov matrices is singular")
                offset = np.eye(fake_feature_cov.shape[0]) * 1e-6
                cov_sqrt = linalg.sqrtm((fake_feature_cov + offset) @ (real_feature_cov + offset))

            if np.iscomplexobj(cov_sqrt):
                if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                    m = np.max(np.abs(cov_sqrt.imag))
                    raise ValueError(f"Imaginary component {m}")

                cov_sqrt = cov_sqrt.real

            mean_diff = fake_feature_mean - real_feature_mean
            mean_norm = mean_diff @ mean_diff

            trace = np.trace(fake_feature_cov) + np.trace(real_feature_cov) - 2 * np.trace(cov_sqrt)

            fid = mean_norm + trace

            if self.verbose:
                print(f"FID score of class {id} ({cls_name}): {fid}")
            fids.append(fid)

        return sum(fids) / len(fids)
