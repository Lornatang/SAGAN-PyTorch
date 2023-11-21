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
from torch import nn, Tensor
from torch.nn.utils import spectral_norm

from .module import BasicConvBlock, SelfAttention

__all__ = [
    "Discriminator", "discriminator",
]


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()

        self.pre_conv = nn.Sequential(spectral_norm(nn.Conv2d(in_channels, 64, 3, 1, 1)),
                                      nn.ReLU(True),
                                      spectral_norm(nn.Conv2d(64, 64, 3, 1, 1)),
                                      nn.AvgPool2d(2))
        self.pre_skip = spectral_norm(nn.Conv2d(in_channels, 64, 1))

        self.trunk = nn.Sequential(BasicConvBlock(64, 128, bn=False, upsample=False, downsample=True),
                                   BasicConvBlock(128, 256, bn=False, upsample=False, downsample=False),
                                   SelfAttention(256),
                                   BasicConvBlock(256, 512, bn=False, upsample=False, downsample=True),
                                   BasicConvBlock(512, 512, bn=False, upsample=False, downsample=True),
                                   BasicConvBlock(512, 512, bn=False, upsample=False, downsample=True))

        self.linear = spectral_norm(nn.Linear(512, 1))

        self.embed = spectral_norm(nn.Embedding(num_classes, 512))
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.relu = nn.ReLU(True)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x: Tensor, cls_idx: Tensor) -> Tensor:
        out = self.pre_conv(x)
        pre_skip = self.avgpool(x)
        pre_skip = self.pre_skip(pre_skip)
        out = out + pre_skip

        out = self.trunk(out)
        out = self.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(cls_idx)
        prod = (out * embed).sum(1)

        return out_linear + prod


def discriminator(num_classes: int = 10, **kwargs) -> Discriminator:
    return Discriminator(num_classes, **kwargs)
