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
    "Generator", "generator",
]


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, num_classes: int = 10) -> None:
        super().__init__()

        self.linear = spectral_norm(nn.Linear(noise_dim, 4 * 4 * 512))
        self.trunk = nn.ModuleList([
            BasicConvBlock(512, 512, num_classes=num_classes),
            BasicConvBlock(512, 512, num_classes=num_classes),
            BasicConvBlock(512, 256, num_classes=num_classes),
            SelfAttention(256),
            BasicConvBlock(256, 128, num_classes=num_classes),
            BasicConvBlock(128, 64, num_classes=num_classes)
        ])

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.colorize = spectral_norm(nn.Conv2d(64, 3, 3, 1, 1, bias=False))

    def forward(self, x: Tensor, cls_idx: Tensor) -> Tensor:
        x = self.linear(x)
        x = x.view(-1, 512, 4, 4)

        for conv in self.trunk:
            if isinstance(conv, BasicConvBlock):
                x = conv(x, cls_idx)
            else:
                x = conv(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.colorize(x)
        x = self.tanh(x)

        return x


def generator(num_classes: int = 10, **kwargs) -> Generator:
    return Generator(num_classes=num_classes, **kwargs)
