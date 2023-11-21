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
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torch.nn.utils import spectral_norm

__all__ = [
    "BasicConvBlock", "ConditionalNorm", "SelfAttention",
]


class BasicConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            padding=1,
            stride=1,
            num_classes=None,
            bn=True,
            upsample=True,
            downsample=False,
    ) -> None:
        super().__init__()
        self.bn = bn
        self.upsample = upsample
        self.downsample = downsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False if bn else True))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False if bn else True))

        self.skip_proj = False
        if in_channels != out_channels or upsample or downsample:
            self.conv_skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
            self.skip_proj = True

        if bn:
            self.norm1 = ConditionalNorm(in_channels, num_classes)
            self.norm2 = ConditionalNorm(out_channels, num_classes)

        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor, cls_idx: int = None) -> Tensor:
        out = x

        if self.bn:
            out = self.norm1(out, cls_idx)
        out = self.relu(out)
        if self.upsample:
            out = F_torch.upsample(out, scale_factor=2)
        out = self.conv1(out)

        if self.bn:
            out = self.norm2(out, cls_idx)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample:
            out = F_torch.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = x
            if self.upsample:
                skip = F_torch.upsample(skip, scale_factor=2)
            skip = self.conv_skip(skip)
            if self.downsample:
                skip = F_torch.avg_pool2d(skip, 2)

        else:
            skip = x

        return out + skip


class ConditionalNorm(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()

        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.embed = nn.Embedding(num_classes, channels * 2)
        self.embed.weight.data[:, :channels] = 1
        self.embed.weight.data[:, channels:] = 0

    def forward(self, x: Tensor, cls_idx: int) -> Tensor:
        x = self.bn(x)
        embed = self.embed(cls_idx)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        x = gamma * x + beta

        return x


class SelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.query = spectral_norm(nn.Conv1d(channels, channels // 8, 1, bias=False))
        self.key = spectral_norm(nn.Conv1d(channels, channels // 8, 1, bias=False))
        self.value = spectral_norm(nn.Conv1d(channels, channels, 1, bias=False))

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        # [B, C, H, W] -> [B, C, H * W]
        flatten = x.view(shape[0], shape[1], -1)
        # [B, C, H * W] -> [B, C // 8, H * W]
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        # [B, C // 8, H * W] * [B, H * W, C // 8] -> [B, H * W, H * W]
        query_key = torch.bmm(query, key)
        attn = F_torch.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        x = self.gamma * attn + x
        return x
