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

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "ImageData",
]


class ImageData(Dataset):
    def __init__(self, root: str, transform: transforms = None):
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(root)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)
