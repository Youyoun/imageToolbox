from typing import Tuple

import torch
from torch.utils import data as data


class NoiseDataset(data.Dataset):
    def __init__(self,
                 images: list,
                 noise: callable,
                 blur: callable,
                 transforms: callable = None,
                 percent_data: float = 0.1):
        super().__init__()
        self.images = images
        self.noise_fn = noise
        self.blur_fn = blur
        self.transforms = transforms
        self.percent_data = percent_data
        self.blur_transform = blur

    def __getitem__(self, ind) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.images[ind]
        assert label.max() == 1, "Image maximal value is > 1"
        label = self.transforms(label)
        min_ = 0
        if label.min() < 0:
            min_ = -1
        noisified = self.noise_fn(self.blur_fn(label))
        noisified = torch.clamp(noisified, min_, 1)
        return noisified, label

    def __len__(self) -> int:
        return int(len(self.images) * self.percent_data)
