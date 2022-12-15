from ctypes import Union
from enum import auto

import random
from typing import List, Tuple, Dict, Any

import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T

from toolbox.utils import StrEnum

"""
The methods are defined in:
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
Thanks PyTorch
"""


class Transform:
    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float):
        self.flip_prob = p

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.flip_prob})"


class RandomVerticalFlip(Transform):
    def __init__(self, p: float):
        self.flip_prob = p

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.flip_prob})"


class Random90Rotation(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            image = F.rotate(image, 90)
            target = F.rotate(target, 90)
        return image, target

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(proba={self.p})"


class RandomCrop(Transform):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomScaling(Transform):
    """
    Issue with random scaling: Image is smaller so cant create batch between differents sizes.
    Maybe try something like crop resize instead ?
    Or train at different sizes.

    Just don't use this at all for now.
    """

    def __init__(self, scales: List[float], p: float = 0.5):
        self.p = p
        self.scales = scales

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            scale = random.choice(self.scales)
            image = F.resize(image, (int(image.shape[-2] * scale), int(image.shape[-1] * scale)),
                             interpolation=T.InterpolationMode.NEAREST)
            target = F.resize(target, (int(target.shape[-2] * scale), int(target.shape[-1] * scale)),
                              interpolation=T.InterpolationMode.NEAREST)
        return image, target

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(proba={self.p}, scale={self.scales})"


class AvailableTransforms(StrEnum):
    RandomHorizontalFlip = auto()
    RandomVerticalFlip = auto()
    Random90Rotation = auto()
    RandomScaling = auto()
    RandomCrop = auto()


class TransformsFactory:
    _transforms = {
        AvailableTransforms.RandomVerticalFlip: RandomVerticalFlip,
        AvailableTransforms.RandomHorizontalFlip: RandomHorizontalFlip,
        AvailableTransforms.Random90Rotation: Random90Rotation,
        AvailableTransforms.RandomScaling: RandomScaling,
        AvailableTransforms.RandomCrop: RandomCrop
    }

    @staticmethod
    def get(transform_name: AvailableTransforms, **kwargs) -> Transform:
        return TransformsFactory._transforms[transform_name](**kwargs)


def _convert_transform_str_to_enum(transform_name: str) -> AvailableTransforms:
    for mode in AvailableTransforms:
        if mode == transform_name:
            return mode
    raise ValueError(f"{transform_name} is not implemented here.")


def get_transforms(transforms: List[Tuple[AvailableTransforms | str, Dict[str, Any]]]) -> Compose:
    if transforms is not None and len(transforms) > 0:
        augments = []
        for transform, transform_kwargs in transforms:
            if type(transform) == str:
                transform = _convert_transform_str_to_enum(transform)
            augments.append(TransformsFactory.get(transform, **transform_kwargs))
        return Compose(augments)
    return None
