import random
from enum import auto
from typing import List, Tuple, Dict, Any, Union

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

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
    def __init__(self, size: int, pad_if_needed=False, fill=0, padding_mode="constant"):
        self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        _, height, width = image.shape
        if self.pad_if_needed and height < self.size:
            padding = [0, self.size - height]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            target = F.pad(target, padding, self.fill, self.padding_mode)
        if self.pad_if_needed and width < self.size:
            padding = [self.size - width, 0]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            target = F.pad(target, padding, self.fill, self.padding_mode)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))

        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(Size={self.size})"


class CenterCrop(Transform):
    def __init__(self, size: int):
        self.size = size
        self.crop_fn = T.CenterCrop(self.size)

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.crop_fn(image)
        target = self.crop_fn(target)
        return image, target

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(Size={self.size})"


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = T.Normalize(mean, std)

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.normalize(image)
        target = self.normalize(target)
        return image, target

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(Mean={self.mean}, Std={self.std})"


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
    CenterCrop = auto()
    Normalize = auto()


class TransformsFactory:
    _transforms = {
        AvailableTransforms.RandomVerticalFlip: RandomVerticalFlip,
        AvailableTransforms.RandomHorizontalFlip: RandomHorizontalFlip,
        AvailableTransforms.Random90Rotation: Random90Rotation,
        AvailableTransforms.RandomScaling: RandomScaling,
        AvailableTransforms.RandomCrop: RandomCrop,
        AvailableTransforms.CenterCrop: CenterCrop,
        AvailableTransforms.Normalize: Normalize
    }

    @staticmethod
    def get(transform_name: AvailableTransforms, **kwargs) -> Transform:
        return TransformsFactory._transforms[transform_name](**kwargs)


def _convert_transform_str_to_enum(transform_name: str) -> AvailableTransforms:
    for mode in AvailableTransforms:
        if mode == transform_name:
            return mode
    raise ValueError(f"{transform_name} is not implemented here.")


def get_transforms(transforms: List[Tuple[Union[AvailableTransforms, str], Dict[str, Any]]]) -> Compose:
    if transforms is not None and len(transforms) > 0:
        augments = []
        for transform, transform_kwargs in transforms:
            if type(transform) == str:
                transform = _convert_transform_str_to_enum(transform)
            augments.append(TransformsFactory.get(transform, **transform_kwargs))
        return Compose(augments)
    return None
