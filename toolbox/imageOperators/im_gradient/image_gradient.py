import enum
from typing import Tuple

import torch
from torch.nn import functional as F

from .utils import to_4D

PADDING_MODE = "constant"


class Directions(enum.IntEnum):
    X = 0
    Y = 1


class Gradient:
    T = None

    def __new__(cls, x: torch.Tensor, direction: Directions) -> torch.Tensor:
        return Gradient.compute(x, direction)

    @staticmethod
    def get_kernel(kernel: torch.Tensor, direction: Directions) -> torch.Tensor:
        if direction == Directions.X:
            kernel = kernel.view(1, 1, -1, 1)
        elif direction == Directions.Y:
            kernel = kernel.view(1, 1, 1, -1)
        else:
            raise ValueError("Invalid direction")
        return kernel

    @staticmethod
    def get_pad(direction: Directions) -> Tuple[int, int, int, int]:
        if direction == Directions.X:
            pad = (0, 0, 0, 1)
        elif direction == Directions.Y:
            pad = (0, 1, 0, 0)
        else:
            raise ValueError("Invalid direction")
        return pad

    @staticmethod
    def compute(x: torch.Tensor, direction: Directions) -> torch.Tensor:
        _kernel = torch.tensor([-1, 1], dtype=torch.float32)
        x, init_shape = to_4D(x)
        return F.pad(F.conv2d(x, weight=Gradient.get_kernel(_kernel, direction).to(x.device), stride=1),
                     Gradient.get_pad(direction), mode=PADDING_MODE).view(*init_shape)


class GradientTranspose:
    T = Gradient

    def __new__(self, x: torch.Tensor, direction: Directions) -> torch.Tensor:
        return GradientTranspose.compute(x, direction)

    @staticmethod
    def get_pad(direction: Directions) -> Tuple[int, int, int, int]:
        if direction == Directions.X:
            pad = (0, 0, 1, 0)
        elif direction == Directions.Y:
            pad = (1, 0, 0, 0)
        else:
            raise ValueError("Invalid direction")
        return pad

    @staticmethod
    def compute(x: torch.Tensor, direction: Directions) -> torch.Tensor:
        _kernel = torch.tensor([1, -1], dtype=torch.float32)
        x, init_shape = to_4D(x)
        return F.conv2d(F.pad(x, GradientTranspose.get_pad(direction), mode=PADDING_MODE),
                        weight=Gradient.get_kernel(_kernel, direction).to(x.device), stride=1).view(*init_shape)


Gradient.T = GradientTranspose
