import torch
from torch.nn import functional as F

from ..operators.utils import to_4D


class Directions:
    X = 0
    Y = 1


class Gradient:
    T = None

    def __new__(self, x, direction):
        return Gradient.compute(x, direction)

    @staticmethod
    def get_kernel(kernel, direction):
        if direction == Directions.X:
            kernel = kernel.view(1, 1, -1, 1)
        elif direction == Directions.Y:
            kernel = kernel.view(1, 1, 1, -1)
        else:
            raise ValueError("Invalid direction")
        return kernel

    @staticmethod
    def get_pad(direction):
        if direction == Directions.X:
            pad = (0, 0, 0, 1)
        elif direction == Directions.Y:
            pad = (0, 1, 0, 0)
        else:
            raise ValueError("Invalid direction")
        return pad

    @staticmethod
    def compute(x, direction):
        _kernel = torch.tensor([-1, 1], dtype=torch.float32)
        x, init_shape = to_4D(x)
        return F.pad(F.conv2d(x, weight=Gradient.get_kernel(_kernel, direction).to(x.device), stride=1),
                     Gradient.get_pad(direction), mode="constant").view(*init_shape)


class GradientTranspose:
    T = Gradient

    def __new__(self, x, direction):
        return GradientTranspose.compute(x, direction)

    @staticmethod
    def get_pad(direction):
        if direction == Directions.X:
            pad = (0, 0, 1, 0)
        elif direction == Directions.Y:
            pad = (1, 0, 0, 0)
        else:
            raise ValueError("Invalid direction")
        return pad

    @staticmethod
    def compute(x, direction):
        _kernel = torch.tensor([1, -1], dtype=torch.float32)
        x, init_shape = to_4D(x)
        return F.conv2d(F.pad(x, GradientTranspose.get_pad(direction), mode="constant"),
                        weight=Gradient.get_kernel(_kernel, direction).to(x.device), stride=1).view(*init_shape)


Gradient.T = GradientTranspose
