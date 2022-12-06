import torch

from .Function import Function


class NoRegularisation(Function):
    @staticmethod
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x).sum()

    @staticmethod
    def grad(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
