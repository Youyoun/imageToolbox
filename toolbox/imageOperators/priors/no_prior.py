import torch

from ...base_classes import Regularization


class NoRegularisation(Regularization):
    @staticmethod
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x).sum()

    @staticmethod
    def grad(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
