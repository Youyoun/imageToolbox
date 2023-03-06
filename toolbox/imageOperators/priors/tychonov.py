import torch

from ...base_classes import Regularization
from ...base_classes.basic_linear_operator import Operator


class Tychonov(Regularization):
    def __init__(self, op: Operator, mean: float = 0.0):
        self.L = op
        self.mean = mean

    def f(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return (self.L @ (x - self.mean)).pow(2).sum([1, 2, 3])  # Leave Batchsize
        else:
            return (self.L @ (x - self.mean)).pow(2).sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.L.T @ (self.L @ (x - self.mean)) * 2

    def autograd(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x.detach().clone()
        x_.requires_grad_()
        self.f(x_).backward()
        return x_.grad
