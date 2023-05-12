import torch

from ..base_classes import Fidelity
from ..imageOperators import BlurConvolution


class LpFidelity(Fidelity):
    """
    Implements the fidelity term of the form
    1/p * ||Hx - y||_p for p >= 1
    For p = 2, this is the L2 fidelity term.
    """

    def __init__(self, h: BlurConvolution, p: int = 2):
        self.H = h
        self.p = p

    def f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1.0 / self.p) * ((self.H @ x - y) ** self.p).sum()

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Hx = self.H @ x
        return self.H.T @ (torch.abs(Hx - y) ** (self.p - 1) * torch.sign(Hx - y))

    def autograd_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_ = x.detach().clone()
        x_.requires_grad_()
        self.f(x_, y).backward()
        return x_.grad


class L2Fidelity(LpFidelity):
    """
    Implements the L2 fidelity term
    Wraps the LpFidelity class with p = 2
    """

    def __init__(self, h: BlurConvolution):
        super().__init__(h, 2)
