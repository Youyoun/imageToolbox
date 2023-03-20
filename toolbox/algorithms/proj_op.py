import torch

from ..base_classes import ProximityOp


class Identity(ProximityOp):
    """
    Identity operator
    """

    def __init__(self):
        pass

    def f(self, x: torch.Tensor, tau: float = None) -> float:
        return 0.0

    def prox(self, x: torch.Tensor, tau: float = None) -> torch.Tensor:
        return x


class Indicator(ProximityOp):
    """
    Indicator function of a convex set
    """

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def f(self, x: torch.Tensor, tau: float = None) -> float:
        if (x >= self.lower_bound).all() and (x <= self.upper_bound).all():
            return 0
        return torch.inf

    def prox(self, x: torch.Tensor, tau: float = None) -> torch.Tensor:
        return torch.max(torch.tensor(self.lower_bound), torch.min(x, torch.tensor(self.upper_bound)))
