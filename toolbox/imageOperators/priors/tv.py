import torch

from .Function import Function
from ..im_gradient import Gradient, Directions
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class SmoothTotalVariation(Function):
    def __init__(self, smoothing_epsilon: float = 1e-6):
        self.epsilon_tv = smoothing_epsilon
        logger.info(f"Epsilon for smoothed total variation is set to {self.epsilon_tv}")

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            torch.sqrt(Gradient(x, Directions.X) ** 2 + Gradient(x, Directions.Y) ** 2 + self.epsilon_tv))

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = Gradient(x, Directions.X)
        grad_y = Gradient(x, Directions.Y)
        tv = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.epsilon_tv)
        grad_x /= tv
        grad_y /= tv
        return Gradient.T(grad_x, Directions.X) + Gradient.T(grad_y, Directions.Y)

    def autograd_f(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_ = x.detach().clone()
            x_.requires_grad_()
            self.f(x_).backward()
            grad = x_.grad.detach().clone()
        return grad
