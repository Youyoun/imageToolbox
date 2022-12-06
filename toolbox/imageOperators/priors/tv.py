import torch

from src.Function import Function
from src.im_gradient.image_gradient import Gradient, Directions
from . import logger

EPSILON_TV = 1e-6
logger.debug(f"Epsilon for smoothed total variation is set to {EPSILON_TV}")


class TotalVariation(Function):
    def __init__(self):
        pass

    def f(self, x):
        return torch.sum(
            torch.sqrt(Gradient(x, Directions.X) ** 2 + Gradient(x, Directions.Y) ** 2 + EPSILON_TV))

    def grad(self, x):
        grad_x = Gradient(x, Directions.X)
        grad_y = Gradient(x, Directions.Y)
        tv = torch.sqrt(grad_x * grad_x + grad_y * grad_y + EPSILON_TV)
        grad_x /= tv
        grad_y /= tv
        return Gradient.T(grad_x, Directions.X) + Gradient.T(grad_y, Directions.Y)

    def autograd_f(self, x):
        with torch.enable_grad():
            x_ = x.detach().clone()
            x_.requires_grad_()
            self.f(x_).backward()
            grad = x_.grad.detach().clone()
        return grad
