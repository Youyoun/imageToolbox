from typing import Callable, Tuple

import torch

from .jacobian import JTJu
from .power_iteration import power_method


def penalization(lambda_max, eps, use_relu=True):
    if not use_relu:
        return torch.maximum(lambda_max, torch.ones_like(lambda_max) - eps).max()
    else:
        return torch.relu(lambda_max - 1 - eps).max() ** 2


def nonexpansive_penalization(net: Callable,
                              x: torch.Tensor,
                              eps: float,
                              alpha: float = None,
                              n_iters: int = 200,
                              power_iter_tol: float = 1e-5,
                              is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the nonexpansive penalization for a given neural network and input.
    :param net: The neural network to compute the penalization for.
    :param x: The input to compute the penalization for.
    :param eps: The epsilon value for the penalization.
    :param alpha: Useless parameter for this method.
    :param n_iters: The number of iterations to use for the power method.
    :param power_iter_tol: The tolerance for the power method.
    :param is_eval: Whether to use the eval mode.
    """
    x_new = x.clone()
    x_new.requires_grad_()
    y_new = 2 * net(x_new) - x_new  # y = 2T(x) - x

    def operator(u):
        return JTJu(x_new, y_new, u, is_eval)

    u = torch.randn_like(x_new)
    print((operator(u) - 2 * JTJu(x_new, net(x_new), u, is_eval)).abs().max())

    with torch.no_grad():
        lambda_max = power_method(x_new, operator, n_iters, tol=power_iter_tol, is_eval=is_eval).abs()
    return penalization(lambda_max, eps, use_relu=False), lambda_max.max().detach()
