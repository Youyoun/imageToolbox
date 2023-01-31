from typing import Callable, Tuple

import torch

from .power_iteration import power_method
from .jacobian import JTJu


def nonexpansive_penalization(net: Callable,
                              x: torch.Tensor,
                              eps: float,
                              n_iters: int = 200,
                              power_iter_tol: float = 1e-5,
                              is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the nonexpansive penalization for a given neural network and input.
    :param net: The neural network to compute the penalization for.
    :param x: The input to compute the penalization for.
    :param eps: The epsilon value for the penalization.
    :param n_iters: The number of iterations to use for the power method.
    :param power_iter_tol: The tolerance for the power method.
    :param is_eval: Whether to use the eval mode.
    """
    x_new = x.clone()
    x_new.requires_grad_()
    y_new = net(x_new)

    def operator(u):
        return JTJu(x_new, y_new, u, is_eval)

    lambda_max = power_method(x_new, operator, n_iters, tol=power_iter_tol, is_eval=is_eval)
    return torch.relu(eps - lambda_max).max() ** 2, lambda_max.max().detach()
