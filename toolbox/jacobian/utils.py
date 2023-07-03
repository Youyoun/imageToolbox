import functools
from typing import Callable, Tuple

import torch

from .jacobian import alpha_operator, compute_jacobian, sum_J_JT
from .power_iteration import power_method

POWER_ITER_TOL = 1e-5


def get_neuralnet_jacobian_ev(
    net: Callable, x: torch.Tensor, is_eval: bool = False
) -> torch.Tensor:
    """
    Compute jacobian evaluated on x and then return all its eigenvalues (EV decomposition)
    :param net: Callable method, takes as input x and returns a vector of similar shape
    :param x: input vector
    :return: tensor of 1-dim containing all the eigenvalues (size = nelement(x))
    """
    J = compute_jacobian(net, x, is_eval=is_eval)
    if is_eval:
        J = J.detach()
    all_ev, _ = torch.linalg.eigh(1 / 2 * (J + J.transpose(1, 2)))
    return all_ev


def get_min_max_ev_neuralnet_fulljacobian(
    net: Callable, x: torch.Tensor, is_eval: bool = False
) -> Tuple[float, float]:
    """
    Small wrapper func that returns the max and min eigenvalues of a neural network evaluated on x
    by computing the full jacobian and its decompostion.
    :param net: Callable method, takes as input x and returns a vector of similar shape
    :param x: input vector
    :return: Tuple[\lambda_{min}, \lambda_{max}]
    """
    all_ev = get_neuralnet_jacobian_ev(net, x, is_eval=is_eval)
    return all_ev.min().item(), all_ev.max().item()


def get_lambda_min_or_max_poweriter(
    model: Callable, x: torch.Tensor, alpha, is_eval=False, biggest=False, n_iter=50
) -> torch.Tensor:
    """
    Small wrapper func that returns either the max or the min eigenvalues of a neural network evaluated on x
    using the power iteration method
    :param model: Callable method, takes as input x and returns a vector of similar shape
    :param x: input vector
    :param alpha: Float by which to shift the jacobian in order to compute the smallest EV
    :param is_eval: Whether to retain graph or drop it (For train-eval mode)
    :param biggest: return biggest if true, else return smallest
    :param n_iter: Maximum number of iterations in the power method
    :return: the smallest or biggest eigen values depeding on the parameters biggest
    """

    x_detached = x.clone().detach().flatten(start_dim=1)
    x_detached.requires_grad_()
    y_pred = model(x_detached.view(x.shape)).flatten(start_dim=1)
    if not biggest:
        A_dot_u = lambda u: alpha_operator(x_detached, y_pred, u, alpha, is_eval)
        return alpha - power_method(
            x_detached, A_dot_u, n_iter, tol=POWER_ITER_TOL, is_eval=is_eval
        )
    else:
        A_dot_u = lambda u: sum_J_JT(x_detached, y_pred, u, is_eval)
        return power_method(
            x_detached, A_dot_u, n_iter, tol=POWER_ITER_TOL, is_eval=is_eval
        )


def generate_new_prediction(
    net: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a new flattened input that requires grad, computes its image according to net, and then return
    both input and output (flattened version). Serves as entrypoint to the power iter method.
    :param net: Callable method, takes as input x and returns a vector of similar shape
    :param x: input vector (same shape as that required by net)
    :return: Tuple[flattened_input that requires grad, flattened_output]
    """
    x_detached = x.clone().detach().flatten(start_dim=1)
    x_detached.requires_grad_()
    y_pred = net(x_detached.view(x.shape)).flatten(start_dim=1)
    return x_detached, y_pred


def transform_contraint(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(
        net: Callable,
        x: torch.Tensor,
        eps: float = 0.00,
        alpha: float = None,
        max_iters: int = 300,
        power_iter_tol: float = 1e-5,
        is_eval: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        EPS = -1 + eps

        def two_net_minus_identity(x: torch.Tensor) -> torch.Tensor:
            return 2 * net(x) - x

        return func(
            two_net_minus_identity, x, EPS, alpha, max_iters, power_iter_tol, is_eval
        )

    return wrapper
