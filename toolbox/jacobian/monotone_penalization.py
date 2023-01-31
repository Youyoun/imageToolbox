import enum
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn

from .jacobian import alpha_operator, sum_J_JT
from .power_iteration import power_method
from ..utils import get_module_logger, StrEnum

logger = get_module_logger(__name__)

from .utils import get_neuralnet_jacobian_ev


class PenalizationMethods(StrEnum):
    POWER = enum.auto()
    EVDECOMP = enum.auto()
    OPTPOWER = enum.auto()
    OPTPOWERNOALPHA = enum.auto()


class MonotonyRegularization(nn.Module):
    def __init__(self,
                 method: PenalizationMethods,
                 eps: float,
                 alpha: float = 10.0,
                 max_iter: int = 200,
                 power_iter_tolerance: float = 1e-5,
                 eval_mode: bool = False):
        super().__init__()
        self.method = method
        self.eps = eps
        self.alpha = alpha
        self.max_iter = max_iter
        self.eval = eval_mode
        self.power_iter_tol = power_iter_tolerance

    def forward(self, model: Callable, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return monotony_penalization(model, x, self.eps, self.alpha, self.method, self.max_iter, self.power_iter_tol,
                                     self.eval)


def monotone_penalization_fulljacobian(net: Callable,
                                       x: torch.Tensor,
                                       eps: float = 0.00,
                                       alpha: float = None,
                                       max_iters: int = 200,
                                       power_iter_tol: float = 1e-5,
                                       is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalization based on the full jacobian.
    :param net: Neural network
    :param x: Input data
    :param eps: Epsilon for the penalization
    :param alpha: Useless parameter for this method
    :param max_iters: Useless parameter for this method
    :param power_iter_tol: Useless parameter for this method
    :param is_eval: If true, the network is set to eval mode
    :return: Penalization value and lambda min
    """
    all_ev = get_neuralnet_jacobian_ev(net, x)
    if is_eval:
        all_ev.detach_()
    return torch.relu(eps - all_ev).max() ** 2, all_ev.min().detach()


def monotone_penalization_powermethod(net: Callable,
                                      x: torch.Tensor,
                                      eps: float = 0.00,
                                      alpha: float = 10.0,
                                      max_iters: int = 300,
                                      power_iter_tol: float = 1e-5,
                                      is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalization based on the power method.
    :param net: Neural network
    :param x: Input data
    :param eps: Epsilon for the penalization
    :param alpha: Alpha parameter for the power method
    :param max_iters: Maximum number of iterations for the power method
    :param power_iter_tol: Tolerance for the power method
    :param is_eval: If true, the network is set to eval mode
    :return: Penalization value and lambda min
    """
    x_new = x.clone()
    x_new.requires_grad_()
    y_new = net(x_new)

    def operator(u):
        return alpha_operator(x_new, y_new, u, alpha, is_eval)

    lambda_min = alpha - power_method(x_new, operator, max_iters, tol=power_iter_tol, is_eval=is_eval)
    return torch.relu(eps - lambda_min).max() ** 2, lambda_min.min().detach()


def monotone_penalization_optpowermethod(net: Callable,
                                         x: torch.Tensor,
                                         eps: float = 0.00,
                                         alpha: float = 10.0,
                                         max_iters: int = 300,
                                         power_iter_tol: float = 1e-5,
                                         is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalization based on the power method. This method is optimized by not computing the full graph
    during the power method. Instead, we compute the graph only once and then use it to backpropagate.
    :param net: Neural network
    :param x: Input data
    :param eps: Epsilon for the penalization
    :param alpha: Alpha parameter for the power method
    :param max_iters: Maximum number of iterations for the power method
    :param power_iter_tol: Tolerance for the power method
    :param is_eval: If true, the network is set to eval mode
    :return: Penalization value and lambda min
    """
    x_new = x.clone()
    x_new.requires_grad_()
    y_new = net(x_new)

    def operator(u):
        return alpha_operator(x_new, y_new, u, alpha, is_eval)

    vectors, _ = power_method(x_new, operator, max_iters, tol=power_iter_tol, is_eval=True, return_vector=True)
    rayleigh_coeff = alpha - torch.sum(vectors * operator(vectors), dim=1) / torch.sum(vectors * vectors, dim=1)
    lambda_min = rayleigh_coeff.min()
    return torch.relu(eps - lambda_min) ** 2, lambda_min.min().detach()


def monotone_penalization_optpowermethod_noalpha(net: Callable,
                                                 x: torch.Tensor,
                                                 eps: float = 0.00,
                                                 alpha: float = None,
                                                 max_iters: int = 300,
                                                 power_iter_tol: float = 1e-5,
                                                 is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalization based on the power method. This method is optimized by not computing the full graph and not
    knowing the alpha parameter. Instead, we compute the graph only once and then use it to backpropagate.
    :param net: Neural network
    :param x: Input data
    :param eps: Epsilon for the penalization
    :param alpha: Useless parameter for this method
    :param max_iters: Maximum number of iterations for the power method
    :param power_iter_tol: Tolerance for the power method
    :param is_eval: If true, the network is set to eval mode
    :return: Penalization value and lambda min
    """
    # We ignore alpha this time, get lambda max then compute lambda min.
    x_new = x.clone()
    x_new.requires_grad_()
    y_new = net(x_new)

    def operator(u):
        return sum_J_JT(x_new, y_new, u, is_eval)

    lambda_max = power_method(x_new, operator, max_iters, tol=power_iter_tol, is_eval=True).max().item()
    logger.debug(f"Lambda max = {lambda_max}")
    if lambda_max < 0:
        logger.warning("The lowest EV is bigger in module than the largest EV. Setting alpha to 0 in power method.")
        lambda_max = 0
    return monotone_penalization_optpowermethod(net, x, alpha=lambda_max, max_iters=max_iters, eps=eps, is_eval=is_eval)


def monotony_penalization(net: Callable,
                          x: torch.Tensor,
                          eps: float,
                          alpha: float,
                          method: Union[str, PenalizationMethods],
                          n_iters: int = 300,
                          power_iter_tol: float = 1e-5,
                          is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the monotony penalization for a given neural network and input.
    :param net: The neural network to compute the penalization for.
    :param x: The input to compute the penalization for.
    :param eps: The epsilon value for the penalization.
    :param alpha: The alpha value for the penalization.
    :param method: The method to use for computing the penalization.
    :param n_iters: The number of iterations to use for the power method.
    :param power_iter_tol: The tolerance for the power method.
    :param is_eval: Whether to use the eval mode.
    """
    if isinstance(method, str):
        method = PenalizationMethods(method)
    logger.debug(f"Using {method.name} for computing regularisation. Tolerance for PI: {power_iter_tol}")
    if method == PenalizationMethods.POWER:
        return monotone_penalization_powermethod(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    elif method == PenalizationMethods.EVDECOMP:
        return monotone_penalization_fulljacobian(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    elif method == PenalizationMethods.OPTPOWER:
        return monotone_penalization_optpowermethod(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    elif method == PenalizationMethods.OPTPOWERNOALPHA:
        return monotone_penalization_optpowermethod_noalpha(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    else:
        raise ValueError(f"No such penalization method {method}")


