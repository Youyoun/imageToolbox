import enum
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn

from .jacobian import alpha_operator, sum_J_JT
from .power_iteration import power_method
from ..imageOperators.utils import get_module_logger, StrEnum

logger = get_module_logger(__name__)

from .utils import generate_new_prediction, get_neuralnet_jacobian_ev


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
        return penalization(model, x, self.eps, self.alpha, self.method, self.max_iter, self.power_iter_tol, self.eval)


def penalization_fulljacobian(net: Callable,
                              x: torch.Tensor,
                              eps: float = 0.00,
                              alpha: float = None,
                              max_iters: int = 200,
                              power_iter_tol: float = 1e-5,
                              is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    all_ev = get_neuralnet_jacobian_ev(net, x)
    if is_eval:
        all_ev.detach_()
    return torch.max(torch.max(eps - all_ev), torch.zeros(1, device=x.device)) ** 2, all_ev.detach()


def penalization_powermethod(net: Callable,
                             x: torch.Tensor,
                             eps: float = 0.00,
                             alpha: float = 10.0,
                             max_iters: int = 300,
                             power_iter_tol: float = 1e-5,
                             is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    x_new, y_new = generate_new_prediction(net, x)

    def operator(u):
        return alpha_operator(x_new, y_new, u, alpha, is_eval)

    lambda_min = alpha - power_method(x_new, operator, max_iters, tol=power_iter_tol, is_eval=is_eval)
    return torch.max(eps - lambda_min, torch.zeros_like(lambda_min)) ** 2, lambda_min


def penalization_optpowermethod(net: Callable,
                                x: torch.Tensor,
                                eps: float = 0.00,
                                alpha: float = 10.0,
                                max_iters: int = 300,
                                power_iter_tol: float = 1e-5,
                                is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    x_new, y_new = generate_new_prediction(net, x)

    def operator(u):
        return alpha_operator(x_new, y_new, u, alpha, is_eval)

    vectors, _ = power_method(x_new, operator, max_iters, tol=power_iter_tol, is_eval=True, return_vector=True)
    rayleigh_coeff = alpha - torch.sum(vectors * operator(vectors), dim=1) / torch.sum(vectors * vectors, dim=1)
    lambda_min = rayleigh_coeff.min()
    return torch.max(eps - lambda_min, torch.zeros_like(lambda_min)) ** 2, lambda_min


def penalization_optpowermethod_noalpha(net: Callable,
                                        x: torch.Tensor,
                                        eps: float = 0.00,
                                        alpha: float = None,
                                        max_iters: int = 300,
                                        power_iter_tol: float = 1e-5,
                                        is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # We ignore alpha this time, get lambda max then compute lambda min.
    x_new, y_new = generate_new_prediction(net, x)

    def operator(u):
        return sum_J_JT(x_new, y_new, u, is_eval)

    lambda_max = power_method(x_new, operator, max_iters, tol=power_iter_tol, is_eval=True)
    logger.debug(f"Lambda max = {lambda_max}")
    if lambda_max < 0:
        logger.warning("The lowest EV is bigger in module than the largest EV. Setting alpha to 0 in power method.")
        lambda_max = 0
    return penalization_optpowermethod(net, x, alpha=lambda_max, max_iters=max_iters, eps=eps, is_eval=is_eval)


def penalization(net: Callable,
                 x: torch.Tensor,
                 eps: float,
                 alpha: float,
                 method: Union[str, PenalizationMethods],
                 n_iters: int = 300,
                 power_iter_tol: float = 1e-5,
                 is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug(f"Using {method.name} for computing regularisation. Tolerance for PI: {power_iter_tol}")
    if method == PenalizationMethods.POWER:
        return penalization_powermethod(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    elif method == PenalizationMethods.EVDECOMP:
        return penalization_fulljacobian(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    elif method == PenalizationMethods.OPTPOWER:
        return penalization_optpowermethod(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    elif method == PenalizationMethods.OPTPOWERNOALPHA:
        return penalization_optpowermethod_noalpha(net, x, eps, alpha, n_iters, power_iter_tol, is_eval)
    else:
        raise ValueError(f"No such penalization method {method}")
