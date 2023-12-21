import enum
from typing import Callable, Tuple

import torch
import torch.nn as nn

from ..utils import StrEnum, get_module_logger
from .jacobian import alpha_operator, sum_J_JT
from .power_iteration import (
    conjugate_gradient_smallest_ev,
    lanczos,
    lobpcg,
    power_method,
)

logger = get_module_logger(__name__)

from .utils import get_neuralnet_jacobian_ev


class PenalizationMethods(StrEnum):
    POWER = enum.auto()
    POWERNOALPHA = enum.auto()
    EVDECOMP = enum.auto()
    OPTPOWER = enum.auto()
    OPTPOWERNOALPHA = enum.auto()
    CG = enum.auto()  # Conjugate gradient method for eigenvalues
    LOBPCG = enum.auto()
    LANCZOS = enum.auto()


def penalization(lambda_min, eps, use_relu=True):
    if not use_relu:
        return torch.maximum(-1 * lambda_min, torch.zeros_like(lambda_min) - eps).max()
    else:
        return torch.relu(eps - lambda_min).max() ** 2


class MonotonyRegularization(nn.Module):
    def __init__(
        self,
        method: PenalizationMethods,
        eps: float,
        alpha: float = 10.0,
        max_iter: int = 200,
        power_iter_tol: float = 1e-5,
        eval_mode: bool = False,
        use_relu_penalization: bool = False,
    ):
        super().__init__()
        self.method = method
        if isinstance(self.method, str):
            self.method = PenalizationMethods(self.method)
        self.eps = eps
        self.alpha = alpha
        self.max_iters = max_iter
        self.is_eval = eval_mode
        self.power_iter_tol = power_iter_tol
        self.use_relu = use_relu_penalization
        self.iterates = None
        logger.debug(
            f"Using {self.method.name} for computing regularisation. Tolerance for PI: {self.power_iter_tol}"
        )

    def forward(
        self, model: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._monotonicity_penalization(model, x)

    def _penalization_fulljacobian(
        self, net: Callable, x: torch.Tensor, force_compute_batch=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Penalization based on the full jacobian.
        :param net: Neural network
        :param x: Input data
        :return: Penalization value and lambda min
        """
        if not force_compute_batch:
            assert x.shape[0] == 1, "Batch size must be 1 for full jacobian"
        all_ev = get_neuralnet_jacobian_ev(net, x, self.is_eval)
        if self.is_eval:
            all_ev.detach_()
        return (
            penalization(all_ev.min(), self.eps, use_relu=self.use_relu),
            all_ev.min().detach(),
        )

    def _penalization_powermethod(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Penalization based on the power method.
        :param net: Neural network
        :param x: Input data
        :return: Penalization value and lambda min
        """
        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return alpha_operator(x_new, y_new, u, self.alpha, self.is_eval)

        _, lambda_min, self.iterates = power_method(
            x_new,
            operator,
            self.max_iters,
            tol=self.power_iter_tol,
            save_iterates=True,
        )
        self.iterates = [self.alpha - ev for ev in self.iterates]
        lambda_min = self.alpha - lambda_min
        return (
            penalization(lambda_min, self.eps, self.use_relu),
            lambda_min.min().detach(),
        )

    def _penalization_powermethod_noalpha(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Penalization based on the power method. This method is optimized by not computing the full graph and not
        knowing the alpha parameter. Instead, we compute the graph only once and then use it to backpropagate.
        :param net: Neural network
        :param x: Input data
        :return: Penalization value and lambda min
        """
        # We ignore alpha this time, get lambda max then compute lambda min.
        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return sum_J_JT(x_new, y_new, u, self.is_eval)

        with torch.no_grad():
            _, lambda_max, _ = power_method(
                x_new,
                operator,
                self.max_iters,
                tol=self.power_iter_tol,
            )
            lambda_max = lambda_max.abs().max().item()
        logger.debug(f"Lambda max = {lambda_max}")
        if lambda_max < 0:
            logger.warning(
                "The lowest EV is bigger in module than the largest EV. Setting alpha to 0 in power method."
            )
            lambda_max = 0

        # Small hack to compute lambda min using this method.
        old_alpha = self.alpha
        self.alpha = lambda_max
        output = self._penalization_powermethod(net, x)
        self.alpha = old_alpha
        return output

    def _penalization_optpowermethod(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Penalization based on the power method. This method is optimized by not computing the full graph
        during the power method. Instead, we compute the graph only once and then use it to backpropagate.
        :param net: Neural network
        :param x: Input data
        :return: Penalization value and lambda min
        """
        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return alpha_operator(x_new, y_new, u, self.alpha, self.is_eval)

        with torch.no_grad():
            vectors, _, self.iterates = power_method(
                x_new,
                operator,
                self.max_iters,
                tol=self.power_iter_tol,
                save_iterates=True,
            )
            self.iterates = [self.alpha - ev for ev in self.iterates]
        vtOv = torch.sum(
            (vectors * operator(vectors)).view(vectors.shape[0], -1), dim=1
        )
        vtv = torch.sum((vectors * vectors).view(vectors.shape[0], -1), dim=1)
        rayleigh_coeff = vtOv / vtv

        lambda_min = (self.alpha - rayleigh_coeff).min()
        return (
            penalization(lambda_min, self.eps, self.use_relu),
            lambda_min.min().detach(),
        )

    def _penalization_optpowermethod_noalpha(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Penalization based on the power method. This method is optimized by not computing the full graph and not
        knowing the alpha parameter. Instead, we compute the graph only once and then use it to backpropagate.
        :param net: Neural network
        :param x: Input data
        :return: Penalization value and lambda min
        """
        # We ignore alpha this time, get lambda max then compute lambda min.
        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return sum_J_JT(x_new, y_new, u, self.is_eval)

        with torch.no_grad():
            _, lambda_max, _ = power_method(
                x_new,
                operator,
                self.max_iters,
                tol=self.power_iter_tol,
            )
            lambda_max = lambda_max.abs().max().item() * 1.1
        logger.debug(f"Lambda max = {lambda_max}")
        if lambda_max < 0:
            logger.warning(
                "The lowest EV is bigger in module than the largest EV. Setting alpha to 0 in power method."
            )
            lambda_max = 0

        # Small hack to compute lambda min using this method.
        old_alpha = self.alpha
        self.alpha = lambda_max
        output = self._penalization_optpowermethod(net, x)
        self.alpha = old_alpha
        return output

    def _penalization_cg(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return sum_J_JT(x_new, y_new, u, self.is_eval)

        with torch.no_grad():
            vectors, _, self.iterates = conjugate_gradient_smallest_ev(
                x_new,
                operator,
                max_iter=self.max_iters,
                tol=self.power_iter_tol,
                save_iterates=True,
            )

        vtOv = torch.sum(
            (vectors * operator(vectors)).view(vectors.shape[0], -1), dim=1
        )
        vtv = torch.sum((vectors * vectors).view(vectors.shape[0], -1), dim=1)
        rayleigh_coeff = vtOv / vtv

        lambda_min = rayleigh_coeff.min()
        return (
            penalization(lambda_min, self.eps, self.use_relu),
            lambda_min.min().detach(),
        )

    def _penalization_lobpcg(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We will perform the Lanczos algorithm to compute the lowest eigenvalue on each element of the batch.

        vectors = torch.zeros_like(x)
        self.iterates = []
        for idx in range(x.shape[0]):
            x_new = x[idx : idx + 1].clone()
            x_new.requires_grad_()
            y_new = net(x_new)

            def operator(u):
                return sum_J_JT(x_new, y_new, u, self.is_eval)

            with torch.no_grad():
                v, _, iterates = lobpcg(
                    x_new,
                    operator,
                    max_iter=self.max_iters,
                    tol=self.power_iter_tol,
                    save_iterates=True,
                )
                self.iterates.append(iterates)
            vectors[idx] = v.squeeze(0)
            print(_)

        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return sum_J_JT(x_new, y_new, u, self.is_eval)

        vtOv = torch.sum(
            (vectors * operator(vectors)).reshape(vectors.shape[0], -1), dim=1
        )
        vtv = torch.sum((vectors * vectors).reshape(vectors.shape[0], -1), dim=1)
        rayleigh_coeff = vtOv / vtv

        lambda_min = rayleigh_coeff.min()
        return (
            penalization(lambda_min, self.eps, self.use_relu),
            lambda_min.min().detach(),
        )

    def _penalization_lanczos(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We will perform the Lanczos algorithm to compute the lowest eigenvalue on each element of the batch.

        vectors = torch.zeros_like(x)
        self.iterates = []
        for idx in range(x.shape[0]):
            x_new = x[idx : idx + 1].clone()
            x_new.requires_grad_()
            y_new = net(x_new)

            def operator(u):
                return sum_J_JT(x_new, y_new, u, self.is_eval)

            with torch.no_grad():
                v, _, iterates = lanczos(
                    x_new,
                    operator,
                    max_iter=self.max_iters,
                    tol=self.power_iter_tol,
                    save_iterates=True,
                )
                self.iterates.append(iterates)
            vectors[idx] = v.squeeze(0)

        x_new = x.clone()
        x_new.requires_grad_()
        y_new = net(x_new)

        def operator(u):
            return sum_J_JT(x_new, y_new, u, self.is_eval)

        vtOv = torch.sum(
            (vectors * operator(vectors)).reshape(vectors.shape[0], -1), dim=1
        )
        vtv = torch.sum((vectors * vectors).reshape(vectors.shape[0], -1), dim=1)
        rayleigh_coeff = vtOv / vtv

        lambda_min = rayleigh_coeff.min()
        return (
            penalization(lambda_min, self.eps, self.use_relu),
            lambda_min.min().detach(),
        )

    def _monotonicity_penalization(
        self, net: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the monotony penalization for a given neural network and input.
        :param net: The sneural network to compute the penalization for.
        :param x: The input to compute the penalization for.
        """
        if self.method == PenalizationMethods.POWER:
            return self._penalization_powermethod(net, x)
        elif self.method == PenalizationMethods.POWERNOALPHA:
            return self._penalization_powermethod_noalpha(net, x)
        elif self.method == PenalizationMethods.EVDECOMP:
            return self._penalization_fulljacobian(net, x)
        elif self.method == PenalizationMethods.OPTPOWER:
            return self._penalization_optpowermethod(net, x)
        elif self.method == PenalizationMethods.OPTPOWERNOALPHA:
            return self._penalization_optpowermethod_noalpha(net, x)
        elif self.method == PenalizationMethods.CG:
            return self._penalization_cg(net, x)
        elif self.method == PenalizationMethods.LOBPCG:
            return self._penalization_lobpcg(net, x)
        elif self.method == PenalizationMethods.LANCZOS:
            return self._penalization_lanczos(net, x)
        else:
            raise ValueError(f"No such penalization method {self.method}")


class MonotonyRegularizationShift(MonotonyRegularization):
    def __init__(
        self,
        method: PenalizationMethods,
        eps: float,
        alpha: float = 10.0,
        max_iter: int = 200,
        power_iter_tol: float = 1e-5,
        eval_mode: bool = False,
        use_relu_penalization: bool = False,
    ):
        super().__init__(
            method,
            -1 + eps,
            alpha,
            max_iter,
            power_iter_tol,
            eval_mode,
            use_relu_penalization,
        )

    @staticmethod
    def shift_model(model):
        def two_net_minus_identity(x: torch.Tensor) -> torch.Tensor:
            return 2 * model(x) - x

        return two_net_minus_identity

    def forward(
        self, model: Callable, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._monotonicity_penalization(
            MonotonyRegularizationShift.shift_model(model), x
        )
