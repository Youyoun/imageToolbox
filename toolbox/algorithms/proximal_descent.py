from typing import Callable, Dict

import torch
import tqdm

from ..base_classes import BasicSolver, FunctionNotDefinedError, GenericFunction, Fidelity, ProximityOp
from ..metrics import MetricsDictionary, mean_absolute_error, compute_relative_difference, SNR


class ProximalDescent(BasicSolver):
    """
    Proximal descent algorithm. See https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning
    :param fidelity: Fidelity function. Must have the signature fidelity_function(x, y)
    :param prox: Regularization function. Must have the signature regul_function(x, gamma)
    :param gamma: Step size
    :param lambda_: Regularization parameter
    :param max_iter: Maximum number of iterations
    :param do_compute_metrics: If True, compute metrics
    """

    def __init__(self,
                 fidelity: Fidelity,
                 prox: ProximityOp,
                 gamma: float,
                 lambda_: float,
                 max_iter: int = 1000,
                 do_compute_metrics: bool = True,
                 device="cpu"):
        super().__init__()
        self.fidelity = fidelity
        self.prox = prox
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.do_compute_metrics = do_compute_metrics
        self.device = device

        self.metrics = None
        if self.do_compute_metrics:
            self.metrics = MetricsDictionary()

    def compute_metrics(self,
                        xk: torch.Tensor,
                        xk_old: torch.Tensor,
                        input_vector: torch.Tensor,
                        real_x: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics.
        :param xk: Current iterate.
        :param xk_old: Previous iterate.
        :param input_vector: Input vector.
        :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
        """
        try:
            r_x = self.prox.f(xk, self.lambda_)
        except FunctionNotDefinedError:
            r_x = 0.0
        try:
            f_x = self.fidelity.f(xk, y=input_vector)
        except FunctionNotDefinedError:
            f_x = 0.0

        return {
            "||x_{k+1} - x_k||_2 / ||y||_2": compute_relative_difference(xk, xk_old, input_vector),
            "||x_{k+1} - x||_1": mean_absolute_error(xk, real_x) if real_x is not None else 0,
            "R(x_{k+1})": r_x,
            "F(x_{k+1})": f_x,
            "F(x_{k+1}) + \\lambda R(x_{k+1})": f_x + self.lambda_ * r_x,
            "SNR": SNR(xk),
        }

    def solve(self, input_vector: torch.Tensor, real_x: torch.Tensor = None) -> (torch.Tensor, MetricsDictionary):
        """
        Solve the optimization problem.
        :param input_vector: Input vector.
        :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
        :return: Last iterate and metrics dictionary.
        """
        _tol = 1e-8

        input_vector = input_vector.to(self.device)
        xk_old = input_vector.clone()
        xk = xk_old
        if self.do_compute_metrics:
            self.metrics = MetricsDictionary()
        for step in tqdm.tqdm(range(self.max_iter)):
            # Prox_{\gamma R} (x_k - \gamma \nabla F(x_k))
            xk = self.prox.prox(xk - self.gamma * self.fidelity.grad(xk, y=input_vector),
                                self.gamma * self.lambda_)
            if self.do_compute_metrics:
                self.metrics.add(self.compute_metrics(xk.cpu(), xk_old.cpu(), input_vector.cpu(), real_x.cpu()))
            if compute_relative_difference(xk.cpu(), xk_old.cpu(), input_vector.cpu()) <= _tol:
                print(f"Descent reached tolerance={_tol} at step {step}")
                break
            xk_old = xk.clone()
        return xk.cpu(), self.metrics


def prox_descent(input_vector: torch.Tensor,
                 fidelity_grad: Callable,
                 regul_prox: Callable,
                 gamma: float,
                 lambda_: float = 1.0,
                 max_iter: int = 1000,
                 real_x: torch.Tensor = None,
                 regul_function: Callable = None,
                 fidelity_function: Callable = None) -> (torch.Tensor, MetricsDictionary):
    """
    Wrapper for the ProximalDescent class.
    :param input_vector: Input vector.
    :param fidelity_grad: Gradient of the fidelity term. Must have the signature grad_xF(x, y) where x is the input vector
        and y is the fidelity term.
    :param regul_prox: Proximal operator of the regularization term. Must have the signature prox_gammaR(x, gamma)
        where x is the input vector and gamma is the step size.
    :param gamma: Step size.
    :param lambda_: Regularization parameter.
    :param max_iter: Maximum number of iterations.
    :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
    :param regul_function: Regularization function. Must have the signature regul_function(x, gamma)
        where x is the input vector and gamma is the step size.
    :param fidelity_function: Fidelity function. Must have the signature fidelity_function(x, y)
    :return: Last iterate and metrics dictionary.
    """
    fidelity = GenericFunction(fidelity_function, fidelity_grad, None)
    regul = GenericFunction(regul_function, None, regul_prox)

    solver = ProximalDescent(fidelity, regul, gamma, lambda_, max_iter)
    return solver.solve(input_vector, real_x)
