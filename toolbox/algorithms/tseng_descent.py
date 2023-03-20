from typing import Callable, Union, Dict

import numpy as np
import torch

from .proj_op import Identity
from ..base_classes import BasicSolver, FunctionNotDefinedError, Fidelity, Regularization, GenericFunction, ProximityOp
from ..metrics import MetricsDictionary, mean_absolute_error, compute_relative_difference, SNR
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class TsengOperator(Fidelity):
    """
    Operator for Tseng's gradient descent algorithm.
    :param fid: Fidelity function.
    :param reg: Regularization function.
    :param lambda_: Regularization parameter.
    """

    def __init__(self, fid: Fidelity, reg: Regularization, lambda_: float = 1.0):
        super().__init__()
        self.fid = fid
        self.reg = reg
        self.lambda_ = lambda_

    def f(self, x: torch.Tensor, y: torch.Tensor):
        """
        F(x) + \lambda * R(x)
        """
        return self.fid.f(x, y) + self.lambda_ * self.reg.f(x)

    def grad(self, x: torch.Tensor, y: torch.Tensor):
        """
        \nabla F(x) + \lambda * \nabla R(x)
        """
        return self.fid.grad(x, y) + self.lambda_ * self.reg.grad(x)


class TsengDescent(BasicSolver):
    """
    Tseng's gradient descent algorithm.
    """

    def __init__(self,
                 fidelity: Fidelity,
                 regularization: Regularization,
                 gamma: float,
                 lambda_: float = 1.0,
                 max_iter: int = 1000,
                 use_armijo: bool = True,
                 do_compute_metrics: bool = True,
                 indicator_fn: ProximityOp = Identity()):
        super().__init__()
        self.fidelity = fidelity
        self.regularization = regularization
        self.operator = TsengOperator(self.fidelity, self.regularization, lambda_)
        self.indicator = indicator_fn

        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.use_armijo = use_armijo
        self.do_compute_metrics = do_compute_metrics

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
            r_x = self.regularization.f(xk)
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

    def solve(self,
              input_vector: torch.Tensor,
              real_x: torch.Tensor = None, ) -> (torch.Tensor, MetricsDictionary):
        """
        Tseng's gradient descent algorithm.
        :param input_vector: Input vector.
        :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
        :return: Last iterate and metrics dictionary.
        """
        _tol = 1e-8
        logger.info("Running Tseng's gradient descent algorithm...")
        logger.debug(f"Parameters: gamma={self.gamma}, lambda={self.lambda_}, max_iter={self.max_iter}")
        logger.debug(f"Input vector shape: {input_vector.shape}")

        armijo = None
        gamma = self.gamma
        if self.use_armijo:
            armijo = GammaSearch(self.operator, sigma=self.gamma, gamma_min=1e-6, reset_each_search=False)

        xk_old = input_vector.clone()
        xk = xk_old
        self.metrics = MetricsDictionary()
        for step in range(self.max_iter):
            if armijo is not None:
                gamma = armijo.run_search_get_gamma(xk, y=input_vector)
            # Update (one step)
            ak = self.operator.grad(xk, y=input_vector)
            zk = self.indicator.prox(xk - gamma * ak)
            xk = self.indicator.prox(zk - gamma * (self.operator.grad(zk, y=input_vector) - ak))

            # Compute metrics
            if self.do_compute_metrics:
                self.metrics.add(self.compute_metrics(xk, xk_old, input_vector, real_x))
            if compute_relative_difference(xk, xk_old, input_vector) <= _tol:
                print(f"Descent reached tolerance={_tol} at step {step}")
                break
            xk_old = xk.clone()
        return xk, self.metrics


def tseng_gradient_descent(input_vector: Union[np.ndarray, torch.Tensor],
                           fidelity_gradient: Callable,
                           regularization_gradient: Callable,
                           gamma: float,
                           lambda_: float,
                           use_armijo: bool = False,
                           max_iter: int = 1000,
                           real_x: np.ndarray = None,
                           regularization_function: Callable = None,
                           fidelity_function: Callable = None,
                           do_compute_metrics: bool = True,
                           indicator: ProximityOp = Identity()) -> (Union[np.ndarray, torch.Tensor], MetricsDictionary):
    """
    Tseng's gradient descent algorithm.
    :param input_vector: Input vector.
    :param fidelity_gradient: Gradient of the fidelity term. Must have the signature grad_xF(x, y) where x is the input vector
        and y is the fidelity term.
    :param regularization_gradient: Gradient of the regularization term. Must have the signature grad_xR(x) where x is the input vector.
    :param gamma: Step size.
    :param lambda_: Regularization parameter.
    :param use_armijo: If True, use Armijo rule to find the step size. gamma is used as the initial step size.
    :param max_iter: Maximum number of iterations.
    :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
    :param regularization_function: Regularization function. Must have the signature regul_function(x, gamma)
        where x is the input vector and gamma is the step size.
    :param fidelity_function: Fidelity function. Must have the signature fidelity_function(x, y)
    :param do_compute_metrics: If True, compute metrics.
    :param indicator: Indicator function (Identity if no constraint on interval).
    :return: Last iterate and metrics dictionary.
    """
    fidelity = GenericFunction(fidelity_function, fidelity_gradient, None)
    regul = GenericFunction(regularization_function, regularization_gradient, None)

    solver = TsengDescent(fidelity, regul, gamma, lambda_, max_iter, use_armijo, do_compute_metrics, indicator)
    return solver.solve(input_vector, real_x)


class GammaSearch:
    def __init__(self,
                 operator: TsengOperator,
                 theta: float = 0.9,
                 beta: float = 0.8,
                 sigma: float = 1,
                 gamma_min: float = 1e-4,
                 reset_each_search: bool = False,
                 projection: ProximityOp = None):
        """
        Gamma search for the Tseng's gradient descent algorithm. The search is based on the Armijo rule. The search is
        performed by decreasing the step size until the Armijo rule is satisfied. The step size is decreased by a factor
        of beta at each iteration. The search stops when the step size is smaller than gamma_min. The search is reset
        after each iteration if reset_each_search is True.
        :param operator: Operator to apply to get the gradient of the fidelity term and the regularization term.
        :param theta: Armijo rule parameter (step size decrease threshold).
        :param beta: Armijo rule parameter (step size decrease factor).
        :param sigma: Armijo rule parameter (initial step size).
        :param gamma_min: Minimum step size.
        :param reset_each_search: If True, the search is reset after each iteration.
        :param projection: Projection operator to apply to the iterate after computing the gradient step.
        """
        self.theta = theta
        self.beta = beta
        self.sigma = sigma
        self.gamma_min = gamma_min
        logger.debug(f"Armijo rule parameters set to θ={theta}, β={beta}, σ={sigma}")

        self.operator = operator
        self.power = 0.0
        self.projection = projection
        if self.projection is None:
            self.projection = Identity()
        self.reset_each_search = reset_each_search
        logger.debug(f"Armijo is {'not' if not self.reset_each_search else ''} reset after each iteration.")

    @property
    def gamma(self):
        """
        Compute the step size. The step size is computed as sigma * beta ** power.
        :return: Step size.
        """
        return self.sigma * self.beta ** self.power

    def reset_power(self):
        """
        Reset the power of the step size.
        :return: None
        """
        self.power = 0

    def run_search(self, x, y):
        """
        Run the Armijo search. The search is performed by decreasing the step size until the Armijo rule is satisfied.
        The step size is decreased by a factor of beta at each iteration. The search stops when the step size is smaller
        than gamma_min. The search is reset after each iteration if reset_each_search is True.
        Armijo search: \gamma ||A(Z_C(x_k, y)) - A(x_k)|| \leq \theta || Z_C(x_k, \gamma) - x_k||
        :param x: Current iterate.
        :param y: Input vector for the fidelity term.
        :return: None (To get the gamma, either use the gamma property or the run_search_get_gamma method).
        """
        if self.reset_each_search:
            self.reset_power()

        while True:
            # A(x_k) = \nabla F(x_k, y) + \lambda \nabla R(x_k)
            nabla_f = self.operator.grad(x, y)

            # proj_C(x_k - \gamma A(x_k)
            Zc = self.projection.prox(x - self.gamma * nabla_f)

            # \gamma ||A(Z_C(x_k, y)) - A(x_k)||
            diff_op = self.gamma * torch.linalg.norm((self.operator.grad(Zc, y) - nabla_f).flatten(), ord=2)

            # \theta || Z_C(x_k, \gamma) - x_k||
            diff_x = self.theta * torch.linalg.norm((Zc - x).flatten(), ord=2)

            if diff_op <= diff_x:
                break
            else:
                self.power += 1
                if self.gamma < self.gamma_min:
                    logger.warning(f"Gamma search stopped at gamma={self.gamma} < gamma_min={self.gamma_min}.")
                    break

    def run_search_get_gamma(self, x, y):
        """
        Run the Armijo search and return the step size.
        :param x: Current iterate.
        :param y: Input vector for the fidelity term.
        :return: Step size.
        """
        self.run_search(x, y)
        return self.gamma
