from typing import Callable

import numpy as np

from ..metrics import MetricsDictionary, mean_absolute_error, compute_relative_difference, SNR
from ..utils import get_module_logger

logger = get_module_logger(__name__)


def tseng_gradient_descent(input_vector: np.ndarray,
                           grad_xF: Callable,
                           grad_xR: Callable,
                           gamma: float,
                           lambda_: float,
                           use_armijo: bool = False,
                           max_iter: int = 1000,
                           real_x: np.ndarray = None,
                           regul_function: Callable = None,
                           fidelity_function: Callable = None,
                           do_compute_metrics: bool = True) -> (np.ndarray, MetricsDictionary):
    """
    Tseng's gradient descent algorithm.
    :param input_vector: Input vector.
    :param grad_xF: Gradient of the fidelity term. Must have the signature grad_xF(x, y) where x is the input vector
        and y is the fidelity term.
    :param grad_xR: Gradient of the regularization term. Must have the signature grad_xR(x) where x is the input vector.
    :param gamma: Step size.
    :param lambda_: Regularization parameter.
    :param use_armijo: If True, use Armijo rule to find the step size. gamma is used as the initial step size.
    :param max_iter: Maximum number of iterations.
    :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
    :param regul_function: Regularization function. Must have the signature regul_function(x, gamma)
        where x is the input vector and gamma is the step size.
    :param fidelity_function: Fidelity function. Must have the signature fidelity_function(x, y)
    :param do_compute_metrics: If True, compute metrics.
    :return: Last iterate and metrics dictionary.
    """
    _tol = 1e-8
    logger.info("Running Tseng's gradient descent algorithm...")
    logger.debug(f"Parameters: gamma={gamma}, lambda={lambda_}, use_armijo={use_armijo}, max_iter={max_iter}")
    logger.debug(f"Input vector shape: {input_vector.shape}")
    logger.debug("Computing metrics..." if do_compute_metrics else "Not computing metrics...")

    def operator(x):
        return grad_xF(x, y=input_vector) + lambda_ * grad_xR(x)

    armijo = None
    if use_armijo:
        armijo = GammaSearch(operator, sigma=gamma, gamma_min=1e-6, reset_each_search=False)

    xk_old = input_vector.copy()
    xk = xk_old
    metrics = MetricsDictionary()
    for step in range(max_iter):
        if armijo is not None:
            gamma = armijo.run_search_get_gamma(xk, y=input_vector)
        # Update (one step)
        ak = operator(xk)
        zk = xk - gamma * ak
        xk = zk - gamma * (operator(zk) - ak)

        # Compute metrics
        if not do_compute_metrics:
            continue
        else:
            R_x, F_x = 0, 0
            if regul_function is not None:
                R_x = regul_function(xk, lambda_)
            if fidelity_function is not None:
                F_x = fidelity_function(xk, input_vector)
            metrics.add(
                {
                    "||x_{k+1} - x_k||_2 / ||y||_2": compute_relative_difference(xk, xk_old, input_vector),
                    "||x_{k+1} - x||_1": mean_absolute_error(xk, real_x) if real_x is not None else 0,
                    "R(x_{k+1})": R_x,
                    "F(x_{k+1})": F_x,
                    "F(x_{k+1}) + \\lambda R(x_{k+1})": F_x + lambda_ * R_x,
                    "SNR": SNR(xk),
                }
            )
            if metrics["||x_{k+1} - x_k||_2 / ||y||_2"][-1] <= _tol:
                print(f"Descent reached tolerance={_tol} at step {step}")
                break
            xk_old = xk.copy()
    return xk, metrics


class GammaSearch:
    def __init__(self,
                 operator: Callable,
                 theta: float = 0.9,
                 beta: float = 0.8,
                 sigma: float = 1,
                 gamma_min: float = 1e-4,
                 reset_each_search: bool = False,
                 projection: Callable = None):
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

        self.grad_op = operator
        self.power = 0.0
        self.projection = projection
        if self.projection is None:
            self.projection = lambda x: x
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
            nabla_f = self.grad_op(x)

            # proj_C(x_k - \gamma A(x_k)
            Zc = self.projection(x - self.gamma * nabla_f)

            # \gamma ||A(Z_C(x_k, y)) - A(x_k)||
            diff_op = self.gamma * np.linalg.norm((self.grad_op(Zc) - nabla_f).flatten(), ord=2)

            # \theta || Z_C(x_k, \gamma) - x_k||
            diff_x = self.theta * np.linalg.norm((Zc - x).flatten(), ord=2)

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
