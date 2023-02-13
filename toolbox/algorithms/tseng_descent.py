from typing import Callable

import numpy as np

from ..metrics import MetricsDictionary, mean_absolute_error, compute_relative_difference, SNR


def tseng_gradient_descent(input_vector: np.ndarray,
                           grad_xF: Callable,
                           grad_xR: Callable,
                           gamma: float,
                           lambda_: float,
                           max_iter: int = 1000,
                           real_x: np.ndarray = None,
                           regul_function: Callable = None,
                           fidelity_function: Callable = None) -> (np.ndarray, MetricsDictionary):
    """
    Tseng's gradient descent algorithm.
    :param input_vector: Input vector.
    :param grad_xF: Gradient of the fidelity term. Must have the signature grad_xF(x, y) where x is the input vector
        and y is the fidelity term.
    :param grad_xR: Gradient of the regularization term. Must have the signature grad_xR(x) where x is the input vector.
    :param gamma: Step size.
    :param lambda_: Regularization parameter.
    :param max_iter: Maximum number of iterations.
    :param real_x: Real vector to compute the L1 distance between the current iterate and the real vector.
    :param regul_function: Regularization function. Must have the signature regul_function(x, gamma)
        where x is the input vector and gamma is the step size.
    :param fidelity_function: Fidelity function. Must have the signature fidelity_function(x, y)
    :return: Last iterate and metrics dictionary.
    """
    _tol = 1e-8

    xk_old = input_vector.copy()
    xk = xk_old
    metrics = MetricsDictionary()
    for step in range(max_iter):
        # Update (one step)
        ak = grad_xF(xk, y=input_vector) + lambda_ * grad_xR(xk)
        zk = xk - gamma * ak
        xk = zk - gamma * (grad_xF(zk, y=input_vector) + lambda_ * grad_xR(zk) - ak)

        # Compute metrics
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
                "SNR": SNR(xk, input_vector),
            }
        )
        if metrics["||x_{k+1} - x_k||_2 / ||y||_2"][-1] <= _tol:
            print(f"Descent reached tolerance={_tol} at step {step}")
            break
        xk_old = xk.copy()
    return xk, metrics
