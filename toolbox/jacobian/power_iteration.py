from typing import Callable

import numpy as np
import torch

from ..utils import get_module_logger

logger = get_module_logger(__name__)

MIN_POWER_ITERS = 10


def power_method(x: torch.Tensor,
                 operator: Callable,
                 max_iter: int = 10,
                 tol: float = 1e-10,
                 is_eval: bool = False,
                 return_vector: bool = False):
    """
    Use the power method to compute the largest eigen value (in magnitude)
    :param x: input vector of shape (BATCH, N) where N is the number of pixel in an image (flatten)
    :param operator: Callable function that serves as the operator. Should take x as input vector and return vector of same size
    :param max_iter: Maximum number of iterations on the power method, in case convergence is not reached.
    :param tol: Tolerance on which to stop the power method in case it's reached.
    :param is_eval: Save the graph for back propagation or drop it.
    :param return_vector: Either to return the eigenvector with the eigenvalue or only the eigenvalue.
    :return: (Eigenvectors, Eigenvalues) if return_vector is True, else Eigenvalue
    """
    # Make sure that the operator is symmetric else it doesn't yield a correct result.
    assert x.ndim == 2, f"Input should be two dimensionnal, {x.ndim=}"
    u = torch.randn_like(x)
    u = u / torch.sum(u * u, dim=1).unsqueeze(-1)

    z = None
    zold = None

    for it in np.arange(max_iter):
        v = operator(u)  # \alpha*u - (J+J^t).u (bs, C*W*H)
        z = torch.sum(u * v, dim=1) / torch.sum(u * u, dim=1)  # z = u^T . v / u^T.u

        if it > MIN_POWER_ITERS:
            with torch.no_grad():
                rel_var = torch.norm(z - zold).item()
            if rel_var < tol:
                logger.debug(f"Power iteration converged at iteration: {it}, val: {z.max().item()}")
                break
        zold = z

        # u = v / ||v|| = [alpha * I - (J + J^Y)]u / ||[alpha * I - (J + J^Y)].u||
        u = v / torch.sum(v * v, dim=1).unsqueeze(-1)

        if is_eval:
            v.detach_()
            u.detach_()

    if is_eval:  # Just in case
        z.detach_()
        u.detach_()
    if return_vector:
        return u, z.view(-1)
    return z.view(-1)
