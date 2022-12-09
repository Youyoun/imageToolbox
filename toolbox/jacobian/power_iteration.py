from typing import Callable

import numpy as np
import torch

from ..utils import get_module_logger

logger = get_module_logger(__name__)

MIN_POWER_ITERS = 10


def batch_norm(tensor_: torch.Tensor) -> torch.Tensor:
    init_shape = tensor_.shape
    return torch.norm(tensor_.view(init_shape[0], -1), dim=1).view(init_shape[0], -1)


def batch_normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    init_shape = vector.shape
    return (vector.view(init_shape[0], -1) / batch_norm(vector)).view(*init_shape)


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
    input_shape = x.shape  # First dimension is always considered to be batch
    u = batch_normalize_vector(torch.randn_like(x))

    z = None
    zold = None
    for it in np.arange(max_iter):
        v = operator(u)  # \alpha*u - (J+J^t).u (bs, C*W*H)

        # z = u^T . v / u^T.u
        z = torch.sum(u.view(input_shape[0], -1) * v.view(input_shape[0], -1), dim=1) / batch_norm(u).squeeze()

        if it > MIN_POWER_ITERS:
            with torch.no_grad():
                rel_var = torch.norm(z - zold).item()
            if rel_var < tol:
                logger.debug(f"Power iteration converged at iteration: {it}, val: {z.max().item()}")
                break
        zold = z

        # u = v / ||v|| = [alpha * I - (J + J^Y)]u / ||[alpha * I - (J + J^Y)].u||
        u = batch_normalize_vector(v)

        # if is_eval:
        #     v.detach_()
        #     u.detach_()

    # if is_eval:  # Just in case
    #     z.detach_()
    #     u.detach_()
    if return_vector:
        return u, z.view(-1)
    return z.view(-1)
