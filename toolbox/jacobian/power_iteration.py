from typing import Any, Callable, List, Literal, Tuple, Union, overload

import numpy as np
import torch
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator
from scipy.sparse.linalg import eigsh as scipy_eigsh
from scipy.sparse.linalg import lobpcg as scipy_lobpcg

from ..base_classes import Operator
from ..utils import get_module_logger

logger = get_module_logger(__name__)

MIN_POWER_ITERS = 10


def batch_norm(tensor_: torch.Tensor) -> torch.Tensor:
    init_shape = tensor_.shape
    return torch.norm(tensor_.reshape(init_shape[0], -1), dim=1).reshape(init_shape[0], -1)


def batch_normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    init_shape = vector.shape
    return (vector.reshape(init_shape[0], -1) / batch_norm(vector)).reshape(*init_shape)


def power_method(
    x: torch.Tensor,
    operator: Callable,
    max_iter: int = 10,
    tol: float = 1e-10,
    save_iterates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Use the power method to compute the largest eigen value (in magnitude)
    :param x: input vector of shape (BATCH, C, W, H)
    :param operator: Callable function that serves as the operator. Should take x as input vector and return vector of same size
    :param max_iter: Maximum number of iterations on the power method, in case convergence is not reached.
    :param tol: Tolerance on which to stop the power method in case it's reached.
    :param is_eval: Save the graph for back propagation or drop it.
    :param return_vector: Either to return the eigenvector with the eigenvalue or only the eigenvalue.
    :return: (Eigenvectors, Eigenvalues) if return_vector is True, else Eigenvalue
    """
    logger.debug(f"Power method with tol: {tol} and max_iter: {max_iter}")

    # Make sure that the operator is symmetric else it doesn't yield a correct result.
    input_shape = x.shape  # First dimension is always considered to be batch
    u = batch_normalize_vector(torch.randn_like(x))

    iterates = []

    z = None
    zold = None
    for it in np.arange(max_iter):
        v = operator(u)  # \alpha*u - (J+J^t).u (bs, C*W*H)

        # z = u^T . v / u^T.u
        z = (
            torch.sum(u.view(input_shape[0], -1) * v.view(input_shape[0], -1), dim=1)
            / batch_norm(u).squeeze()
        )
        if save_iterates:
            iterates.append(z.detach().cpu().numpy())

        if it > MIN_POWER_ITERS:
            with torch.no_grad():
                rel_var = torch.norm(z - zold).item()
            if rel_var < tol:
                logger.debug(
                    f"Power iteration converged at iteration: {it}, val: {z.max().item()}"
                )
                break
        zold = z

        # u = v / ||v|| = [alpha * I - (J + J^Y)]u / ||[alpha * I - (J + J^Y)].u||
        u = batch_normalize_vector(v)
    return u, z.view(-1), iterates


def conjugate_gradient_smallest_ev(
    x,
    operator,
    tol=1e-8,
    max_iter=10,
    save_iterates=False,
):
    logger.debug(f"Conjugate gradient with tol: {tol} and max_iter: {max_iter}")
    input_shape = x.shape
    uk = batch_normalize_vector(torch.randn_like(x))

    wk = operator(uk)
    lk = torch.sum(wk * uk, dim=[i for i in range(1, len(input_shape))], keepdim=True)
    gk = wk - lk * uk
    pk = gk.clone()
    zk = operator(pk)

    lambdas = []
    if save_iterates:
        lambdas = [lk.squeeze().detach().cpu().numpy()]

    if max_iter is None:
        N = x.nelement()
        max_iter = N * 10
        logger.debug(f"Max iter not specified, using {max_iter}")

    for k in range(max_iter):
        ak = torch.sum(zk * uk, dim=[i for i in range(1, len(input_shape))], keepdim=True)
        bk = torch.sum(zk * pk, dim=[i for i in range(1, len(input_shape))], keepdim=True)
        ck = torch.sum(uk * pk, dim=[i for i in range(1, len(input_shape))], keepdim=True)
        dk = torch.sum(pk * pk, dim=[i for i in range(1, len(input_shape))], keepdim=True)

        delta = (lk * dk - bk) ** 2 - 4 * (bk * ck - ak * dk) * (ak - lk * ck)
        alphak = (lk * dk - bk + torch.sqrt(delta)) / (2 * (bk * ck - ak * dk))
        lk = (lk + alphak * ak) / (1 + alphak * ck)
        gammak = torch.sqrt(1 + 2 * ck * alphak + dk * alphak**2)

        if save_iterates:
            lambdas.append(lk.squeeze().detach().cpu().numpy())

        uk = (uk + alphak * pk) / gammak
        wk = (wk + alphak * zk) / gammak
        gk = wk - lk * uk
        if batch_norm(gk).max() < tol:
            logger.debug(f"Conjugate gradient converged at iteration: {k}")
            break

        betak = -torch.sum(zk * gk, dim=[i for i in range(1, len(input_shape))], keepdim=True) / bk
        pk = gk + betak * pk
        zk = operator(pk)

    return uk, lk.view(-1), lambdas


def lobpcg(
    x,
    operator,
    tol=1e-8,
    max_iter=20,
    save_iterates=False,
):
    logger.debug(f"LOBPCG with tol: {tol} and max_iter: {max_iter}")
    input_shape = x.shape

    A_f = (
        lambda u: operator(torch.Tensor(u).view(input_shape).to(x.device))
        .view(-1, 1)
        .detach()
        .cpu()
        .numpy()
    )

    k = 1
    U = torch.randn(x.nelement(), k).numpy()
    eigenvalues, eigenvectors, history = scipy_lobpcg(
        A_f,
        U,
        largest=False,
        maxiter=max_iter,
        retLambdaHistory=True,
        tol=tol,
    )
    return (
        torch.Tensor(eigenvectors).view(input_shape),
        torch.Tensor(eigenvalues),
        history,
    )


def lanczos(
    x,
    operator,
    tol=1e-4,
    max_iter=20,
    save_iterates=False,
):
    logger.debug(f"LANCZOS with tol: {tol} and max_iter: {max_iter}")
    input_shape = x.shape

    A_f = (
        lambda u: operator(torch.Tensor(u).view(input_shape).to(x.device))
        .view(-1, 1)
        .detach()
        .cpu()
        .numpy()
    )
    A_f = LinearOperator((x.nelement(), x.nelement()), matvec=A_f, rmatvec=A_f)

    k = 1
    try:
        eigenvalues, eigenvectors = scipy_eigsh(
            A_f,
            k,
            which="SA",
            maxiter=max_iter,
            tol=tol,
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as e:
        logger.warning(f"ARPACK error: {e}")
        eigenvalues = e.eigenvalues
        eigenvectors = e.eigenvectors
        if len(eigenvalues) == 0:
            logger.warning("ARPACK failed to converge, retrying with higher tolerance")
            return lanczos(x, operator, tol=tol * 10, max_iter=max_iter)
    return (
        torch.Tensor(eigenvectors).view(input_shape),
        torch.Tensor(eigenvalues),
        None,
    )
