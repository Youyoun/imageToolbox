from typing import Callable

import torch
from torch.autograd.functional import jacobian


def Ju(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, is_eval: bool = False
) -> torch.Tensor:
    """
    Returns the jacobian vector product. The jacobian is evaluated on x according to y (J_y(x)).
    :param x: input vector (must require grad)
    :param y: output vector (output of neural network)
    :param u: The gradient direction (the vector of JVP)
    :param is_eval: Either to save the graph for backprogation or not (train or eval mode)
    :return: J_y(x) @ v
    """
    # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
    w = torch.ones_like(y, requires_grad=True)
    rop = torch.autograd.grad(
        torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=not is_eval
    )[0]
    if is_eval:
        w.detach_()
    return rop


def JTu(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, is_eval: bool = False
) -> torch.Tensor:
    """
    Returns the jacobian transposed vector product (Equivalent to Lop).
    The jacobian is evaluated on x according to y (J_y(x)) and then transposed.
    :param x: input vector (must require grad)
    :param y: output vector (output of neural network)
    :param u: The gradient direction (the vector of JVP)
    :param is_eval: Either to save the graph for backprogation or not (train or eval mode)
    :return: J_y(x).T @ v
    """
    return torch.autograd.grad(y, x, u, create_graph=not is_eval, retain_graph=True)[0]


def alpha_operator(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, alpha, is_eval: bool = False
) -> torch.Tensor:
    """
    Simple transformation: Computed [\alpha * I - 1/2 (J_y(x).T + J_y(x))] @ v
    :param x: input vector (must require grad)
    :param y: output vector (output of neural network)
    :param u: The gradient direction (the vector of JVP)
    :param is_eval: Either to save the graph for backprogation or not (train or eval mode)
    :param alpha: floating by which to shift the Jacobian
    :return: \alpha * u - (J_y(x).T + J_y(x)) @ u
    """
    return alpha * u - sum_J_JT(
        x, y, u, is_eval
    )  # JTu(x, y, u, is_eval) - Ju(x, y, u, is_eval)


def sum_J_JT(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, is_eval: bool = False
) -> torch.Tensor:
    """
    Returns the additive symmetric form of the jacobian times a certain vector ((M + M^T) / 2)
    The jacobian is evaluated on x according to y (J_y(x)).
    :param x: input vector (must require grad)
    :param y: output vector (output of neural network)
    :param u: The gradient direction (the vector of JVP)
    :param is_eval: Either to save the graph for backprogation or not (train or eval mode)
    :return: 1 / 2 * [J_y(x) + J_y(x).T] @ v
    """
    return 1 / 2 * (Ju(x, y, u, is_eval) + JTu(x, y, u, is_eval))


def JTJu(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, is_eval: bool = False
) -> torch.Tensor:
    """ "
    Returns the jacobian transposed times the jacobian times a certain vector (M^T @ M)
    The jacobian is evaluated on x according to y (J_y(x)).
    :param x: input vector (must require grad)
    :param y: output vector (output of neural network)
    :param u: The gradient direction (the vector of JVP)
    :param is_eval: Either to save the graph for backprogation or not (train or eval mode)
    :return: J_y(x).T @ J_y(x) @ v
    """
    return JTu(x, y, Ju(x, y, u, is_eval=is_eval), is_eval=is_eval)


def compute_jacobian(
    net: Callable, x: torch.Tensor, is_eval: bool = False
) -> torch.Tensor:
    """
    Compute the exact jacobian of the neural network (callable function at least)
    Be careful, the size of the matrix is too big (And so the computation is too slow for images of a certain size)
    :param net: nn.Module, the network the transforms x -> y
    :param x: the input of size: (BATCH, CHANNELS, WIDTH, HEIGHT)
    :return: The jacobian tensor
    """
    bs = x.shape[0]
    size = x.nelement() // bs
    shape_size = len(x.shape) * 2 - 1
    permute_idx = [
        shape_size // 2,
        *list(range(0, shape_size // 2)),
        *list(range(shape_size // 2 + 1, shape_size)),
    ]
    return batch_jacobian(net, x, is_eval).permute(*permute_idx).view(bs, size, size)


def batch_jacobian(f: Callable, x: torch.Tensor, is_eval: bool = False) -> torch.Tensor:
    """
    Compute the jacobian on a batch of data using a simple trick.
    :param f: callable function or neural network nn.Module
    :param x: input of shape (BATCH, CHANNELS, WIDTH, HEIGHT)
    :param is_eval:  Either to save the graph for backprogation or not (train or eval mode)
    :return: A Batch of jacobians at each elements that constitutes the input vector.
    """
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x, create_graph=not is_eval, vectorize=True)
