"""
Kind of an interface class to define operators used in this project
"""
from abc import ABC, abstractmethod

import torch


class Function(ABC):
    """
    Abstract class to define a function
    """

    def __call__(self, *args, **kwargs):
        return self.grad(*args, **kwargs)

    @abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def prox(self, *args, **kwargs):
        raise NotImplementedError()


class FunctionNotDefinedError(Exception):
    pass


class Fidelity(Function, ABC):
    """
    Abstract class to define a fidelity function
    Signature of the function: f(x, y)
    """

    @abstractmethod
    def f(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def prox(self, x: torch.Tensor, tau: float):
        raise FunctionNotDefinedError("Prox function not defined")


class Regularization(Function, ABC):
    """
    Abstract class to define a regularization function
    Signature of the function: f(x)
    """

    @abstractmethod
    def f(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, x: torch.Tensor):
        raise NotImplementedError()

    def prox(self, x: torch.Tensor, tau: float):
        raise FunctionNotDefinedError("Prox function not defined")


class ProximityOp(Function, ABC):
    """
    Abstract class to define a proximity operator
    Signature of the function: prox(x, tau)
    """

    @abstractmethod
    def f(self, x: torch.Tensor, tau: float):
        raise NotImplementedError()

    def grad(self, *args, **kwargs):
        raise FunctionNotDefinedError("Gradient function not defined")

    @abstractmethod
    def prox(self, x: torch.Tensor, tau: float):
        raise NotImplementedError()


class GenericFunction(Fidelity, Regularization, ProximityOp):
    """
    Generic class to define a function
    """

    def __init__(self, forward_fn, grad_fn, prox_fn):
        super().__init__()
        self.forward_fn = forward_fn
        self.grad_fn = grad_fn
        self.prox_fn = prox_fn

    def f(self, *args, **kwargs):
        if self.forward_fn is None:
            raise FunctionNotDefinedError("Forward function not defined")
        return self.forward_fn(*args, **kwargs)

    def grad(self, *args, **kwargs):
        if self.grad_fn is None:
            raise FunctionNotDefinedError("Gradient function not defined")
        return self.grad_fn(*args, **kwargs)

    def prox(self, *args, **kwargs):
        if self.prox_fn is None:
            raise FunctionNotDefinedError("Prox function not defined")
        return self.prox_fn(*args, **kwargs)
