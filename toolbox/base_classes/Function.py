"""
Kind of an interface class to define operators used in this project
"""
from abc import ABC, abstractmethod

import torch


class Function(ABC):
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

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


class GenericFunction(Function):
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


class Fidelity(Function, ABC):
    def f(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def grad(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def prox(self, x: torch.Tensor, tau: float):
        raise FunctionNotDefinedError("Prox function not defined")


class Regularization(Function, ABC):
    @abstractmethod
    def f(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, x: torch.Tensor):
        raise NotImplementedError()

    def prox(self, x: torch.Tensor, tau: float):
        raise FunctionNotDefinedError("Prox function not defined")


class ProximityOp(Function, ABC):
    @abstractmethod
    def f(self, x: torch.Tensor, tau: float):
        raise NotImplementedError()

    def grad(self, *args, **kwargs):
        raise FunctionNotDefinedError("Gradient function not defined")

    @abstractmethod
    def prox(self, x: torch.Tensor, tau: float):
        raise NotImplementedError()
