"""
Kind of an interface class to define operators used in this project
"""
from abc import ABC, abstractmethod


class Function(ABC):
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    @abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError()
