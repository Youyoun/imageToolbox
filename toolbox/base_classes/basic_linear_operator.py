from abc import ABC

import torch


class Operator(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.matvec(x)

    def transpose(self):
        return TransposedOperator(self)

    T = property(transpose)


class TransposedOperator(Operator):
    def __init__(self, operator: Operator):
        super().__init__()
        self.op = operator

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.op.rmatvec(x)
