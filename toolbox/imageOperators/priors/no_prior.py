import torch

from src.Function import Function
from src.utils import get_module_logger

logger = get_module_logger(__name__)


class NoRegularisation(Function):
    def __init__(self):
        pass

    @staticmethod
    def f(x):
        return torch.zeros_like(x).sum()

    @staticmethod
    def grad(x):
        return torch.zeros_like(x)
