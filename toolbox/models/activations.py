import enum
from typing import Union

from torch import nn

from toolbox.utils import StrEnum


class Activation(StrEnum):
    Tanh = enum.auto()
    Sigmoid = enum.auto()
    LeakyReLU = enum.auto()
    ReLU = enum.auto()
    Identity = enum.auto()
    Softplus = enum.auto()
    TanhShifted = enum.auto()


class TanhShifted(nn.Tanh):
    def forward(self, input_):
        return (super().forward(2 * input_ - 1) + 1) / 2


def _convert_str_type_to_activation(str_: str) -> Activation:
    for act in Activation:
        if act == str_:
            return act
    raise ValueError(f"Activation provided is not available: {str_}")


def get_activation(activ_type: Union[Activation, str]) -> nn.Module:
    if type(activ_type) == str:
        activ_type = _convert_str_type_to_activation(activ_type.lower())
    if activ_type.name == "TanhShifted":
        return TanhShifted()
    return getattr(nn, activ_type.name)()
