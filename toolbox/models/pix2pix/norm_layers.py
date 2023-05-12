import enum
from typing import Union

from torch import nn as nn

from ...utils import StrEnum


class NormTypes(StrEnum):
    Instance = enum.auto()
    Batch = enum.auto()
    none = enum.auto()


NormLayer = Union[nn.Identity, nn.BatchNorm2d, nn.InstanceNorm2d]


def get_norm_layer(num_features: int, norm_type: Union[str, NormTypes] = "none") -> NormLayer:
    """Return a normalization layer
    Parameters:
        num_features (int) -- The number of features for the norm layer
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == NormTypes.Batch:
        return nn.BatchNorm2d(num_features, affine=True, track_running_stats=True)
    elif norm_type == NormTypes.Instance:
        return nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    elif norm_type == NormTypes.none:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Normalization layer {norm_type} is not found")
