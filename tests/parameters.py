from typing import Union
import pytest

import numpy as np
import torch

FLOAT_TOL = 1e-4
BATCH_SIZE = 64
NDIM_X = 256
NDIM_Y = 256


def are_equal(v1: Union[torch.Tensor, np.ndarray], v2: Union[torch.Tensor, np.ndarray]):
    if isinstance(v1, np.ndarray):
        v1 = torch.from_numpy(v1)
    if isinstance(v2, np.ndarray):
        v2 = torch.from_numpy(v2)
    return torch.isclose(v1, v2, rtol=0.0, atol=FLOAT_TOL).all()
