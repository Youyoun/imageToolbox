from typing import Tuple

import torch


def to_4D(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    init_shape = t.shape
    if t.ndim == 2:
        return t.view(1, 1, *init_shape), init_shape
    elif t.ndim == 3:
        assert (
            t.shape[0] == 1 or t.shape[0] == 3
        ), f"Number of channels is neither 1 nor 3 {t.shape[0]}"
        return t.view(1, *init_shape), init_shape
    elif t.ndim == 4:
        assert (
            t.shape[1] == 1 or t.shape[1] == 3
        ), f"Number of channels is neither 1 nor 3 {t.shape[1]}"
        return t, t.shape
    else:
        raise ValueError("Was expecting image to be at maximum 4D tensor.")
