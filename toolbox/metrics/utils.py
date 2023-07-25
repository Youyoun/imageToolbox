from typing import List, Union

import numpy as np
import torch


def to_numpy_if_tensor(*args: Union[torch.Tensor, np.ndarray]) -> List[np.ndarray]:
    """
    Convert all tensors to numpy arrays if they are tensors.
    :param args: Tensors to convert.
    :return: List of numpy arrays.
    """
    return [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]


def to_tensor_if_numpy(*args: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
    """
    Convert all numpy arrays to tensors if they are numpy arrays.
    :param args: Numpy arrays to convert.
    :return: List of tensors.
    """
    return [
        torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg.detach().cpu()
        for arg in args
    ]
