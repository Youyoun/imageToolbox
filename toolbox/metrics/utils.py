import torch
import numpy as np
from typing import List, Union


def to_numpy_if_tensor(*args: Union[torch.Tensor, np.ndarray]) -> List[np.ndarray]:
    """
    Convert all tensors to numpy arrays if they are tensors.
    :param args: Tensors to convert.
    :return: List of numpy arrays.
    """
    return [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
