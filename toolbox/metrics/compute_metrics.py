from enum import Enum, auto
from typing import Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import to_numpy_if_tensor


class MetricNames(Enum):
    MSE = auto()
    PSNR = auto()
    MAE = auto()
    RELDIFF = auto()
    SNR = auto()
    SSIM = auto()
    LOSS = auto()
    L1 = auto()
    L2 = auto()
    PENALTY = auto()
    NORM = auto()
    EIGENVALUE = auto()


def mean_squared_error(tensor1: Union[torch.Tensor, np.ndarray], tensor2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the mean squared error between two tensors or numpy arrays.
    Wrapper around np.linalg.norm.
    :param tensor1: First tensor.
    :param tensor2: Second tensor.
    :return: Mean squared error = ||tensor1 - tensor2||_2^2 / N
    """
    tensor1, tensor2 = to_numpy_if_tensor(tensor1, tensor2)
    return np.linalg.norm(tensor1.flatten() - tensor2.flatten(), ord=2) ** 2 / tensor1.size


def mean_absolute_error(tensor1: Union[torch.Tensor, np.ndarray], tensor2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the mean absolute error between two tensors or numpy arrays.
    Wrapper around np.linalg.norm.
    :param tensor1: First tensor.
    :param tensor2: Second tensor.
    :return: Mean absolute error = ||tensor1 - tensor2||_1 / N
    """
    tensor1, tensor2 = to_numpy_if_tensor(tensor1, tensor2)
    return np.linalg.norm(tensor1.flatten() - tensor2.flatten(), ord=1) / tensor1.size


def compute_relative_difference(tensor1: Union[torch.Tensor, np.ndarray],
                                tensor2: Union[torch.Tensor, np.ndarray],
                                reference_vector: Union[torch.Tensor, np.ndarray] = None) -> float:
    """
    Compute the relative difference between two tensors or numpy arrays.
    Wrapper around np.linalg.norm.
    :param tensor1: First tensor.
    :param tensor2: Second tensor.
    :param reference_vector: Reference vector to compute the relative difference. If None, tensor1 is used.
    :return: Relative difference = ||tensor1 - tensor2||_2 / ||reference_vector||_2
    """
    tensor1, tensor2 = to_numpy_if_tensor(tensor1, tensor2)
    if reference_vector is None:
        reference_vector = tensor1
    return np.linalg.norm(tensor1.flatten() - tensor2.flatten(), ord=2) / np.linalg.norm(reference_vector.flatten(),
                                                                                         ord=2)


def PSNR(tensor_true: Union[torch.Tensor, np.ndarray], tensor_test: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the peak signal-to-noise ratio (PSNR) between two tensors or numpy arrays.
    Wrapper around skimage.metrics.peak_signal_noise_ratio.
    :param tensor_true: True tensor.
    :param tensor_test: Test tensor.
    :return: PSNR = 20 * log10(max(I)) - 10 * log10(MSE)
    """
    tensor_true, tensor_test = to_numpy_if_tensor(tensor_true, tensor_test)
    return peak_signal_noise_ratio(tensor_true, tensor_test, data_range=1)


def SSIM(tensor_true: Union[torch.Tensor, np.ndarray], tensor_test: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the structural similarity index (SSIM) between two tensors or numpy arrays.
    Wrapper around skimage.metrics.structural_similarity.
    :param tensor_true: True tensor.
    :param tensor_test: Test tensor.
    :return: SSIM
    """
    tensor_true, tensor_test = to_numpy_if_tensor(tensor_true, tensor_test)
    return structural_similarity(tensor_true, tensor_test, data_range=1)


def SNR(tensor_true: Union[torch.Tensor, np.ndarray], tensor_test: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the signal-to-noise ratio (SNR) between two tensors or numpy arrays.
    :param tensor_true: True tensor.
    :param tensor_test: Test tensor.
    :return: SNR = 10 * log10(var(tensor_true) / var(tensor_test - tensor_true))
    """
    tensor_true, tensor_test = to_numpy_if_tensor(tensor_true, tensor_test)
    return 10 * np.log10(np.var(tensor_true) / np.var(tensor_test - tensor_true))
