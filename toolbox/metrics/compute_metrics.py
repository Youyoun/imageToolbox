from enum import Enum, auto
from typing import Union

import numpy as np
import torch
from piq import PieAPP
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import to_numpy_if_tensor, to_tensor_if_numpy


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


def mean_squared_error(
    tensor1: Union[torch.Tensor, np.ndarray], tensor2: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute the mean squared error between two tensors or numpy arrays.
    Wrapper around np.linalg.norm.
    :param tensor1: First tensor.
    :param tensor2: Second tensor.
    :return: Mean squared error = ||tensor1 - tensor2||_2^2 / N
    """
    tensor1, tensor2 = to_numpy_if_tensor(tensor1, tensor2)
    return (
        np.linalg.norm(tensor1.flatten() - tensor2.flatten(), ord=2) ** 2 / tensor1.size
    ).item()


def mean_absolute_error(
    tensor1: Union[torch.Tensor, np.ndarray], tensor2: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute the mean absolute error between two tensors or numpy arrays.
    Wrapper around np.linalg.norm.
    :param tensor1: First tensor.
    :param tensor2: Second tensor.
    :return: Mean absolute error = ||tensor1 - tensor2||_1 / N
    """
    tensor1, tensor2 = to_numpy_if_tensor(tensor1, tensor2)
    return (
        np.linalg.norm(tensor1.flatten() - tensor2.flatten(), ord=1) / tensor1.size
    ).item()


def compute_relative_difference(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
    reference_vector: Union[torch.Tensor, np.ndarray, None] = None,
) -> float:
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
    return (
        np.linalg.norm(tensor1.flatten() - tensor2.flatten(), ord=2)
        / np.linalg.norm(reference_vector.flatten(), ord=2)
    ).item()


def PSNR(
    tensor_true: Union[torch.Tensor, np.ndarray],
    tensor_test: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute the peak signal-to-noise ratio (PSNR) between two tensors or numpy arrays.
    Wrapper around skimage.metrics.peak_signal_noise_ratio.
    :param tensor_true: True tensor.
    :param tensor_test: Test tensor.
    :return: PSNR = 20 * log10(max(I)) - 10 * log10(MSE)
    """
    tensor_true, tensor_test = to_numpy_if_tensor(tensor_true, tensor_test)
    return peak_signal_noise_ratio(tensor_true, tensor_test)


def SSIM(
    tensor_true: Union[torch.Tensor, np.ndarray],
    tensor_test: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute the structural similarity index (SSIM) between two tensors or numpy arrays.
    Wrapper around skimage.metrics.structural_similarity.
    :param tensor_true: True tensor.
    :param tensor_test: Test tensor.
    :return: SSIM
    """
    tensor_true, tensor_test = to_numpy_if_tensor(tensor_true, tensor_test)
    return structural_similarity(
        tensor_true,
        tensor_test,
        channel_axis=0 if len(tensor_test.shape) == 3 else None,
    )


def SNR(tensor: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the signal-to-noise ratio (SNR) of an image.
    :param tensor: Tensor.
    :return: SNR = 10 * log10(mean(I) / std(I))
    """
    (tensor,) = to_numpy_if_tensor(tensor)
    return 10 * np.log10(np.mean(tensor) / np.std(tensor)).item()


def pieapp(
    tensor_true: Union[torch.Tensor, np.ndarray],
    tensor_test: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute the pieAPP score between two tensors or numpy arrays.
    Wrapper around piq.PieAPP.
    :param tensor_true: True tensor.
    :param tensor_test: Test tensor.
    :return: pieAPP score
    """
    tensor_true, tensor_test = to_tensor_if_numpy(tensor_true, tensor_test)
    if len(tensor_true.shape) == 3:
        tensor_true = tensor_true.unsqueeze(0)
        tensor_test = tensor_test.unsqueeze(0)
    if tensor_true.max() > 1:
        max_range = 255.0
    else:
        max_range = 1.0
    tensor_true = tensor_true / max_range
    tensor_test = tensor_test / max_range
    if tensor_test.max() > 1 or tensor_test.min() < 0:
        print("tensor_test must be in the range [0, 1]. Clipping tensor_test.")
        tensor_test = torch.clamp(tensor_test, 0, 1)
    return PieAPP()(tensor_true, tensor_test).abs().item()
