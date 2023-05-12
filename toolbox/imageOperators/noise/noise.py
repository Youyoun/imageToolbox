import enum
from typing import Union

import numpy as np
import torch

from ...utils import StrEnum

SEED = 50
np.random.seed(SEED)


class NoiseModes(StrEnum):
    GAUSSIAN = enum.auto()
    POISSON = enum.auto()


class NoiseClass:
    """Class that implements __call__ only"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class GaussianNoise(NoiseClass):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std + self.mean


class PoissonNoise(NoiseClass):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Detect if a signed image was input
        low_clip = 0.0
        if x.min() < 0:
            low_clip = -1.0

        if low_clip == -1.0:
            old_max = x.max()
            x = (x + 1.0) / (old_max + 1.0)

        noisy = np.random.poisson(x.numpy() * self.scale) / self.scale

        if low_clip == -1.0:
            noisy = noisy * (float(old_max) + 1.0) - 1.0
        return torch.from_numpy(noisy).float().clip(low_clip, 1)


def _convert_noise_str_to_enum(noise_mode: str) -> NoiseModes:
    for mode in NoiseModes:
        if mode == noise_mode:
            return mode
    raise ValueError(f"{noise_mode} is not implemented here.")


def get_noise_func(
    noise_mode: Union[NoiseModes, str],
    mean_gauss: float = 0.0,
    std_gauss: float = 1.0,
    scale_poisson: float = 1.0,
) -> NoiseClass:
    if noise_mode == NoiseModes.POISSON:
        return PoissonNoise(scale_poisson)
    elif noise_mode == NoiseModes.GAUSSIAN:
        return GaussianNoise(mean_gauss, std_gauss)
    else:
        raise ValueError(f"No such noise function available {noise_mode}")
