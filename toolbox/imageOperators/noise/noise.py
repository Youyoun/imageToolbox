from enum import IntEnum

import numpy as np
import torch

SEED = 50
np.random.seed(SEED)


class NoiseModes(IntEnum):
    GAUSSIAN = 0
    POISSON = 1


class GaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std + self.mean


class PoissonNoise:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        # Detect if a signed image was input
        low_clip = 0.
        if x.min() < 0:
            low_clip = -1.

        if low_clip == -1.:
            old_max = x.max()
            x = (x + 1.) / (old_max + 1.)

        noisy = np.random.poisson(x.numpy() * self.scale) / self.scale

        if low_clip == -1.:
            noisy = noisy * (float(old_max) + 1.) - 1.
        return torch.from_numpy(noisy).float().clip(low_clip, 1)


def get_noise_func(noise_parameters):
    if noise_parameters.NOISE_MODE == NoiseModes.POISSON:
        return PoissonNoise(noise_parameters.POISSON_SCALE)
    elif noise_parameters.NOISE_MODE == NoiseModes.GAUSSIAN:
        return GaussianNoise(noise_parameters.NOISE_MEAN, noise_parameters.NOISE_STD)
    else:
        raise ValueError(f"No such noise function available {noise_parameters.NOISE_MODE}")
