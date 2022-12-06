import enum
from pathlib import Path
from typing import Union

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import torchvision.transforms as t_transforms
from PIL import Image

from ..utils import get_module_logger, StrEnum

logger = get_module_logger(__name__)

CURRENT_FILE_PATH = Path(__file__).parent
DID_LOG_ONCE = False


class Kernels(StrEnum):
    GAUSSIAN = enum.auto()
    UNIFORM = enum.auto()
    TYPE_A = enum.auto()
    TYPE_B = enum.auto()
    TYPE_C = enum.auto()
    TYPE_D = enum.auto()
    TYPE_E = enum.auto()
    TYPE_F = enum.auto()
    TYPE_G = enum.auto()
    TYPE_H = enum.auto()


kernel_type_img_map = {
    Kernels.TYPE_A: CURRENT_FILE_PATH / "kernels/kernel1.png",
    Kernels.TYPE_B: CURRENT_FILE_PATH / "kernels/kernel2.png",
    Kernels.TYPE_C: CURRENT_FILE_PATH / "kernels/kernel3.png",
    Kernels.TYPE_D: CURRENT_FILE_PATH / "kernels/kernel4.png",
    Kernels.TYPE_E: CURRENT_FILE_PATH / "kernels/kernel5.png",
    Kernels.TYPE_F: CURRENT_FILE_PATH / "kernels/kernel6.png",
    Kernels.TYPE_G: CURRENT_FILE_PATH / "kernels/kernel7.png",
    Kernels.TYPE_H: CURRENT_FILE_PATH / "kernels/kernel8.png"
}


class Operator:
    def __init__(self, *args, **kwargs):
        pass

    def matvec(self, x):
        raise NotImplemented()

    def rmatvec(self, x):
        raise NotImplemented()

    def __matmul__(self, x):
        return self.matvec(x)

    def transpose(self):
        return TransposedOperator(self)

    T = property(transpose)


class TransposedOperator(Operator):
    def __init__(self, operator):
        super().__init__()
        self.op = operator

    def __matmul__(self, x):
        return self.op.rmatvec(x)


class IdentityOperator(Operator):
    def matvec(self, x):
        return x

    def rmatvec(self, x):
        return x


def get_kernel_fft(kernel, dimensions):
    kernel_h = F.pad(kernel, (dimensions[3] // 2, dimensions[3] // 2, dimensions[2] // 2, dimensions[2] // 2),
                     'constant')
    kernel_h = torch.fft.ifftshift(kernel_h)
    # kernel_h = kernel_h.expand(dimensions[0], dimensions[1], dimensions[2] + 2 * padding, dimensions[3] + 2 * padding)
    return torch.fft.fft2(kernel_h)


def expand_x_dims(x):
    init_shape = x.shape
    if x.ndim == 2:
        x = x.view(1, 1, *x.shape)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    return x, init_shape


class GaussianBlurFFT(Operator):
    PAD_MODE = "replicate"

    def __init__(self, ksize, s=0.5):
        super().__init__()
        # Define Gaussian Kernel
        self.kernel_size = ksize
        self.std = s
        self.kernel = get_kernel(ksize, Kernels.GAUSSIAN, self.std)
        self.matmul_func = self.matvec

    def fft_convolve(self, x, conj: bool):
        padding = self.kernel_size // 2
        fft_h = get_kernel_fft(self.kernel.to(x.device), x.shape, padding)
        x_pad = torch.nn.functional.pad(x, (padding, padding, padding, padding), self.PAD_MODE)
        fft_x = torch.fft.fft2(x_pad)
        if conj:
            x_new = torch.real(torch.fft.ifft2(fft_x * torch.conj(fft_h)))
        else:
            x_new = torch.real(torch.fft.ifft2(fft_x * fft_h))
        return x_new

    def blurr(self, x, conj: bool):
        x, init_shape = expand_x_dims(x)
        padding = self.kernel_size // 2
        x_new = self.fft_convolve(x, conj)
        x_new = x_new[:, :, padding:- padding, padding:-padding]
        return x_new.view(*init_shape)

    def matvec(self, x):
        #         print("Calling matvec")
        return self.blurr(x, False)

    def rmatvec(self, x):
        #         print("Calling rmatvec")
        return self.blurr(x, True)


class BlurConvolution(Operator):
    PAD_MODE = "replicate"

    def __init__(self, ksize: int, type_: Kernels, s: float = 0.5):
        super().__init__()
        global DID_LOG_ONCE
        # Define Gaussian Kernel
        self.std = s
        self.kernel = get_kernel(ksize, type_, self.std)
        self.kernel_size = self.kernel.shape[0]

        if not DID_LOG_ONCE:
            logger.debug(f"Kernel sum of values: {self.kernel.sum()}")
            DID_LOG_ONCE = True
        # assert self.kernel.sum() == 1.00

    def blurr(self, x, kernel):
        x, init_shape = expand_x_dims(x)
        padding = self.kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), self.PAD_MODE)
        x_blurred = F.conv2d(x_padded, kernel.to(x.device), bias=None, padding=0)
        return x_blurred.view(init_shape)

    def matvec(self, x):
        return self.blurr(x, torch.flip(self.kernel, [0, 1]).view(1, 1, self.kernel_size, self.kernel_size))

    def rmatvec(self, x):
        return self.blurr(x, self.kernel.view(1, 1, self.kernel_size, self.kernel_size))


class ZeroOperator(Operator):
    def matvec(self, x):
        return torch.zeros_like(x)

    def rmatvec(self, x):
        return torch.zeros_like(x)


def _convert_str_type_to_kernel(str_: str):
    for ker in Kernels:
        if ker == str_:
            return ker
    raise ValueError(f"Kernel provided is not available: {str_}")


def get_kernel(ksize: int, type_: Union[Kernels, str], sigma: float):
    if type(type_) == str:
        type_ = _convert_str_type_to_kernel(type_)
    if type_ == Kernels.GAUSSIAN:
        kernel = np.zeros(shape=(ksize, ksize))
        kernel[ksize // 2, ksize // 2] = 1
        return torch.from_numpy(gaussian_filter(kernel, sigma)).float()  ## already normalized
    elif type_ == Kernels.UNIFORM:
        return torch.ones((ksize, ksize)) / ksize ** 2
    else:
        if type_ not in Kernels:
            raise ValueError(f"Invalid kernel specified: {type_}")
        kernel = t_transforms.ToTensor()(Image.open(kernel_type_img_map[type_])).squeeze()
        return kernel / kernel.sum()
