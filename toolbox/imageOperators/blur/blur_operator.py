import enum
import math
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as t_transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters._gabor import gabor_kernel

from ...base_classes import Operator
from ...utils import get_module_logger, StrEnum

logger = get_module_logger(__name__)

CURRENT_FILE_PATH = Path(__file__).parent
DID_LOG_ONCE = False


class Kernels(StrEnum):
    GAUSSIAN = enum.auto()
    UNIFORM = enum.auto()
    GABOR = enum.auto()
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


class IdentityOperator(Operator):
    """
    Identity Operator
    """
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return x


def expand_x_dims(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """
    Expand the dimensions of x to 4D
    """
    init_shape = x.shape
    if x.ndim == 2:
        x = x.view(1, 1, *x.shape)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    return x, init_shape


class GaussianBlurFFT(Operator):
    """
    Gaussian Blur using FFT
    Only constant padding is supported
    """
    PAD_MODE = "constant"

    def __init__(self, ksize: int, s: float = 0.5):
        super().__init__()
        # Define Gaussian Kernel
        self.kernel_size = ksize
        self.std = s
        self.kernel = get_kernel(ksize, Kernels.GAUSSIAN, self.std)
        self.matmul_func = self.matvec

    def fft_convolve(self, x: torch.Tensor, conj: bool) -> torch.Tensor:
        padding = self.kernel_size // 2
        x_pad = torch.nn.functional.pad(x, (padding, padding, padding, padding), self.PAD_MODE)
        fft_x = torch.fft.fft2(x_pad)
        fft_h = torch.fft.fft2(self.kernel.to(x.device), s=x_pad.shape[2:]).unsqueeze(0).unsqueeze(0)
        if conj:
            print("Computing conjugate")
            x_new = torch.real(torch.fft.ifft2(fft_x * torch.conj(fft_h)))
        else:
            x_new = torch.real(torch.fft.ifft2(fft_x * fft_h))
        return x_new

    def blurr(self, x: torch.Tensor, conj: bool) -> torch.Tensor:
        x, init_shape = expand_x_dims(x)
        padding = self.kernel_size // 2
        x_new = self.fft_convolve(x, conj)
        if conj:
            x_new = x_new[:, :, :-2 * padding, :-2 * padding]
        else:
            x_new = x_new[:, :, 2 * padding:, 2 * padding:]
        return x_new.view(*init_shape)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.blurr(x, False)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.blurr(x, True)


class BlurConvolution(Operator):
    """
    Blur operator using convolution
    """
    PAD_MODE = "constant"

    def __init__(self,
                 ksize: int,
                 type_: Union[Kernels, str],
                 s: Union[float, Tuple[float, float]] = 0.5,
                 **kwargs):
        super().__init__()
        global DID_LOG_ONCE
        # Define Gaussian Kernel
        self.std = s
        self.type = type_
        self.kernel = get_kernel(ksize, type_, self.std, **kwargs)
        self.kernel_size = self.kernel.shape

        if not DID_LOG_ONCE:
            logger.debug(f"Kernel sum of values: {self.kernel.sum()}")
            DID_LOG_ONCE = True
        # assert self.kernel.sum() == 1.00

    def conv_fw(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Convolution forward pass
        Pad the image according to PAD_MODE, then convolve with the kernel
        """
        x, init_shape = expand_x_dims(x)
        pad_x, pad_y = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        x_padded = F.pad(x, (pad_y, pad_y, pad_x, pad_x), self.PAD_MODE)
        x_blurred = F.conv2d(x_padded, kernel.to(x.device), bias=None, padding=0)
        return x_blurred.view(init_shape)

    def conv_bw(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Convolution transpose or the gradient of the convolution
        For constant padding, we can use the transpose of the convolution and simply crop the image
        For replicate padding, we need to add the extra values to the borders and the corners and then crop
        """
        x, init_shape = expand_x_dims(x)
        pad_x, pad_y = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        x_blurred = F.conv_transpose2d(x, kernel.to(x.device), bias=None, padding=0)
        if self.PAD_MODE == "replicate":
            # Manage borders
            x_blurred[..., pad_x, pad_y:-pad_y] += x_blurred[..., :pad_x, pad_y:-pad_y].sum(dim=-2)
            x_blurred[..., -pad_x - 1, pad_y:-pad_y] += x_blurred[..., -pad_x:, pad_y:-pad_y].sum(dim=-2)
            x_blurred[..., pad_x:-pad_x, pad_y] += x_blurred[..., pad_x:-pad_x, :pad_y].sum(dim=-1)
            x_blurred[..., pad_x:-pad_x, -pad_y - 1] += x_blurred[..., pad_x:-pad_x, -pad_y:].sum(dim=-1)

            # Manage corners
            x_blurred[..., pad_x, pad_y] += x_blurred[..., :pad_x, :pad_y].sum(dim=(-2, -1))
            x_blurred[..., pad_x, -pad_y - 1] += x_blurred[..., :pad_x, -pad_y:].sum(dim=(-2, -1))
            x_blurred[..., -pad_x - 1, pad_y] += x_blurred[..., -pad_x:, :pad_y].sum(dim=(-2, -1))
            x_blurred[..., -pad_x - 1, -pad_y - 1] += x_blurred[..., -pad_x:, -pad_y:].sum(dim=(-2, -1))
        return x_blurred[..., pad_x:-pad_x, pad_y:-pad_y].view(init_shape)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_fw(x, torch.flip(self.kernel, [0, 1]).view(1, 1, *self.kernel_size))

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_bw(x, torch.flip(self.kernel, [0, 1]).view(1, 1, *self.kernel_size))

    def __repr__(self):
        return f"BlurConvolution(kernel_size={self.kernel_size}, type_={self.type}, std={self.std})"


class ZeroOperator(Operator):
    """
    Zero operator
    """
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _convert_str_type_to_kernel(str_: str) -> Kernels:
    """
    Convert a string to a Kernel enum
    """
    for ker in Kernels:
        if ker == str_:
            return ker
    raise ValueError(f"Kernel provided is not available: {str_}")


def get_kernel(ksize: int,
               type_: Union[Kernels, str],
               sigma: Union[float, Tuple[float, float]],
               **kwargs) -> torch.Tensor:
    """
    Get a kernel of a given type and size.
    :param ksize: kernel size
    :param type_: kernel type
    :param sigma: standard deviation
    :param kwargs: additional arguments to input in scikit-image functions (e.g. theta for Gabor)
    """
    if type(type_) == str:
        type_ = _convert_str_type_to_kernel(type_)
    if type_ == Kernels.GAUSSIAN:
        # The correct formula for ksize is: int(4 * sigma + 0.5) + 1
        kernel = np.zeros(shape=(ksize, ksize))
        kernel[ksize // 2, ksize // 2] = 1
        if isinstance(sigma, float):
            sigma = (sigma, sigma)
        return torch.from_numpy(gaussian_filter(kernel, sigma)).float()  # already normalized
    elif type_ == Kernels.UNIFORM:
        return torch.ones((ksize, ksize)) / ksize ** 2
    elif type_ == Kernels.GABOR:
        if isinstance(sigma, float):
            sigma = (sigma, sigma)
        filter_ = np.abs(gabor_kernel(sigma_x=sigma[0], sigma_y=sigma[1], **kwargs))
        return torch.from_numpy(filter_).float()
    else:
        if type_ not in Kernels:
            raise ValueError(f"Invalid kernel specified: {type_}")
        kernel = t_transforms.ToTensor()(Image.open(kernel_type_img_map[type_])).squeeze()
        return kernel / kernel.sum()
