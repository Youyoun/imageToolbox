import pathlib
from typing import Tuple, Union

import torch
from PIL import Image, ImageOps
from torchvision import transforms as t_transforms

from .noise import NoiseModes, get_noise_func
from ..blur import BlurConvolution, Kernels
from ...utils import get_module_logger

logger = get_module_logger(__name__)

DID_LOG_ONCE = False


def get_clean_image(image_path: Union[pathlib.Path, str]) -> Tuple[Image.Image, torch.Tensor]:
    orig_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(orig_image)
    gray_t = t_transforms.ToTensor()(gray_image)
    return gray_image, gray_t


def get_noisy_image(im_t: torch.Tensor,
                    blur_type: Union[Kernels, str],
                    kernel_size: int,
                    kernel_std: float,
                    noise_mode: Union[NoiseModes, str],
                    mean: float = 0.0,
                    std: float = 1.0,
                    scale: float = 100.0) -> Tuple[
    Image.Image, torch.Tensor, BlurConvolution]:
    assert im_t.max() == 1 and im_t.min() == 0, "Image bounds are  < 0 or > 1"
    global DID_LOG_ONCE
    if not DID_LOG_ONCE:
        logger.info(f"Using Kernel: {blur_type} Noise type: {noise_mode}")
        logger.debug(
            f"Noise parameters: {f'μ={mean} σ={std}' if noise_mode == NoiseModes.GAUSSIAN else f'λ={scale}'}")
        DID_LOG_ONCE = True
    blur_op = BlurConvolution(kernel_size, blur_type, kernel_std)
    noise_functions = get_noise_func(noise_mode, mean, std, scale_poisson=scale)
    noisy_t = torch.clamp(noise_functions(blur_op @ im_t), 0, 1)
    noisy_image = t_transforms.ToPILImage()(noisy_t)
    return noisy_image, noisy_t, blur_op
