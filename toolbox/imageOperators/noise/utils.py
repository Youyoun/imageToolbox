from typing import Tuple

import torch
from PIL import Image, ImageOps
from torchvision import transforms as t_transforms

from .noise import NoiseModes, get_noise_func
from ..blur.blur_operator import BlurConvolution
from ..utils import get_module_logger

logger = get_module_logger(__name__)

DID_LOG_ONCE = False


def get_clean_image(image_path: str) -> Tuple[Image.Image, torch.Tensor]:
    orig_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(orig_image)
    gray_t = t_transforms.ToTensor()(gray_image)
    return gray_image, gray_t


def get_noisy_image(im_t: torch.Tensor, parameters) -> Tuple[Image.Image, torch.Tensor, BlurConvolution]:
    assert im_t.max() == 1 and im_t.min() == 0, "Image bounds are  < 0 or > 1"
    global DID_LOG_ONCE
    if not DID_LOG_ONCE:
        logger.info(f"Using Kernel: {parameters.BLUR_TYPE.name} Noise type: {parameters.NOISE_MODE.name}")
        logger.debug(
            f"Noise parameters: {f'μ={parameters.NOISE_MEAN} σ={parameters.NOISE_STD}' if parameters.NOISE_MODE == NoiseModes.GAUSSIAN else f'λ={parameters.POISSON_SCALE}'}")
        DID_LOG_ONCE = True
    blur_op = BlurConvolution(parameters.KERNEL_SIZE, parameters.BLUR_TYPE, parameters.BLUR_STD)
    noise_functions = get_noise_func(parameters)
    noisy_t = torch.clamp(noise_functions(blur_op @ im_t), 0, 1)
    noisy_image = t_transforms.ToPILImage()(noisy_t)
    return noisy_image, noisy_t, blur_op
