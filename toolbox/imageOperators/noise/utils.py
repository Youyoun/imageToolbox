import torch
from PIL import Image, ImageOps
from torchvision import transforms as t_transforms

from . import logger
from .noise import NoiseModes, get_noise_func
from src.blur.blur_operator import BlurConvolution

DID_LOG_ONCE = False


def get_clean_image(image_path):
    orig_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(orig_image)
    gray_t = t_transforms.ToTensor()(gray_image)
    return gray_image, gray_t


def get_noisy_image(im_arr, parameters):
    global DID_LOG_ONCE
    if not LOG_ONCE:
        logger.info(f"Using Kernel: {parameters.BLUR_TYPE.name} Noise type: {parameters.NOISE_MODE.name}")
        logger.debug(
            f"Noise parameters: {f'μ={parameters.NOISE_MEAN} σ={parameters.NOISE_STD}' if parameters.NOISE_MODE == NoiseModes.GAUSSIAN else f'λ={parameters.POISSON_SCALE}'}")
        LOG_ONCE = True
    blur_op = BlurConvolution(parameters.KERNEL_SIZE, parameters.BLUR_TYPE, parameters.BLUR_STD)
    noise_functions = get_noise_func(parameters)
    noisy_t = torch.clamp(noise_functions(blur_op @ im_arr), 0, 1)
    noisy_image = t_transforms.ToPILImage()(noisy_t)
    return noisy_image, noisy_t, blur_op
