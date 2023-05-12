# Taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# All copyrights go to that link.
from enum import auto
from typing import Union

from ...utils import StrEnum
from .discriminators import NLayerDiscriminator, PixelDiscriminator
from .init_model import init_net
from .norm_layers import NormTypes
from .resnet_generator import ResnetGenerator
from .unet_generator import UnetGenerator


class GeneratorModels(StrEnum):
    Resnet_9blocks = auto()
    Resnet_6blocks = auto()
    Unet_128 = auto()
    Unet_256 = auto()


class DiscriminatorModels(StrEnum):
    Basic = auto()
    N_layers = auto()
    Pixel = auto()


def get_generator(
    input_nc: int,
    output_nc: int,
    ngf: int,
    netG: Union[str, GeneratorModels],
    norm: Union[str, NormTypes] = NormTypes.none,
    use_dropout: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list = [],
):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    if netG == GeneratorModels.Resnet_9blocks:
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm, use_dropout=use_dropout, n_blocks=9
        )
    elif netG == GeneratorModels.Resnet_6blocks:
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm, use_dropout=use_dropout, n_blocks=6
        )
    elif netG == GeneratorModels.Unet_128:
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm, use_dropout=use_dropout)
    elif netG == GeneratorModels.Unet_256:
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm, use_dropout=use_dropout)
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def get_discriminator(
    input_nc: int,
    ndf: int,
    netD: Union[str, DiscriminatorModels],
    n_layers_D: int = 3,
    norm: Union[str, NormTypes] = NormTypes.none,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list = [],
) -> Union[NLayerDiscriminator, PixelDiscriminator]:
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    if netD == DiscriminatorModels.Basic:  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm)
    elif netD == DiscriminatorModels.N_layers:  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm)
    elif netD == DiscriminatorModels.Pixel:  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return init_net(net, init_type, init_gain, gpu_ids)
