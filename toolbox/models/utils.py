from typing import Union

import torch.nn.init as init
from torch import nn

from toolbox.utils import get_module_logger

from .activations import Activation, get_activation

STRIDE = 1
KERNEL_SIZE = 3
PADDING = 1  # KERNEL_SIZE // 2
PADDING_MODE = "zeros"

logger = get_module_logger(__name__)


def get_model(
    in_c: int,
    out_c: int,
    mid_c: int,
    n_layers: int,
    last_activation: Union[Activation, str],
    mid_activation: Union[Activation, str],
    use_batchnorm: bool = False,
) -> nn.Module:
    if n_layers == 1:
        model = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                KERNEL_SIZE,
                STRIDE,
                PADDING,
                bias=True,
                padding_mode=PADDING_MODE,
            ),
            get_activation(last_activation),
        )
    else:
        layers = [
            nn.Conv2d(
                in_c,
                mid_c,
                KERNEL_SIZE,
                STRIDE,
                PADDING,
                bias=False,
                padding_mode=PADDING_MODE,
            ),
            get_activation(mid_activation),
        ]
        for i in range(1, n_layers - 1):
            layers.append(
                nn.Conv2d(
                    mid_c,
                    mid_c,
                    KERNEL_SIZE,
                    STRIDE,
                    PADDING,
                    bias=False,
                    padding_mode=PADDING_MODE,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(mid_c))
            layers.append(get_activation(mid_activation))
        layers.append(
            nn.Conv2d(
                mid_c,
                out_c,
                KERNEL_SIZE,
                STRIDE,
                PADDING,
                bias=False,
                padding_mode=PADDING_MODE,
            )
        )
        layers.append(get_activation(last_activation))
        model = nn.Sequential(*layers)
    _initialize_weights(model)
    return model


def _initialize_weights(model: nn.Module) -> None:
    logger.debug("Models Initialized")
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
