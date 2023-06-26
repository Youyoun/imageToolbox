"""
A small version of the residual U-Net architecture for semantic segmentation.
Contains 3 down- and up-sampling blocks.
Should run on 64x64 images.
Encoder feature maps are 64x64, 32x32, 16x16.
"""

import torch
import torch.nn as nn

from ..imageOperators.im_gradient.utils import to_4D


def get_input_layer(channels: int, filters: int, use_batchnorm: bool):
    if use_batchnorm:
        return nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
        )


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=stride, padding=1
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class RUnet(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        filters: list = [32, 64, 128, 256],
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.input_layer = get_input_layer(channels, filters[0], use_batchnorm)
        self.input_skip = nn.Conv2d(channels, filters[0], 3, padding=1)

        self.residual_down1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_down2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample1 = nn.ConvTranspose2d(filters[3], filters[3], 2, 2)
        self.residual_up1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample2 = nn.ConvTranspose2d(filters[2], filters[2], 2, 2)
        self.residual_up2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample3 = nn.ConvTranspose2d(filters[1], filters[1], 2, 2)
        self.residual_up3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channels, 1, 1), nn.Tanh()
        )

    def forward(self, x):
        x, init_shape = to_4D(x)
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_down1(x1)
        x3 = self.residual_down2(x2)
        x4 = self.bridge(x3)

        x5 = self.upsample1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.residual_up1(x5)
        x6 = self.upsample2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.residual_up2(x6)
        x7 = self.upsample3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.residual_up3(x7)

        return self.output_layer(x7).view(init_shape)
