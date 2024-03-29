from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..imageOperators.blur import Kernels, get_kernel


class SingleFilterConvolution(nn.Module):
    def __init__(self, ksize: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(ksize, int):
            self.kernel_size = (ksize, ksize)
        else:
            self.kernel_size = ksize
        self.blur_para = nn.Parameter(
            torch.randn(1, 1, self.kernel_size[0], self.kernel_size[1])
        )

    def forward(self, x):
        pad_x, pad_y = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        x_padded = F.pad(x, (pad_y, pad_y, pad_x, pad_x), "replicate")
        return F.conv2d(
            x_padded,
            self.blur_para.repeat(x.shape[1], 1, 1, 1),
            groups=x.shape[1],
            bias=None,
            padding=0,
        )

    def forward_t(self, x):
        pad_x, pad_y = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        x_blurred = F.conv_transpose2d(
            x,
            self.blur_para.to(x.device).repeat(x.shape[1], 1, 1, 1),
            bias=None,
            padding=0,
            groups=x.shape[1],
        )
        return x_blurred[:, :, pad_x:-pad_x, pad_y:-pad_y]


class SingleFilterConvolutionProjected(nn.Module):
    def __init__(self, ksize: Union[int, Tuple[int, int]], use_symmetric_form=False):
        super().__init__()
        if isinstance(ksize, int):
            self.kernel_size = (ksize, ksize)
        else:
            self.kernel_size = ksize
        self.kernel = get_kernel(self.kernel_size[0], Kernels.GAUSSIAN, 1.0)
        self.blur_para = nn.Parameter(self.kernel.unsqueeze(0).unsqueeze(0))
        self.use_symmetric_form = use_symmetric_form

    @property
    def normalized_kernel(self):
        blur_para_pos = F.relu(self.blur_para)
        return blur_para_pos / torch.sum(blur_para_pos)

    def convolution(self, x):
        pad_x, pad_y = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        x_padded = F.pad(x, (pad_y, pad_y, pad_x, pad_x), "replicate")
        return F.conv2d(
            x_padded,
            self.normalized_kernel.repeat(x.shape[1], 1, 1, 1),
            groups=x.shape[1],
            bias=None,
            padding=0,
        )
    
    def convolution_t(self, x):
        pad_x, pad_y = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        x_blurred = F.conv_transpose2d(
            x,
            self.normalized_kernel.to(x.device).repeat(x.shape[1], 1, 1, 1),
            bias=None,
            padding=0,
            groups=x.shape[1],
        )
        return x_blurred[:, :, pad_x:-pad_x, pad_y:-pad_y]

    def forward(self, x):
        if self.use_symmetric_form:
            return self.convolution_t(self.convolution(x))
        else:
            return self.convolution(x)

    def forward_t(self, x):
        return self.convolution_t(x)
