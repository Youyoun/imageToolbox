import torch.nn as nn

from .activations import Activation
from .utils import get_model

FEATURES = 64


class DnCNN(nn.Module):
    def __init__(self, channels: int = 1, depth: int = 17, use_batchnorm: bool = False):
        super().__init__()
        self.cnn = get_model(
            channels,
            channels,
            FEATURES,
            depth,
            Activation.Identity,
            Activation.ReLU,
            use_batchnorm,
        )

    def forward(self, x):
        return x - self.cnn(x)

    def get_residual(self, x):
        return self.cnn(x)
