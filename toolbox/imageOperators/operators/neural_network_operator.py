import torch

from src.Function import Function
from src.models.models import ModelFactory, Models
from src.utils import get_module_logger

logger = get_module_logger(__name__)


class ModelAsRegularisation(Function):
    def __init__(self, model_paras, device="cpu"):
        model = ModelFactory.get(model_paras).to(device)
        self.config = model_paras
        self.model = model
        self.device = device

    def f(self, x):
        return torch.zeros(1).to(x.device)

    def grad(self, x):
        orig_size = x.shape
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if self.config.MODEL_TYPE == Models.DnCNN:
            return self.model.get_residual(x).view(*orig_size)
        else:
            return self.model(x).view(*orig_size)

    def parameters(self):
        return self.model.parameters()
