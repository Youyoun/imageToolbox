import torch

FLOAT_TOL = 1e-3
BATCH_SIZE = 64
NDIM_X = 256
NDIM_Y = 256


def are_equal(v1, v2):
    return torch.isclose(v1, v2, rtol=FLOAT_TOL, atol=FLOAT_TOL).all()
