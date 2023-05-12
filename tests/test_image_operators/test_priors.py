import pytest
import torch

from tests.parameters import BATCH_SIZE, NDIM_X, NDIM_Y, are_equal
from toolbox.imageOperators import IdentityOperator, SmoothTotalVariation, Tychonov


class TestPriors:
    @staticmethod
    def test_tychonov_reg_specific_vector():
        x = torch.Tensor([[[2, 6, 2, 1], [4, 7, 1, 9], [5, 2, 0, 1], [2, 5, 6, 6]]])
        tych = Tychonov(IdentityOperator(), 0)
        assert tych.f(x) == 323
        assert are_equal(tych.grad(x), 2 * x)

    @staticmethod
    def test_tykhonov_reg_batched_vector():
        x = torch.randn(BATCH_SIZE, 1, NDIM_X, NDIM_Y)
        tych = Tychonov(IdentityOperator(), 0)
        grad_x = tych.grad(x)
        assert grad_x.ndim == 4
        assert are_equal(tych.f(x), x.pow(2).sum([1, 2, 3]))
        assert are_equal(grad_x, 2 * x)

    @staticmethod
    def test_tykhonov_reg_same_batched_vector():
        x = torch.Tensor([[[2, 6, 2, 1], [4, 7, 1, 9], [5, 2, 0, 1], [2, 5, 6, 6]]])
        x = x.repeat(BATCH_SIZE, 1, 1, 1)
        tych = Tychonov(IdentityOperator(), 0)
        assert are_equal(tych.f(x), torch.Tensor([323 for _ in range(BATCH_SIZE)]))
        assert are_equal(tych.grad(x), 2 * x)

    @staticmethod
    @pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
    def test_total_variation_specific_vector(eps):
        x = torch.Tensor([[[2, 6, 2, 1], [4, 7, 1, 9], [5, 2, 0, 1], [2, 5, 6, 6]]])
        grad_y = torch.Tensor([[[4, -4, -1, 0], [3, -6, 8, 0], [-3, -2, 1, 0], [3, 1, 0, 0]]])
        grad_x = torch.Tensor([[[2, 1, -1, 8], [1, -5, -1, -8], [-3, 3, 6, 5], [0, 0, 0, 0]]])
        tv_grad = torch.Tensor(
            [
                [
                    [-1.3413, 1.6216, 0.4408, -1.7053],
                    [-0.8172, 2.5990, -2.3417, 2.9920],
                    [1.7299, -1.6243, -1.8291, -1.8353],
                    [-1.7064, 0.8361, 1.9813, 0.9998],
                ]
            ]
        )  # Computed with EPS=1e-2
        tv = torch.sum(torch.sqrt(grad_x**2 + grad_y**2 + eps))
        tv_prior = SmoothTotalVariation(eps)
        assert tv_prior.f(x) == tv
        if eps == 1e-2:
            assert are_equal(tv_prior.grad(x), tv_grad)

    @staticmethod
    def test_tv_reg_batched_vector_grad_autograd():
        x = torch.randn(BATCH_SIZE, 1, NDIM_X, NDIM_Y)
        tv = SmoothTotalVariation()
        assert are_equal(tv.grad(x), tv.autograd_f(x))
