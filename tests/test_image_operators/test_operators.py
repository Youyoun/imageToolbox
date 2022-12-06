from pathlib import Path

import pytest
import torch
from skimage.filters import gaussian

from toolbox.imageOperators import GaussianBlurFFT, BlurConvolution, Kernels, IdentityOperator, Gradient, Directions, \
    get_clean_image
from tests.parameters import BATCH_SIZE, NDIM_X, NDIM_Y, are_equal


class TestImageGradient:
    SPECIFIC_X = torch.Tensor([[[2, 6, 2, 1],
                                [4, 7, 1, 9],
                                [5, 2, 0, 1],
                                [2, 5, 6, 6]]])
    SPECIFIC_X_GRAD_Y = torch.Tensor([[[4, -4, -1, 0],
                                       [3, -6, 8, 0],
                                       [-3, -2, 1, 0],
                                       [3, 1, 0, 0]]])
    SPECIFIC_X_GRAD_X = torch.Tensor([[[2, 1, -1, 8],
                                       [1, -5, -1, -8],
                                       [-3, 3, 6, 5],
                                       [0, 0, 0, 0]]])

    SPECIFIC_X_GRAD_YT = torch.Tensor([[[-2., -4., 4., 1.],
                                        [-4., -3., 6., -8.],
                                        [-5., 3., 2., -1.],
                                        [-2., -3., -1., 0.]]])
    SPECIFIC_X_GRAD_XT = torch.Tensor([[[-2, -6, -2, -1],
                                        [-2, -1, 1, -8],
                                        [-1, 5, 1, 8],
                                        [3, -3, -6, -5]]])

    @staticmethod
    def test_gradient_specific_single_vector():
        assert are_equal(TestImageGradient.SPECIFIC_X_GRAD_X,
                         Gradient(TestImageGradient.SPECIFIC_X, Directions.X))
        assert are_equal(TestImageGradient.SPECIFIC_X_GRAD_Y,
                         Gradient(TestImageGradient.SPECIFIC_X, Directions.Y))

    @staticmethod
    def test_gradient_transpose_specific_single_vector():
        assert are_equal(TestImageGradient.SPECIFIC_X_GRAD_XT,
                         Gradient.T(TestImageGradient.SPECIFIC_X, Directions.X))
        assert are_equal(TestImageGradient.SPECIFIC_X_GRAD_YT,
                         Gradient.T(TestImageGradient.SPECIFIC_X, Directions.Y))

    @staticmethod
    def test_gradient_random_single_vector():
        x = torch.randn(1, NDIM_X, NDIM_Y)
        grad_x = Gradient(x, Directions.X)
        grad_y = Gradient(x, Directions.Y)
        for i in range(NDIM_X - 1):
            for j in range(NDIM_Y - 1):
                assert x[:, i + 1, j] - x[:, i, j] == grad_x[:, i, j]
                assert x[:, i, j + 1] - x[:, i, j] == grad_y[:, i, j]

    @staticmethod
    def test_gradient_transpose_random_single_vector():
        x = torch.randn(1, NDIM_X, NDIM_Y)
        grad_xt = Gradient.T(x, Directions.X)
        grad_yt = Gradient.T(x, Directions.Y)
        for i in range(1, NDIM_X):
            for j in range(1, NDIM_Y):
                assert x[:, i - 1, j] - x[:, i, j] == grad_xt[:, i, j]
                assert x[:, i, j - 1] - x[:, i, j] == grad_yt[:, i, j]

    @staticmethod
    def test_gradient_random_batch_vector():
        x = torch.randn(BATCH_SIZE, 1, NDIM_X, NDIM_Y)
        grad_x = Gradient(x, Directions.X)
        grad_y = Gradient(x, Directions.Y)
        for i in range(NDIM_X - 1):
            for j in range(NDIM_Y - 1):
                assert are_equal(x[:, :, i + 1, j] - x[:, :, i, j], grad_x[:, :, i, j])
                assert are_equal(x[:, :, i, j + 1] - x[:, :, i, j], grad_y[:, :, i, j])

    @staticmethod
    def test_gradient_transpose_random_batch_vector():
        x = torch.randn(BATCH_SIZE, 1, NDIM_X, NDIM_Y)
        grad_xt = Gradient.T(x, Directions.X)
        grad_yt = Gradient.T(x, Directions.Y)
        for i in range(1, NDIM_X):
            for j in range(1, NDIM_Y):
                assert are_equal(x[:, :, i - 1, j] - x[:, :, i, j], grad_xt[:, :, i, j])
                assert are_equal(x[:, :, i, j - 1] - x[:, :, i, j], grad_yt[:, :, i, j])


class TestBlurOperators:
    @staticmethod
    @pytest.mark.parametrize('std', [1, 1.5, 2, 2.5, 3, 5])
    def test_fft_blur(std):  # There's a problem with image that have a pair dimension.
        _, x = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        blurr = GaussianBlurFFT(31, std)
        blurred = blurr @ x
        np_blur = torch.from_numpy(gaussian(x, sigma=std, mode="nearest"))
        assert are_equal(np_blur, blurred)

    @staticmethod
    @pytest.mark.parametrize('std', [0.5, 1, 1.5, 2, 2.5, 3])
    def test_conv_gaussian_blur(std):
        x = torch.randn(BATCH_SIZE, 1, NDIM_X, NDIM_Y)
        blurr = BlurConvolution(31, Kernels.GAUSSIAN, std)
        blurred = blurr @ x
        blurred_t = blurr.T @ x
        for i in range(BATCH_SIZE):
            np_blur = torch.from_numpy(gaussian(x[i], sigma=std, mode="nearest"))
            assert are_equal(np_blur, blurred[i])
            assert are_equal(np_blur, blurred_t[i])
        assert blurred.ndim == 4

    @staticmethod
    @pytest.mark.parametrize('std_', [0.5, 1, 1.5, 2, 2.5, 3])
    def test_conv_blur_single_image(std_):
        x = torch.randn(1, NDIM_X, NDIM_Y)
        blurr = BlurConvolution(31, Kernels.GAUSSIAN, std_)
        blurred = blurr @ x
        assert blurred.ndim == 3
        np_blur = torch.from_numpy(gaussian(x, sigma=std_, mode="nearest"))
        assert are_equal(np_blur, blurred)

    @staticmethod
    def test_identity_batch():
        x = torch.randn(BATCH_SIZE, 1, NDIM_X, NDIM_Y)
        id = IdentityOperator()
        assert are_equal(id @ x, x)

    @staticmethod
    def test_identity_single():
        x = torch.randn(1, NDIM_X, NDIM_Y)
        id = IdentityOperator()
        assert are_equal(id @ x, x)
