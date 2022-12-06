from pathlib import Path

import pytest
import torch
import torchvision.transforms as ttransforms
from PIL import Image
from scipy.stats import shapiro

from toolbox.imageOperators import get_noise_func, NoiseModes, get_clean_image
from .parameters import BATCH_SIZE, NDIM_X, NDIM_Y, are_equal

POISSON_SIMILARITY_THRESH = 2


class TestCleanImageLoading:
    """
    Tests are only implemented in the black and white setting (n_channels = 1)
    """

    @staticmethod
    def test_get_clean_image():
        cat_im = Image.open(Path(__file__).parent / "chelseaColor.png").convert('L')
        cat_t = ttransforms.ToTensor()(cat_im)
        cat_im_test, cat_t_test = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        assert are_equal(cat_t_test, cat_t)

    @staticmethod
    def test_load_image_pil_tensor_equality():
        cat_im, cat_t = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        assert are_equal(ttransforms.ToTensor()(cat_im), cat_t)

    @staticmethod
    def test_load_image_is_grayscale():
        _, cat_t = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        assert cat_t.ndim == 3 and cat_t.shape[0] == 1

    @staticmethod
    def test_load_image_bounds():
        _, cat_t = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        assert cat_t.max() < 1 and cat_t.min() > 0


class TestNoiseApplicationBW:
    """
    Tests are only implemented in the black and white setting (n_channels = 1)
    """

    @staticmethod
    @pytest.mark.parametrize("std_", [0.01, 0.05, 0.1, 0.12])
    def test_gauss_noise_single_image(std_):
        mean_ = 0
        _, x = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        noise_func = get_noise_func(NoiseModes.GAUSSIAN, mean_, std_)
        y = noise_func(x)
        noise = y - x
        assert not are_equal(y, x)
        assert are_equal(noise.mean(), torch.Tensor([mean_]))
        assert are_equal(noise.std(), torch.Tensor([std_]))
        assert shapiro(noise).pvalue > 0.05

    @staticmethod
    @pytest.mark.parametrize("std_", [0.01, 0.05, 0.1, 0.12])
    def test_gauss_noise_batch_image(std_):
        mean_ = 0
        _, x = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        im_shape = x.shape
        x = x.repeat(BATCH_SIZE, 1, 1, 1)
        assert x.shape == (BATCH_SIZE, *im_shape)
        noise_func = get_noise_func(NoiseModes.GAUSSIAN, mean_, std_)
        y = noise_func(x)
        noise = y - x
        assert y.ndim == 4
        assert not are_equal(y, x)
        assert are_equal(noise.mean(dim=[1, 2, 3]), torch.ones(BATCH_SIZE) * mean_)
        assert are_equal(noise.std(dim=[1, 2, 3]), torch.ones(BATCH_SIZE) * std_)
        for i in range(BATCH_SIZE):
            assert shapiro(noise[i]).pvalue > 0.05

    @staticmethod
    @pytest.mark.parametrize("scale", [100, 200, 300, 500])
    def test_poisson_single_image(scale):
        _, x = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        noise_func = get_noise_func(NoiseModes.POISSON, scale_poisson=scale)
        y = noise_func(x)
        scaled_x = (x * scale).round()
        scaled_y = (y * scale).round()
        for i in range(1, scale - 1):
            same_intensity_pixels = scaled_y[scaled_x == i]
            if len(same_intensity_pixels) == 0:
                continue
            assert torch.abs(same_intensity_pixels.mean() - i) / i < POISSON_SIMILARITY_THRESH
            assert torch.abs(same_intensity_pixels.std() ** 2 - i) / i < POISSON_SIMILARITY_THRESH
        assert not are_equal(y, x)

    @staticmethod
    @pytest.mark.parametrize("scale", [100, 200, 300, 500])
    def test_poisson_batch_image(scale):
        _, x = get_clean_image(Path(__file__).parent / "chelseaColor.png")
        im_shape = x.shape
        x = x.repeat(BATCH_SIZE, 1, 1, 1)
        assert x.shape == (BATCH_SIZE, *im_shape)
        noise_func = get_noise_func(NoiseModes.POISSON, scale_poisson=scale)
        y = noise_func(x)
        assert y.ndim == 4
        for e in range(BATCH_SIZE):
            scaled_x = (x[e] * scale).round()
            scaled_y = (y[e] * scale).round()
            for i in range(1, scale - 1):
                same_intensity_pixels = scaled_y[scaled_x == i]
                if len(same_intensity_pixels) < 20:
                    continue
                mean_ = same_intensity_pixels.mean()
                assert torch.abs(mean_ - i) / mean_ < POISSON_SIMILARITY_THRESH, i
                assert torch.abs(same_intensity_pixels.var() - i) / mean_ < POISSON_SIMILARITY_THRESH, i
        assert not are_equal(y, x)
