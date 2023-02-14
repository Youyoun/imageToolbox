from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as ttransforms
from PIL import Image
from skimage.metrics import structural_similarity

from toolbox.metrics import compute_relative_difference, mean_absolute_error, mean_squared_error, SNR, PSNR, SSIM


def test_compute_relative_difference():
    tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
    tensor2 = np.array([[2, 3, 4], [5, 6, 7]])
    assert compute_relative_difference(tensor1, tensor2) == 1.0
    assert compute_relative_difference(tensor1, tensor2, tensor1) == 1.0
    assert compute_relative_difference(tensor1, tensor2, tensor2) == 1.0


def test_compute_relative_difference_random():
    tensor1 = np.random.rand(100, 100)
    tensor2 = np.random.rand(100, 100)
    assert compute_relative_difference(tensor1, tensor2) == (
            np.linalg.norm(tensor1 - tensor2, ord='fro') / np.linalg.norm(tensor1, ord='fro'))
    assert compute_relative_difference(tensor1, tensor2, tensor1) == (
            np.linalg.norm(tensor1 - tensor2, ord='fro') / np.linalg.norm(tensor1, ord='fro'))
    assert compute_relative_difference(tensor1, tensor2, tensor2) == (
            np.linalg.norm(tensor1 - tensor2, ord='fro') / np.linalg.norm(tensor2, ord='fro'))


def test_mean_absolute_error():
    tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
    tensor2 = np.array([[2, 3, 4], [5, 6, 7]])
    assert mean_absolute_error(tensor1, tensor2) == 1.0


def test_mean_absolute_error_random():
    tensor1 = np.random.rand(100, 100)
    tensor2 = np.random.rand(100, 100)
    assert mean_absolute_error(tensor1, tensor2) == np.mean(np.abs(tensor1 - tensor2))


def test_mean_squared_error():
    tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
    tensor2 = np.array([[2, 3, 4], [5, 6, 7]])
    assert pytest.approx(mean_squared_error(tensor1, tensor2)) == 1.0


def test_mean_squared_error_random():
    tensor1 = np.random.rand(100, 100)
    tensor2 = np.random.rand(100, 100)
    assert pytest.approx(mean_squared_error(tensor1, tensor2)) == np.mean(np.square(tensor1 - tensor2))


def test_snr():
    tensor1 = np.random.rand(100, 100)
    assert SNR(tensor1) == 10 * np.log10(np.mean(tensor1) / np.std(tensor1))


def test_psnr_1_range_image():
    cat_im = Image.open(Path(__file__).parent / "chelseaColor.png").convert('L')
    cat_t = ttransforms.ToTensor()(cat_im)
    cat_t_noisy = torch.clamp(cat_t + 0.1 * torch.randn(cat_t.shape), 0, 1)
    assert pytest.approx(PSNR(cat_t, cat_t_noisy)) == pytest.approx(
        10 * torch.log10(1 / torch.mean((cat_t - cat_t_noisy) ** 2)).item())


def test_psnr_2_range_image():
    cat_im = Image.open(Path(__file__).parent / "chelseaColor.png").convert('L')
    cat_t = ttransforms.Normalize(0.5, 0.5)(ttransforms.ToTensor()(cat_im))
    cat_t_noisy = torch.clamp(cat_t + 0.1 * torch.randn(cat_t.shape), -1, 1)
    assert pytest.approx(PSNR(cat_t, cat_t_noisy)) == pytest.approx(
        10 * torch.log10(4 / torch.mean((cat_t - cat_t_noisy) ** 2)).item())


def test_ssim_1_range_image():
    cat_im = Image.open(Path(__file__).parent / "chelseaColor.png").convert('L')
    cat_t = ttransforms.ToTensor()(cat_im).squeeze()
    cat_t_noisy = torch.clamp(cat_t + 0.1 * torch.randn(cat_t.shape), 0, 1)
    assert pytest.approx(SSIM(cat_t, cat_t_noisy)) == structural_similarity(cat_t.numpy(), cat_t_noisy.numpy())


def test_ssim_2_range_image():
    cat_im = Image.open(Path(__file__).parent / "chelseaColor.png").convert('L')
    cat_t = ttransforms.Normalize(0.5, 0.5)(ttransforms.ToTensor()(cat_im)).squeeze()
    cat_t_noisy = torch.clamp(cat_t + 0.1 * torch.randn(cat_t.shape), -1, 1)
    assert pytest.approx(SSIM(cat_t, cat_t_noisy)) == structural_similarity(cat_t.numpy(), cat_t_noisy.numpy())
