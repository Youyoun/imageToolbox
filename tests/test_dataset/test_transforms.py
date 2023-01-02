from pathlib import Path

import pytest
import torch
import torchvision.transforms as ttransforms
from torchvision.transforms import functional as F
from PIL import Image

from toolbox.dataset import Random90Rotation, RandomVerticalFlip, RandomHorizontalFlip, get_transforms, Compose, \
    AvailableTransforms, RandomCrop, CenterCrop
from tests.parameters import are_equal

"""
Test:
- Compose
- RandomHorizontalFlip
- RandomVerticalFlip
- Random90Rotation
- RandomCrop
- get_transform()
"""


class TestTransforms:
    """
    Tests are only implemented in the black and white setting (n_channels = 1)
    """

    @staticmethod
    def load_image_and_transform(transform):
        cat_im = Image.open(Path(__file__).parent.parent / "test_image_operators/chelseaColor.png").convert('L')
        cat_t = ttransforms.ToTensor()(cat_im)
        flipped_cat_t, flipped_cat_t_2 = transform(cat_t, cat_t)
        return cat_t, flipped_cat_t, flipped_cat_t_2

    @staticmethod
    def test_random_horizontal_flip_p1():
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomHorizontalFlip(p=1.0))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, ttransforms.RandomHorizontalFlip(p=1.0)(cat_t))

    @staticmethod
    def test_random_horizontal_flip_p0():
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomHorizontalFlip(p=0.0))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, cat_t)

    @staticmethod
    @pytest.mark.parametrize("p", [i / 10 for i in range(1, 10)])
    def test_random_horizontal_flip_randomp(p: float):
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomHorizontalFlip(p=p))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)

    @staticmethod
    def test_random_vertical_flip_p1():
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomVerticalFlip(p=1.0))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, ttransforms.RandomVerticalFlip(p=1.0)(cat_t))

    @staticmethod
    def test_random_vertical_flip_p0():
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomVerticalFlip(p=0.0))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, cat_t)

    @staticmethod
    @pytest.mark.parametrize("p", [i / 10 for i in range(1, 10)])
    def test_random_vertical_flip_randomp(p: float):
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomVerticalFlip(p=p))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)

    @staticmethod
    def test_random_rotation_p1():
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(Random90Rotation(p=1.0))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, F.rotate(cat_t, 90))

    @staticmethod
    def test_random_rotation_p0():
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(Random90Rotation(p=0.0))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, cat_t)

    @staticmethod
    @pytest.mark.parametrize("p", [i / 10 for i in range(1, 10)])
    def test_random_horizontal_flip_randomp(p: float):
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(Random90Rotation(p=p))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)

    @staticmethod
    def test_compose():
        transform = Compose([Random90Rotation(p=1.0), RandomHorizontalFlip(p=1.0), RandomVerticalFlip(p=1.0)])
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(transform)
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
        assert are_equal(flipped_cat_t, ttransforms.RandomVerticalFlip(p=1)(
            ttransforms.RandomHorizontalFlip(p=1)(F.rotate(cat_t, 90))))

    @staticmethod
    @pytest.mark.parametrize("p", [i / 10 for i in range(1, 10)])
    def test_compose_random_p(p: float):
        transform = Compose([Random90Rotation(p=p), RandomHorizontalFlip(p=p), RandomVerticalFlip(p=p)])
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(transform)
        assert are_equal(flipped_cat_t, flipped_cat_t_2)

    @staticmethod
    @pytest.mark.parametrize("p", [i / 10 for i in range(1, 10)])
    def test_get_transform_with_object(p: float):
        transform = get_transforms([(AvailableTransforms.Random90Rotation, {"p": p}),
                                    (AvailableTransforms.RandomVerticalFlip, {"p": p})])
        assert isinstance(transform.transforms[0], Random90Rotation) and transform.transforms[0].p == p
        assert isinstance(transform.transforms[1], RandomVerticalFlip) and transform.transforms[1].flip_prob == p

    @staticmethod
    @pytest.mark.parametrize("size", [64, 128, 256, 299])
    def test_random_crop(size):
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(RandomCrop(size=size))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)

    @staticmethod
    @pytest.mark.parametrize("size", [64, 128, 256, 299])
    def test_center_crop(size):
        cat_t, flipped_cat_t, flipped_cat_t_2 = TestTransforms.load_image_and_transform(CenterCrop(size=size))
        assert are_equal(flipped_cat_t, flipped_cat_t_2)
