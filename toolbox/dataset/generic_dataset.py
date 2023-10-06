import glob
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import tqdm
from torch.utils import data as data

from ..imageOperators import get_clean_image
from ..utils import get_module_logger
from .transforms import AvailableTransforms, get_transforms

logger = get_module_logger(__name__)


class GenericDataset(data.Dataset):
    """
    Loads images from a folder and applies a certain transformation to them (add noise, blur image, etc.).
    The images provided are in the range 0 and 1.

    :param root: Path to the folder containing the images.
    :param n_images: Number of images to load. If 0, all images are loaded.
    :param clean_transform_fn: Function to apply to the clean images.
    :param augments: List of tuples (transform, kwargs) to apply to the images.
    :param load_in_memory: If True, the images are loaded in memory.
    :param colorized: If True, the images are loaded in color.
    """

    def __init__(
        self,
        root: Union[str, Path],
        n_images: int,
        clean_transform_fn: Callable,
        augments: List[Tuple[Union[AvailableTransforms, str], Dict]] = None,
        load_in_memory: bool = False,
        colorized: bool = False,
    ):
        self.data_path = Path(root)
        self.n_images = n_images
        self.im_path = glob.glob(str(self.data_path / "*.jpg")) + glob.glob(
            str(self.data_path / "*.JPEG")
        )
        if n_images > 0:
            logger.info(f"Loading {n_images} images")
            self.im_path = self.im_path[: self.n_images]
        self.images = None
        self.noisy_images = None
        self.colorized = colorized

        self.transform_fn = clean_transform_fn

        self.transforms = get_transforms(augments)
        logger.info(f"Using Transforms: {self.transforms}")

        self.load_in_memory = load_in_memory
        if self.load_in_memory:
            logger.info("Loading images in memory")
            self.images = [
                get_clean_image(im_path, not self.colorized)[1]
                for im_path in tqdm.tqdm(self.im_path)
            ]
            self.noisy_images = [self.transform_fn(im) for im in tqdm.tqdm(self.images)]

    def __getitem__(self, item):
        if self.load_in_memory:
            clean_im, noisy_im = self.images[item], self.noisy_images[item]
            if self.transforms is not None:
                return self.transforms(clean_im, noisy_im)
            return clean_im, noisy_im
        else:
            _, clean_im = get_clean_image(self.im_path[item], not self.colorized)
            if self.transforms is not None:
                return self.transforms(clean_im, self.transform_fn(clean_im))
            return clean_im, self.transform_fn(clean_im)

    def get_image_path(self, item):
        return self.im_path[item]

    def __len__(self):
        return len(self.im_path)
