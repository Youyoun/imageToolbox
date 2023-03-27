import glob
from pathlib import Path
from typing import Union, List, Tuple, Dict, Callable

from torch.utils import data as data

from .transforms import get_transforms, AvailableTransforms
from ..imageOperators import get_clean_image
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class GenericDataset(data.Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 n_images: int,
                 clean_transform_fn: Callable,
                 is_train: bool = True,
                 augments: List[Tuple[Union[AvailableTransforms, str], Dict]] = None,
                 load_in_memory: bool = False):
        self.data_path = Path(root)
        self.n_images = n_images
        self.is_train = is_train
        self.im_path = glob.glob(str(self.data_path / "*.jpg")) + glob.glob(str(self.data_path / "*.JPEG"))
        if n_images > 0:
            self.im_path = self.im_path[: self.n_images]
        self.images = None
        self.noisy_images = None

        self.transform_fn = clean_transform_fn

        self.transforms = get_transforms(augments)
        logger.info(f"Using Transforms: {self.transforms}")

        self.load_in_memory = load_in_memory
        if self.load_in_memory:
            logger.info("Loading images in memory")
            self.images = [get_clean_image(im_path)[1] for im_path in self.im_path]
            self.noisy_images = [self.transform_fn(im) for im in self.images]

    def __getitem__(self, item):
        if self.load_in_memory:
            clean_im, noisy_im = self.images[item], self.noisy_images[item]
        else:
            _, clean_im = get_clean_image(self.im_path[item])
            noisy_im = self.transform_fn(clean_im)
        if self.transforms is not None:
            return self.transforms(clean_im, noisy_im)
        else:
            return clean_im, noisy_im

    def __len__(self):
        return len(self.im_path)
