from pathlib import Path
from typing import Union, List, Tuple, Dict

from torch.utils import data as data

from .transforms import AvailableTransforms, get_transforms
from ..imageOperators import get_clean_image
from ..utils import get_module_logger

logger = get_module_logger(__name__)


def get_all_imgs_in_dir(dir_: Union[str, Path]):
    files_grabbed = []
    types = ("bmp", "jpg", "jpeg", "png")
    for type_ in types:
        files_grabbed.extend((dir_.glob(f"*.{type_}")))
    return files_grabbed


class DHazeDataset(data.Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 n_images: int,
                 transforms: List[Tuple[Union[AvailableTransforms, str], Dict]] = None):
        self.data_path = Path(root)
        self.n_images = n_images
        gt_dir = list(self.data_path.glob("*_GT"))[0]
        hazy_dir = list(self.data_path.glob("*_Hazy"))[0]
        clean_im_path = sorted(get_all_imgs_in_dir(gt_dir))
        self.clean_im_path = [p for p in clean_im_path if "Image" in p.name]
        self.depth_im_path = [p for p in clean_im_path if "Image" not in p.name]
        self.hazed_im_path = sorted(get_all_imgs_in_dir(hazy_dir))
        if n_images > 0:
            self.clean_im_path = self.clean_im_path[: self.n_images]
            self.hazed_im_path = self.hazed_im_path[: self.n_images]
            self.depth_im_path = self.depth_im_path[: self.n_images]

        self.transforms = get_transforms(transforms)
        logger.info(f"Using Transforms: {self.transforms}")
        return

    def __getitem__(self, item):
        _, clean_im = get_clean_image(self.clean_im_path[item], gray_scale=False)
        _, hazed_im = get_clean_image(self.hazed_im_path[item], gray_scale=False)

        if self.transforms is not None:
            return self.transforms(clean_im, hazed_im)
        else:
            return clean_im, hazed_im

    def __len__(self):
        return len(self.clean_im_path)
