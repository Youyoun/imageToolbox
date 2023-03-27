from pathlib import Path
from typing import Union, List, Tuple, Dict, Callable

import lmdb
import six
from torch.utils import data as data

from tqdm import tqdm

from .lmdb_utils import loads_data
from .transforms import get_transforms, AvailableTransforms
from ..imageOperators import get_clean_image
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class GenericLMDBDataset(data.Dataset):
    def __init__(self,
                 db_path: Union[str, Path],
                 clean_transform_fn: Callable,
                 n_images: int = 0,
                 augments: List[Tuple[Union[AvailableTransforms, str], Dict]] = None,
                 load_in_memory: bool = False):
        self.images = None
        self.noisy_images = None

        self.n_images = n_images
        if self.n_images > 0:
            logger.info(f"Using {self.n_images} images")
        self.db_path = Path(db_path)
        self.env = lmdb.open(db_path, subdir=False,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform_fn = clean_transform_fn
        self.transforms = get_transforms(augments)
        logger.info(f"Using Transforms: {self.transforms}")

        self.load_in_memory = load_in_memory
        if self.load_in_memory:
            logger.info("Loading images in memory")
            self.images = [self.get_image(i) for i in tqdm(range(self.length))]
            self.noisy_images = [self.transform_fn(im) for im in tqdm(self.images)]

    def get_image(self, key):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[key])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        _, img = get_clean_image(buf)
        return img

    def __getitem__(self, item):
        if self.load_in_memory:
            clean_im, noisy_im = self.images[item], self.noisy_images[item]
        else:
            clean_im = self.get_image(item)
            noisy_im = self.transform_fn(clean_im)
        if self.transforms is not None:
            return self.transforms(clean_im, noisy_im)
        else:
            return clean_im, noisy_im

    def __len__(self):
        return self.n_images if self.n_images > 0 else self.length
