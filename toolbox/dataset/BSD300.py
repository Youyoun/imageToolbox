import glob
from pathlib import Path
from typing import Union, List, Tuple, Dict

import torch
import torchvision.transforms as T
from torch.utils import data as data

from ..imageOperators import get_clean_image, get_noisy_image, NoiseModes, get_noise_func, IdentityOperator, \
    BlurConvolution
from ..utils import get_module_logger
from .image_splitter import ImageSplitterOverlapIgnored
from .transforms import get_transforms, AvailableTransforms

logger = get_module_logger(__name__)

SCALES = [1.0, 0.9, 0.8, 0.7]


class BSD300PatchedNoisyDataset(data.Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 n_images: int,
                 inner_patch_size: int,
                 outer_patch_size: int,
                 blur_type: Union[str,None],
                 kernel_size: int,
                 kernel_std: float,
                 noise_mode: Union[str, NoiseModes],
                 gaussian_noise_mean: float,
                 gaussian_noise_std: float,
                 poisson_noise_scale: float,
                 is_train: bool = True,
                 transforms: List[Tuple[Union[AvailableTransforms, str], Dict]] = None):
        self.data_path = Path(root)
        self.n_images = n_images
        self.splitter = ImageSplitterOverlapIgnored(inner_patch_size, outer_patch_size)
        self.is_train = is_train
        logger.info(f"Patch sizes: {inner_patch_size, outer_patch_size}")
        self.patches = []
        self.noisy_patches = []
        self.im_idxs = []
        self.im_sizes = []
        self.load_dataset()

        # Noise info
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.kernel_std = kernel_std
        self.noise_mode = noise_mode
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_std = gaussian_noise_std
        self.poisson_noise_scale = poisson_noise_scale

        self.transforms = get_transforms(transforms)
        logger.info(f"Using Transforms: {self.transforms}")
        return

    def reload_dataset(self):
        logger.info("Reloading Dataset")
        self.patches = []
        self.noisy_patches = []
        self.im_idxs = []
        self.im_sizes = []
        self.load_dataset()

    def load_dataset(self):
        im_paths = glob.glob(str(self.data_path / "*.jpg"))[:self.n_images]
        for i, p in enumerate(im_paths):
            gray_image, gray_t = get_clean_image(p)
            noisy_image, noisy_t, BlurrOp = get_noisy_image(gray_t,
                                                            self.blur_type,
                                                            self.kernel_size,
                                                            self.kernel_std,
                                                            self.noise_mode,
                                                            self.gaussian_noise_mean,
                                                            self.gaussian_noise_std,
                                                            self.poisson_noise_scale)
            patches = self.splitter.split_tensor(gray_t)
            self.patches.extend(list(patches.unsqueeze(1)))
            patches = self.splitter.split_tensor(noisy_t)
            self.noisy_patches.extend(list(patches.unsqueeze(1)))
            self.im_idxs.extend([i for _ in range(len(patches))])
            self.im_sizes.append(gray_t.shape)

        if self.is_train:
            for i, p in enumerate(im_paths):
                gray_image, gray_t = get_clean_image(p)
                for s in SCALES:
                    if s == 1:
                        continue
                    gray_t = T.F.resize(gray_t,
                                        [int(gray_t.shape[-2] * s), int(gray_t.shape[-1] * s)],
                                        interpolation=T.InterpolationMode.NEAREST)
                    noisy_image, noisy_t, BlurrOp = get_noisy_image(gray_t,
                                                                    self.blur_type,
                                                                    self.kernel_size,
                                                                    self.kernel_std,
                                                                    self.noise_mode,
                                                                    self.gaussian_noise_mean,
                                                                    self.gaussian_noise_std,
                                                                    self.poisson_noise_scale)
                    patches = self.splitter.split_tensor(gray_t)
                    self.patches.extend(list(patches.unsqueeze(1)))
                    patches = self.splitter.split_tensor(noisy_t)
                    self.noisy_patches.extend(list(patches.unsqueeze(1)))
                    self.im_idxs.extend([10000 for _ in range(len(patches))])
                    self.im_sizes.append(gray_t.shape)
        self.patches = torch.cat(self.patches, dim=0)
        self.noisy_patches = torch.cat(self.noisy_patches, dim=0)
        self.im_idxs = torch.Tensor(self.im_idxs)

    def __getitem__(self, item):
        if self.transforms is not None:
            return self.transforms(self.patches[item], self.noisy_patches[item])
        else:
            return self.patches[item], self.noisy_patches[item]

    def __len__(self):
        return len(self.patches)

    def get_image_patches(self, im_idx):
        if im_idx > self.n_images:
            raise ValueError(f"{im_idx} not in loaded images")
        return self.patches[self.im_idxs == im_idx], self.noisy_patches[self.im_idxs == im_idx]

    def get_image(self, im_idx):
        clean = self.group_patches(self.patches[self.im_idxs == im_idx], self.im_sizes[im_idx])
        noisy = self.group_patches(self.noisy_patches[self.im_idxs == im_idx], self.im_sizes[im_idx])
        return clean, noisy

    def len_images(self):
        return self.n_images

    def group_patches(self, patches, im_size):
        return self.splitter.group_tensor(patches, im_size)


class BSD300NoisyDataset(data.Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 n_images: int,
                 blur_type: Union[str, None],
                 kernel_size: int,
                 kernel_std: float,
                 noise_mode: Union[str, NoiseModes],
                 gaussian_noise_mean: float,
                 gaussian_noise_std: float,
                 poisson_noise_scale: float,
                 is_train: bool = True,
                 transforms: List[Tuple[Union[AvailableTransforms, str], Dict]] = None):
        self.data_path = Path(root)
        self.n_images = n_images
        self.is_train = is_train
        self.im_path = glob.glob(str(self.data_path / "*.jpg"))[:self.n_images]

        # Noise info
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.kernel_std = kernel_std
        self.noise_mode = noise_mode
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_std = gaussian_noise_std
        self.poisson_noise_scale = poisson_noise_scale

        if blur_type is None:
            self.blur_op = IdentityOperator()
        else:
            self.blur_op = BlurConvolution(kernel_size, blur_type, kernel_std)
        self.noise_fn = get_noise_func(self.noise_mode,
                                       self.gaussian_noise_mean,
                                       self.gaussian_noise_std,
                                       scale_poisson=self.poisson_noise_scale)

        self.transforms = get_transforms(transforms)
        logger.info(f"Using Transforms: {self.transforms}")
        return

    def __getitem__(self, item):
        _, clean_im = get_clean_image(self.im_path[item])
        noisy_im = self.noise_fn(self.blur_op @ clean_im)
        if self.transforms is not None:
            return self.transforms(clean_im, noisy_im)
        else:
            return clean_im, noisy_im

    def __len__(self):
        return len(self.im_path)
