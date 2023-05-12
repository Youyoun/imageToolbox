from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch


class Splitters(ABC):
    @abstractmethod
    def split_tensor(self, *args, **kwargs):
        pass

    @abstractmethod
    def group_tensor(self, *args, **kwargs):
        pass


class ImageSplitter(Splitters):
    def __init__(self, n_patches: int):
        assert np.log2(n_patches) % 1 == 0, "Not a power of 2"

        self.ncols = n_patches // 2 if (n_patches // 4) % 2 == 0 else n_patches
        self.nrows = n_patches

    def split_tensor(
        self, tensor_: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int], List[torch.Size]]:
        patch_x_size = tensor_.shape[1] // (self.nrows // 2) + 1
        patch_y_size = tensor_.shape[2] // (max(self.ncols // 2, 2)) + 1

        patches = []
        sizes = []
        for x in range(0, tensor_.shape[1], patch_x_size):
            for y in range(0, tensor_.shape[2], patch_y_size):
                p = tensor_[..., x : x + patch_x_size, y : y + patch_y_size]
                sizes.append(p.shape)
                patches.append(
                    torch.nn.functional.pad(
                        p, (0, patch_y_size - p.shape[2], 0, patch_x_size - p.shape[1])
                    ).unsqueeze(0)
                )
        return torch.cat(patches, dim=0), (self.nrows // 2, self.ncols // 2), sizes

    def group_tensor(self, tensors: torch.Tensor, sizes: List[torch.Size]) -> torch.Tensor:
        y_size = sum([s[2] for s in sizes[: self.ncols // 2]])
        x_size = sum([s[1] for s in sizes[:: self.ncols // 2]])

        patch_x_size = x_size // (self.nrows // 2) + 1
        patch_y_size = y_size // (max(self.ncols // 2, 2)) + 1
        i = 0
        final = torch.zeros((tensors.shape[1], x_size, y_size))
        for x in range(0, x_size, patch_x_size):
            for y in range(0, y_size, patch_y_size):
                final[..., x : x + patch_x_size, y : y + patch_y_size] = tensors[
                    i, : sizes[i][0], : sizes[i][1], : sizes[i][2]
                ]
                i += 1
        return final


class ImageSplitterOverlapIgnored(Splitters):
    def __init__(self, inner_patch_size: int, outer_patch_size: int):
        assert outer_patch_size >= inner_patch_size
        self.inner_patch_size = inner_patch_size
        self.outer_patch_size = outer_patch_size
        self.padding = (outer_patch_size - inner_patch_size) // 2

    def split_tensor(self, tensor_: torch.Tensor) -> torch.Tensor:
        padded_tensor = torch.nn.functional.pad(
            tensor_.unsqueeze(0),
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        ).squeeze(0)
        patches = []
        dim_y, dim_x = tensor_.shape[2], tensor_.shape[1]
        for x in range(0, dim_x, self.inner_patch_size):
            for y in range(0, dim_y, self.inner_patch_size):
                offset_x = (
                    0  # If the patch is at the end and the padding doesn't provide enough pixels
                )
                offset_y = 0  # get more pixels from the other side
                if x + self.inner_patch_size > dim_x or y + self.inner_patch_size > dim_y:
                    offset_x = max(0, x + self.inner_patch_size - dim_x)
                    offset_y = max(0, y + self.inner_patch_size - dim_y)
                p = padded_tensor[
                    ...,
                    x - offset_x : x + self.inner_patch_size + 2 * self.padding,
                    y - offset_y : y + 2 * self.padding + self.inner_patch_size,
                ]
                patches.append(p.unsqueeze(0))
        return torch.cat(patches, dim=0)

    def group_tensor(
        self, tensors: torch.Tensor, orig_size: Union[torch.Size, Tuple[int, int, int]]
    ) -> torch.Tensor:
        dim_x = orig_size[1]
        dim_y = orig_size[2]
        final = torch.zeros((tensors.shape[1], dim_x, dim_y))
        i = 0
        for x in range(0, dim_x, self.inner_patch_size):
            for y in range(0, dim_y, self.inner_patch_size):
                offset_x = (
                    0  # If the patch is at the end and the padding doesn't provide enough pixels
                )
                offset_y = 0  # get more pixels from the other side
                if x + self.inner_patch_size > dim_x or y + self.inner_patch_size > dim_y:
                    offset_x = max(0, x + self.inner_patch_size - dim_x)
                    offset_y = max(0, y + self.inner_patch_size - dim_y)
                if self.padding > 0:
                    final[
                        ..., x : x + self.inner_patch_size, y : y + self.inner_patch_size
                    ] = tensors[
                        i,
                        ...,
                        self.padding + offset_x : -self.padding,
                        self.padding + offset_y : -self.padding,
                    ]
                else:
                    final[
                        ..., x : x + self.inner_patch_size, y : y + self.inner_patch_size
                    ] = tensors[i, ..., offset_x:, offset_y:]
                i += 1
        return final
