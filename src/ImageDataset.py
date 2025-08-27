"""
This file is based on ImageDataset from MONAI, modified to support different image and segmentation readers.
"""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from monai.config import DtypeLike
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed
from torch.utils.data import Dataset


class ImageDataset(Dataset, Randomizable):
    """
    Loads image/segmentation pairs of files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    The difference between this dataset and `ArrayDataset` is that this dataset can apply transform chain to images
    and segs and return both the images and metadata, and no need to specify transform to load images from files.
    For more information, please see the image_dataset demo in the MONAI tutorial repo,
    https://github.com/Project-MONAI/tutorials/blob/master/modules/image_dataset.ipynb
    """

    def __init__(
        self,
        image_files: Sequence[str],
        seg_files: Sequence[str] | None = None,
        labels: Sequence[float] | None = None,
        transform: Callable | None = None,
        seg_transform: Callable | None = None,
        label_transform: Callable | None = None,
        image_only: bool = True,
        transform_with_metadata: bool = False,
        dtype: DtypeLike = np.float32,
        reader: ImageReader | str | None = None,
        seg_reader: ImageReader | str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files: list of image filenames.
            seg_files: if in segmentation task, list of segmentation filenames.
            labels: if in classification task, list of classification labels.
            transform: transform to apply to image arrays.
            seg_transform: transform to apply to segmentation arrays.
            label_transform: transform to apply to the label data.
            image_only: if True return only the image volume, otherwise, return image volume and the metadata.
            transform_with_metadata: if True, the metadata will be passed to the transforms whenever possible.
            dtype: if not None convert the loaded image to this data type.
            reader: register reader to load image file and metadata, if None, will use the default readers.
                If a string of reader name provided, will construct a reader object with the `*args` and `**kwargs`
                parameters, supported reader name: "NibabelReader", "PILReader", "ITKReader", "NumpyReader"
            seg_reader: register reader to load segmentation file and metadata, if None, will use `reader`.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        Raises:
            ValueError: When ``seg_files`` length differs from ``image_files``

        """

        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.label_transform = label_transform
        if image_only and transform_with_metadata:
            raise ValueError("transform_with_metadata=True requires image_only=False.")
        self.image_only = image_only
        self.transform_with_metadata = transform_with_metadata
        self.loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        self.seg_loader = LoadImage(
            seg_reader or reader, image_only, dtype, *args, **kwargs
        )
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.image_files)

    def randomize(self, data: Any | None = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        meta_data, seg_meta_data, seg, label = None, None, None, None

        # Handle slice tuples for 2D slice loading
        image_file = self.image_files[index]
        seg_file = self.seg_files[index] if self.seg_files is not None else None

        # Check if we have slice tuples (file_path, slice_index)
        image_slice_idx = None
        seg_slice_idx = None

        if isinstance(image_file, tuple) and len(image_file) == 2:
            image_file, image_slice_idx = image_file

        if seg_file is not None and isinstance(seg_file, tuple) and len(seg_file) == 2:
            seg_file, seg_slice_idx = seg_file

        # load data and optionally meta
        if self.image_only:
            img = self.loader(image_file)
            if seg_file is not None:
                seg = self.seg_loader(seg_file)
        else:
            img, meta_data = self.loader(image_file)
            if seg_file is not None:
                seg, seg_meta_data = self.seg_loader(seg_file)

                # Copy relevant spatial metadata from image to segmentation
                if meta_data and seg_meta_data:
                    for attribute in meta_data:
                        if attribute not in seg_meta_data:
                            seg_meta_data[attribute] = meta_data[attribute]

        # Helper to derive 2D affine from 3D affine for slice k (slice along last axis)
        def build_2d_affine_from_3d(affine3d: np.ndarray, k: int) -> np.ndarray:
            """Construct 2D affine from 3D affine for slice k (slice along last axis).
            Ensures float64 dtype; returns identity if affine is missing or invalid.
            """
            I2 = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            if not isinstance(affine3d, np.ndarray):
                return I2
            A = np.asarray(affine3d, dtype=np.float64)
            if A.shape != (4, 4):
                return I2
            k = int(k)
            B = np.array(
                [
                    [A[0, 0], A[0, 1], A[0, 3] + k * A[0, 2]],
                    [A[1, 0], A[1, 1], A[1, 3] + k * A[1, 2]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            return B

        # Extract specific slice if slice index is provided
        if image_slice_idx is not None and len(img.shape) >= 3:
            # Slice along last dim (H, W, D -> H, W)
            img = img[..., image_slice_idx]
            if isinstance(meta_data, dict):
                aff = meta_data.get("affine", None)
                if aff is None:
                    aff = meta_data.get("original_affine", None)
                if isinstance(aff, torch.Tensor):
                    aff = aff.detach().cpu().numpy()
                affine2d = build_2d_affine_from_3d(aff, image_slice_idx)
                meta_data["affine"] = affine2d
                meta_data["original_affine"] = affine2d
                meta_data["spatial_shape"] = np.asarray(img.shape[-2:], dtype=np.int64)

        if seg is not None and seg_slice_idx is not None and len(seg.shape) >= 3:
            seg = seg[..., seg_slice_idx]
            if isinstance(seg_meta_data, dict):
                aff = seg_meta_data.get("affine", None)
                if aff is None:
                    aff = seg_meta_data.get("original_affine", None)
                if isinstance(aff, torch.Tensor):
                    aff = aff.detach().cpu().numpy()
                affine2d = build_2d_affine_from_3d(aff, seg_slice_idx)
                seg_meta_data["affine"] = affine2d
                seg_meta_data["original_affine"] = affine2d
                seg_meta_data["spatial_shape"] = np.asarray(
                    seg.shape[-2:], dtype=np.int64
                )

        # ---- Normalize shapes and types to avoid collate errors ----
        def _ensure_channel_first(arr):
            # works for numpy arrays and torch tensors
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                a = arr
                if a.ndim == 2:
                    return a.unsqueeze(0)
                # If last dim looks like channels (H,W,C), move it to front
                if a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[-1] in (1, 3):
                    return a.permute(2, 0, 1).contiguous()
                return a
            else:
                a = np.asarray(arr)
                if a.ndim == 2:
                    return np.expand_dims(a, 0)
                if a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[-1] in (1, 3):
                    return np.transpose(a, (2, 0, 1)).copy()
                return a

        img = _ensure_channel_first(img)
        if seg is not None:
            seg = _ensure_channel_first(seg)

        # ensure label is scalar / python type to avoid tensors with different shapes
        if self.labels is not None:
            label = self.labels[index]
            # convert zero-dim numpy/torch to python scalar
            if isinstance(label, np.ndarray) and label.shape == ():
                label = label.item()
            if isinstance(label, torch.Tensor) and label.dim() == 0:
                label = label.item()
        # ------------------------------------------------------------
        
        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                img, meta_data = apply_transform(
                    self.transform, (img, meta_data), map_items=False, unpack_items=True
                )
            else:
                img = apply_transform(self.transform, img, map_items=False)

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                seg, seg_meta_data = apply_transform(
                    self.seg_transform,
                    (seg, seg_meta_data),
                    map_items=False,
                    unpack_items=True,
                )
            else:
                seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]
            if self.label_transform is not None:
                label = apply_transform(self.label_transform, label, map_items=False)  # type: ignore
        
        
        # construct outputs
        data = [img]
        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if not self.image_only and meta_data is not None:
            data.append(meta_data)
        if not self.image_only and seg_meta_data is not None:
            data.append(seg_meta_data)
        if len(data) == 1:
            return data[0]
        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)
