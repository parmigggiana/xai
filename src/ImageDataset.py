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
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from monai.config import DtypeLike
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.data import MetaTensor, Dataset as MonaiDataset
from monai.utils import MAX_SEED, get_seed


class ImageDataset(MonaiDataset, Randomizable):
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
        dtype: DtypeLike = np.float32,
        reader: ImageReader | str | None = None,
        seg_reader: ImageReader | str | None = None,
        cache_max_items: int | None = None,
        enable_cache: bool | None = None,
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
            Note: this dataset always returns images with metadata (as MetaTensor) and propagates
            metadata through transforms when possible.
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
        # Store inputs and transforms
        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.label_transform = label_transform

        # Always load with metadata
        self.loader = LoadImage(reader, False, dtype, *args, **kwargs)
        self.seg_loader = LoadImage(seg_reader or reader, False, dtype, *args, **kwargs)

        # Random state for sync between image/seg transforms
        self.set_random_state(seed=get_seed())
        self._seed = 0

        # Simple in-memory LRU caches for loaded images and segmentations
        self._enable_cache = True if (enable_cache is None) else bool(enable_cache)
        # default to 96 if not provided
        self._cache_max_items = (
            int(cache_max_items) if (cache_max_items is not None) else 96
        )
        self._img_cache = OrderedDict()
        self._seg_cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.image_files)

    # Cache management helpers
    def clear_cache(self) -> None:
        """Clear in-memory image/segmentation caches."""
        if hasattr(self, "_img_cache"):
            self._img_cache.clear()
        if hasattr(self, "_seg_cache"):
            self._seg_cache.clear()
        # reset counters when clearing cache
        self.reset_cache_stats()

    def enable_cache(self, enable: bool = True) -> None:
        """Enable or disable caching for subsequent loads."""
        self._enable_cache = bool(enable)
        if not self._enable_cache:
            self.clear_cache()

    def set_cache_max_items(self, n: int) -> None:
        """Set the maximum number of items kept in each cache (image and seg)."""
        if n <= 0:
            self._cache_max_items = 0
            self.clear_cache()
            self._enable_cache = False
        else:
            self._cache_max_items = int(n)

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

        # load data and meta with caching
        def _cache_get(cache: OrderedDict, key: str):
            if key in cache:
                val = cache.pop(key)
                cache[key] = val  # move to end (LRU)
                return val
            return None

        def _cache_put(cache: OrderedDict, key: str, val: tuple):
            cache[key] = val
            # evict oldest if needed
            if len(cache) > self._cache_max_items:
                ev_key, _ = cache.popitem(last=False)

        img_key = str(image_file)
        cached = _cache_get(self._img_cache, img_key) if self._enable_cache else None
        if cached is not None:
            img, meta_data = cached
            # avoid mutating cached meta dict downstream
            if isinstance(meta_data, dict):
                meta_data = dict(meta_data)
        else:
            img, meta_data = self.loader(image_file)
            if self._enable_cache:
                _cache_put(self._img_cache, img_key, (img, meta_data))

        if seg_file is not None:
            seg_key = str(seg_file)
            cached_seg = (
                _cache_get(self._seg_cache, seg_key) if self._enable_cache else None
            )
            if cached_seg is not None:
                seg, seg_meta_data = cached_seg
                if isinstance(seg_meta_data, dict):
                    seg_meta_data = dict(seg_meta_data)
            else:
                seg, seg_meta_data = self.seg_loader(seg_file)
                if self._enable_cache:
                    _cache_put(self._seg_cache, seg_key, (seg, seg_meta_data))

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

        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            # Always apply transforms with metadata propagation via MetaTensor
            wrapped_img = (
                MetaTensor(img, meta=meta_data) if isinstance(meta_data, dict) else img
            )
            out = apply_transform(self.transform, wrapped_img, map_items=False)
            if isinstance(out, MetaTensor):
                img = out
                meta_data = dict(out.meta) if out.meta is not None else meta_data
            else:
                img = (
                    MetaTensor(out, meta=meta_data)
                    if isinstance(meta_data, dict)
                    else out
                )

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            wrapped_seg = (
                MetaTensor(seg, meta=seg_meta_data)
                if isinstance(seg_meta_data, dict)
                else seg
            )
            seg_out = apply_transform(self.seg_transform, wrapped_seg, map_items=False)
            if isinstance(seg_out, MetaTensor):
                seg = seg_out
                seg_meta_data = (
                    dict(seg_out.meta) if seg_out.meta is not None else seg_meta_data
                )
            else:
                seg = (
                    MetaTensor(seg_out, meta=seg_meta_data)
                    if isinstance(seg_meta_data, dict)
                    else seg_out
                )

        if self.labels is not None:
            label = self.labels[index]
            if self.label_transform is not None:
                label = apply_transform(self.label_transform, label, map_items=False)  # type: ignore

        # construct MONAI-native dict output
        sample = {"image": img}
        if seg is not None:
            sample["label"] = seg
        if label is not None:
            sample["class"] = label
        # metadata stays inside MetaTensor; no separate meta dicts returned
        return sample
