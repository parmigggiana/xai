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
import os
import copy
from typing import Any
from pathlib import Path
import time
import json
import datetime as _dt

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

        # Detect slice mode (files provided as (path, slice_idx)) and initialize small LRU caches
        self._slice_mode = any(isinstance(x, tuple) and len(x) == 2 for x in self.image_files)
        # Allow tuning via env var; default to a conservative cache size to limit RAM
        self._cache_capacity = int(os.environ.get("XAI_VOLUME_CACHE_CAPACITY", "3"))
        self._img_volume_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._seg_volume_cache: "OrderedDict[str, tuple]" = OrderedDict()

    # -------------------- Profiling helpers (opt-in via env vars) --------------------
    @staticmethod
    def _profiling_enabled() -> bool:
        try:
            val = os.environ.get("XAI_PROFILE_PREPROCESS", "")
            return str(val).lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            return False

    @staticmethod
    def _profiling_dir() -> Path:
        try:
            d = os.environ.get("XAI_PROFILE_DIR", "./outputs/profiling")
        except Exception:
            d = "./outputs/profiling"
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _profiling_log(event: dict) -> None:
        """Append a JSONL event to a per-process file. Safe for multi-worker DataLoader.

        Control via env vars:
          - XAI_PROFILE_PREPROCESS=1 to enable
          - XAI_PROFILE_DIR to choose output folder
        """
        if not ImageDataset._profiling_enabled():
            return
        try:
            out = ImageDataset._profiling_dir() / f"preprocess-{os.getpid()}.jsonl"
            # enrich with timestamp and pid
            event = dict(event)  # shallow copy
            event.setdefault("ts", _dt.datetime.now().isoformat(timespec="seconds"))
            event.setdefault("pid", os.getpid())
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            # Swallow any logging errors to avoid breaking data loading
            pass

    def _get_cached_volume(self, kind: str, file_path: str):
        """Return (array, metadata) for file_path using an LRU cache when in slice mode.
        kind: 'img' or 'seg'
        """
        if not self._slice_mode:
            # No caching necessary
            if kind == "img":
                return self.loader(file_path) if not self.image_only else (self.loader(file_path), None)
            else:
                return self.seg_loader(file_path) if not self.image_only else (self.seg_loader(file_path), None)

        cache = self._img_volume_cache if kind == "img" else self._seg_volume_cache
        loader = self.loader if kind == "img" else self.seg_loader

        # Normalize key to string path
        key = str(file_path)
        if key in cache:
            val = cache.pop(key)
            cache[key] = val  # move to end (most-recent)
            # Defensive copy of metadata dict to avoid in-place mutations downstream
            arr, meta = val
            meta_copy = copy.deepcopy(meta) if isinstance(meta, dict) else meta
            return arr, meta_copy

        # Miss: load once and cache
        if self.image_only:
            arr = loader(file_path)
            meta = None
        else:
            arr, meta = loader(file_path)

        cache[key] = (arr, meta)
        # Evict least-recently-used if over capacity
        if self._cache_capacity > 0 and len(cache) > self._cache_capacity:
            cache.popitem(last=False)

        meta_copy = copy.deepcopy(meta) if isinstance(meta, dict) else meta
        return arr, meta_copy

    def __len__(self) -> int:
        return len(self.image_files)

    def randomize(self, data: Any | None = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        meta_data, seg_meta_data, seg, label = None, None, None, None

        # timing accumulators
        _t_load_img = _t_load_seg = _t_xform_img = _t_xform_seg = None

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

        # load data and optionally meta (with per-volume caching for slice mode)
        if image_slice_idx is not None:
            # Use cached volume loading for images (and seg if present)
            _t0 = time.perf_counter()
            img, meta_data = self._get_cached_volume("img", image_file)
            _t_load_img = time.perf_counter() - _t0
            if seg_file is not None:
                _t1 = time.perf_counter()
                seg, seg_meta_data = self._get_cached_volume("seg", seg_file)
                _t_load_seg = time.perf_counter() - _t1
                # Copy relevant spatial metadata from image to segmentation (non-destructive)
                if isinstance(meta_data, dict) and isinstance(seg_meta_data, dict):
                    for attribute in meta_data:
                        seg_meta_data.setdefault(attribute, meta_data[attribute])
        else:
            # Fallback: default loader path (no caching)
            if self.image_only:
                _t0 = time.perf_counter()
                img = self.loader(image_file)
                _t_load_img = time.perf_counter() - _t0
                if seg_file is not None:
                    _t1 = time.perf_counter()
                    seg = self.seg_loader(seg_file)
                    _t_load_seg = time.perf_counter() - _t1
            else:
                _t0 = time.perf_counter()
                img, meta_data = self.loader(image_file)
                _t_load_img = time.perf_counter() - _t0
                if seg_file is not None:
                    _t1 = time.perf_counter()
                    seg, seg_meta_data = self.seg_loader(seg_file)
                    _t_load_seg = time.perf_counter() - _t1
                    # Copy relevant spatial metadata from image to segmentation
                    if meta_data and seg_meta_data:
                        for attribute in meta_data:
                            if attribute not in seg_meta_data:
                                seg_meta_data[attribute] = meta_data[attribute]
        
        #print(f"[DEBUG] index: {index}, image_file: {image_file}, img mean: {np.mean(img):.4f}, img shape: {img.shape}")                    

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

            _t0 = time.perf_counter()
            if self.transform_with_metadata:
                img, meta_data = apply_transform(
                    self.transform, (img, meta_data), map_items=False, unpack_items=True
                )
            else:
                img = apply_transform(self.transform, img, map_items=False)
            _t_xform_img = time.perf_counter() - _t0

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            _t0 = time.perf_counter()
            if self.transform_with_metadata:
                seg, seg_meta_data = apply_transform(
                    self.seg_transform,
                    (seg, seg_meta_data),
                    map_items=False,
                    unpack_items=True,
                )
            else:
                seg = apply_transform(self.seg_transform, seg, map_items=False)
            _t_xform_seg = time.perf_counter() - _t0

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
        # Optional: emit a profiling event for this item
        try:
            if ImageDataset._profiling_enabled():
                def _shape_dtype(x):
                    try:
                        s = tuple(x.shape) if hasattr(x, "shape") else None
                        dt = str(getattr(x, "dtype", None))
                        return s, dt
                    except Exception:
                        return None, None

                img_shape, img_dtype = _shape_dtype(data[0]) if len(data) >= 1 else (None, None)
                seg_shape, seg_dtype = _shape_dtype(data[1]) if len(data) >= 2 else (None, None)
                ImageDataset._profiling_log(
                    {
                        "event": "preprocess",
                        "index": int(index),
                        "image_file": str(image_file),
                        "seg_file": str(seg_file) if seg_file is not None else None,
                        "slice_idx": int(image_slice_idx) if image_slice_idx is not None else None,
                        "seg_slice_idx": int(seg_slice_idx) if seg_slice_idx is not None else None,
                        "t_load_img_s": float(_t_load_img) if _t_load_img is not None else None,
                        "t_load_seg_s": float(_t_load_seg) if _t_load_seg is not None else None,
                        "t_xform_img_s": float(_t_xform_img) if _t_xform_img is not None else None,
                        "t_xform_seg_s": float(_t_xform_seg) if _t_xform_seg is not None else None,
                        "img_shape": img_shape,
                        "img_dtype": img_dtype,
                        "seg_shape": seg_shape,
                        "seg_dtype": seg_dtype,
                    }
                )
        except Exception:
            pass

        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)
       
