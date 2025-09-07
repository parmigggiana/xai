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

import collections.abc
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import (
    SUPPORTED_PICKLE_MOD,
    pickle_hashing,
)
from monai.transforms import (
    Compose,
    RandomizableTrait,
    Transform,
    reset_ops_id,
    MapTransform,
    LoadImage,
)
from monai.utils import (
    look_up_option,
    optional_import,
)

cp, _ = optional_import("cupy")
lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")
kvikio_numpy, _ = optional_import("kvikio.numpy")


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(
        self, data: Sequence, transform: Sequence[Callable] | Callable | None = None
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable, sequence of callables or None. If transform is not
            a `Compose` instance, it will be wrapped in a `Compose` instance. Sequences
            of callables are applied in order and if `None` is passed, the data is returned as is.
        """
        self.data = data
        try:
            self.transform = (
                Compose(transform) if not isinstance(transform, Compose) else transform
            )
        except Exception as e:
            raise ValueError(
                "`transform` must be a callable or a list of callables that is Composable"
            ) from e

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return self.transform(data_i)

    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class PersistentDataset(Dataset):
    """
    Persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data,
    it can operate transforms for specific fields.  Results from the non-random transform components are computed
    when first used, and stored in the `cache_dir` for rapid retrieval on subsequent uses.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
            'image': 'image1.nii.gz',    'image': 'image2.nii.gz',    'image': 'image3.nii.gz',
            'label': 'label1.nii.gz',    'label': 'label2.nii.gz',    'label': 'label3.nii.gz',
            'extra': 123                 'extra': 456                 'extra': 789
        },                           },                           }]

    For a composite transform like

    .. code-block:: python

        [ LoadImaged(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96),
                                pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
        ToTensord(keys=['image', 'label'])]

    Upon first use a filename based dataset will be processed by the transform for the
    [LoadImaged, Orientationd, ScaleIntensityRanged] and the resulting tensor written to
    the `cache_dir` before applying the remaining random dependant transforms
    [RandCropByPosNegLabeld, ToTensord] elements for use in the analysis.

    Subsequent uses of a dataset directly read pre-processed results from `cache_dir`
    followed by applying the random dependant parts of transform processing.

    During training call `set_data()` to update input data and recompute cache content.

    Note:
        The input data must be a list of file paths and will hash them as cache keys.

        The filenames of the cached files also try to contain the hash of the transforms. In this
        fashion, `PersistentDataset` should be robust to changes in transforms. This, however, is
        not guaranteed, so caution should be used when modifying transforms to avoid unexpected
        errors. If in doubt, it is advisable to clear the cache directory.

    Lazy Resampling:
        If you make use of the lazy resampling feature of `monai.transforms.Compose`, please refer to
        its documentation to familiarize yourself with the interaction between `PersistentDataset` and
        lazy resampling.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Sequence[Callable] | Callable,
        cache_dir: Path | str | None,
        hash_func: Callable[..., bytes] = pickle_hashing,
        pickle_module: str = "pickle",
        pickle_protocol: int = DEFAULT_PROTOCOL,
        hash_transform: Callable[..., bytes] | None = None,
        reset_ops_id: bool = True,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset for model.
                `PersistentDataset` expects input data to be a list of serializable
                and hashes them as cache keys using `hash_func`.
            transform: transforms to execute operations on input data.
            cache_dir: If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If `cache_dir` doesn't exist, will automatically create it.
                If `cache_dir` is `None`, there is effectively no caching.
            hash_func: a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            pickle_module: string representing the module used for pickling metadata and objects,
                default to `"pickle"`. due to the pickle limitation in multi-processing of Dataloader,
                we can't use `pickle` as arg directly, so here we use a string name instead.
                if want to use other pickle module at runtime, just register like:
                >>> from monai.data import utils
                >>> utils.SUPPORTED_PICKLE_MOD["test"] = other_pickle
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save,
                and ``monai.data.utils.SUPPORTED_PICKLE_MOD``.
            pickle_protocol: can be specified to override the default protocol, default to `2`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            hash_transform: a callable to compute hash from the transform information when caching.
                This may reduce errors due to transforms changing during experiments. Default to None (no hash).
                Other options are `pickle_hashing` and `json_hashing` functions from `monai.data.utils`.
            reset_ops_id: whether to set `TraceKeys.ID` to ``Tracekys.NONE``, defaults to ``True``.
                When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors.
                This is useful for skipping the transform instance checks when inverting applied operations
                using the cached content and with re-created transform instances.

        """
        super().__init__(data=data, transform=transform)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hash_func = hash_func
        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol
        if self.cache_dir is not None:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            if not self.cache_dir.is_dir():
                raise ValueError("cache_dir must be a directory.")
        self.transform_hash: str = ""
        if hash_transform is not None:
            self.set_transform_hash(hash_transform)
        self.reset_ops_id = reset_ops_id

    def set_transform_hash(self, hash_xform_func: Callable[..., bytes]):
        """Get hashable transforms, and then hash them. Hashable transforms
        are deterministic transforms that inherit from `Transform`. We stop
        at the first non-deterministic transform, or first that does not
        inherit from MONAI's `Transform` class."""
        hashable_transforms = []
        for _tr in self.transform.flatten().transforms:
            if isinstance(_tr, RandomizableTrait) or not isinstance(_tr, Transform):
                break
            hashable_transforms.append(_tr)
        # Try to hash. Fall back to a hash of their names
        try:
            transform_hash = hash_xform_func(hashable_transforms)
        except TypeError as te:
            if "is not JSON serializable" not in str(te):
                raise te
            names = "".join(tr.__class__.__name__ for tr in hashable_transforms)
            transform_hash = hash_xform_func(names)
        self.transform_hash = transform_hash.decode("utf-8")

    def set_data(self, data: Sequence):
        """
        Set the input data and delete all the out-dated cache content.

        """
        self.data = data
        if self.cache_dir is not None and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _pre_transform(self, item_transformed):
        """
        Process the data from original state up to the first random element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the first identified
            random transform object

        """
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        item_transformed = self.transform(
            item_transformed, end=first_random, threading=True
        )

        if self.reset_ops_id:
            reset_ops_id(item_transformed)
        return item_transformed

    def _post_transform(self, item_transformed):
        """
        Process the data from before the first random transform to the final state ready for evaluation.

        Args:
            item_transformed: The data to be transformed (already processed up to the first random transform)

        Returns:
            the transformed element through the random transforms

        """
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        if first_random is not None:
            item_transformed = self.transform(item_transformed, start=first_random)
        return item_transformed

    def _cachecheck(self, item_transformed):
        """
        A function to cache the expensive input data transform operations
        so that huge data sets (larger than computer memory) can be processed
        on the fly as needed, and intermediate results written to disk for
        future use.

        Args:
            item_transformed: The current data element to be mutated into transformed representation

        Returns:
            The transformed data_element, either from cache, or explicitly computing it.

        Warning:
            The current implementation does not encode transform information as part of the
            hashing mechanism used for generating cache names when `hash_transform` is None.
            If the transforms applied are changed in any way, the objects in the cache dir will be invalid.

        """
        hashfile = None
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            try:
                return torch.load(hashfile, weights_only=False)
            except PermissionError as e:
                if sys.platform != "win32":
                    raise e
            except RuntimeError as e:
                if "Invalid magic number; corrupt file" in str(e):
                    warnings.warn(
                        f"Corrupt cache file detected: {hashfile}. Deleting and recomputing."
                    )
                    hashfile.unlink()
                else:
                    raise e

        _item_transformed = self._pre_transform(
            deepcopy(item_transformed)
        )  # keep the original hashed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_item_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(
                        self.pickle_module, SUPPORTED_PICKLE_MOD
                    ),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


# -----------------------------------------------------------------------------
# Extensions: Support dual image/label pipelines with per-key readers and
# slice tuple handling while leveraging the persistent, disk-backed cache.
# -----------------------------------------------------------------------------


def _build_2d_affine_from_3d(affine3d: np.ndarray, k: int) -> np.ndarray:
    """Construct 2D affine from 3D affine for slice k (slice along last axis)."""
    I2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
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


class LoadPathOrSliceD(MapTransform):
    """Load per-key data from path or (path, slice_idx) tuples.

    - Uses provided reader per key via LoadImage.
    - If tuple is given, extracts a 2D slice along the last dim and updates
      MetaTensor metadata (affine, original_affine, spatial_shape).
    """

    def __init__(
        self,
        keys: Sequence[str],
        readers: dict[str, Any],
        dtype: Any = np.float32,
        loader_kwargs: dict[str, dict] | None = None,
    ) -> None:
        super().__init__(keys)
        loader_kwargs = loader_kwargs or {}
        self._loaders: dict[str, LoadImage] = {}
        for k in keys:
            kwargs = loader_kwargs.get(k, {})
            self._loaders[k] = LoadImage(readers.get(k), False, dtype, **kwargs)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            val = d.get(key)
            if val is None:
                continue
            slice_idx = None
            path = val
            if isinstance(val, tuple) and len(val) == 2:
                path, slice_idx = val
            img, meta = self._loaders[key](path)
            if slice_idx is not None and hasattr(img, "shape") and len(img.shape) >= 3:
                aff = None
                if isinstance(meta, dict):
                    # avoid boolean evaluation on tensors; check None explicitly
                    aff = meta.get("affine", None)
                    if aff is None:
                        aff = meta.get("original_affine", None)
                    if isinstance(aff, torch.Tensor):
                        aff = aff.detach().cpu().numpy()
                img = img[..., int(slice_idx)]
                if isinstance(meta, dict):
                    affine2d = _build_2d_affine_from_3d(aff, int(slice_idx))
                    meta["affine"] = affine2d
                    meta["original_affine"] = affine2d
                    meta["spatial_shape"] = np.asarray(img.shape[-2:], dtype=np.int64)
            d[key] = MetaTensor(img, meta=meta) if isinstance(meta, dict) else img
        return d


class ApplyToKey(Transform):
    """Apply a given transform to a single key within a dict sample."""

    def __init__(self, key: str, xform: Callable | None) -> None:
        self.key = key
        self.xform = xform

    def __call__(self, data: dict) -> dict:
        if self.xform is None or self.key not in data:
            return data
        data = dict(data)
        data[self.key] = self.xform(data[self.key])
        return data


class ImageLabelPersistentDataset(PersistentDataset):
    """Convenience dataset for image/label pairs with separate pipelines.

    This wraps the custom PersistentDataset with:
    - per-key readers (image vs label)
    - support for separate image and label transforms
    - support for (path, slice_idx) tuples for 2D extraction
    Returns dict samples: {"image": ..., "label": ..., "class": ...?}
    """

    def __init__(
        self,
        image_files: Sequence[str],
        seg_files: Sequence[str] | None = None,
        labels: Sequence[Any] | None = None,
        transform: Callable | None = None,
        seg_transform: Callable | None = None,
        reader: Any | None = None,
        seg_reader: Any | None = None,
        cache_dir: Path | str | None = None,
        dtype: Any = np.float32,
        loader_kwargs: dict[str, dict] | None = None,
    ) -> None:
        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        # Build dict-style items list
        items: list[dict[str, Any]] = []
        for i, imgf in enumerate(image_files):
            sample: dict[str, Any] = {"image": imgf}
            if seg_files is not None:
                sample["label"] = seg_files[i]
            if labels is not None:
                sample["class"] = labels[i]
            items.append(sample)

        # Compose: per-key loading, then per-key transforms
        present_keys = [k for k in ("image", "label") if any(k in s for s in items)]
        readers = {"image": reader, "label": seg_reader or reader}
        composed = Compose(
            [
                LoadPathOrSliceD(
                    keys=present_keys,
                    readers=readers,
                    dtype=dtype,
                    loader_kwargs=loader_kwargs,
                ),
                ApplyToKey("image", transform),
                ApplyToKey("label", seg_transform),
            ]
        )

        # Default cache dir if not provided
        if cache_dir is None:
            cache_dir = Path("outputs") / "persistent_cache"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        super().__init__(data=items, transform=composed, cache_dir=cache_dir)

    # Backward-compat convenience
    def clear_cache(self):
        if self.cache_dir and Path(self.cache_dir).exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
