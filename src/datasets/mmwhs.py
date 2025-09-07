from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from matplotlib import cm
from monai.data import DataLoader, NibabelReader

from src.datasets.common import BaseDataset
from src.PersistentDataset import ImageLabelPersistentDataset
from src.utils import meta_safe_collate

mmwhs_labels = {
    0: "Background",
    500: "Left ventricle blood cavity",
    600: "Right ventricle blood cavity",
    420: "Left atrium blood cavity",
    550: "Right atrium blood cavity",
    205: "Myocardium of the left ventricle",
    820: "Ascending aorta",
    850: "Pulmonary artery",
}


class PyTorchMMWHS(ImageLabelPersistentDataset):
    """
    MM-WHS Dataset for CT and MRI volumes using MONAI's ImageDataset.

    This class scans both train and test directories but only keeps
    samples that have corresponding unencrypted labels (training files).
    The actual train/val/test splitting is handled by the parent MMWHS class.

    Args:
        base_path (str): Root directory of the dataset.
        domain (str): Either 'CT' or 'MR'.
        indices (list, optional): List of indices to use for this dataset split.
        transform (callable, optional): Transform to apply to the images.
        seg_transform (callable, optional): Transform to apply to the segmentations.
        slice_2d (bool): If True, slices the 3D volumes into 2D slices. Defaults to False (3D).
    """

    def __init__(
        self,
        base_path: str,
        domain: str,
        indices: Optional[list] = None,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        slice_2d: bool = False,
        **image_dataset_kwargs,
    ) -> None:
        domain = domain.lower()
        assert domain in ["ct", "mr"], "Domain must be 'CT' or 'MR'."

        self.base_path = base_path
        self.domain = domain
        self.slice_2d = slice_2d
        self.indices = indices

        # Define paths for train and test directories
        self.train_data_path = (
            Path(self.base_path)
            / "MM-WHS 2017 Dataset"
            / "MM-WHS 2017 Dataset"
            / f"{self.domain}_train"
        )

        self.test_data_path = (
            Path(self.base_path)
            / "MMWHS"
            / "MM-WHS 2017 Dataset"
            / "MM-WHS 2017 Dataset"
            / f"{self.domain}_test"
        )

        # Load all available samples with unencrypted labels
        image_files, seg_files = self._load_all_file_lists()

        # If indices are provided, filter the files to only include those indices
        if self.indices is not None:
            image_files = [image_files[i] for i in self.indices]
            seg_files = [seg_files[i] for i in self.indices]

        # Initialize PersistentDataset-backed dataset with file lists
        super().__init__(
            image_files=image_files,
            seg_files=seg_files,
            transform=transform,
            seg_transform=seg_transform,
            reader=NibabelReader(),
            seg_reader=NibabelReader(),
            **image_dataset_kwargs,
        )

    def _load_all_file_lists(self):
        """Load file lists from train directory only, since test labels are encrypted."""
        image_files = []
        seg_files = []

        # Only scan the train directory as it has unencrypted labels
        if self.train_data_path.exists():
            for img_file in sorted(self.train_data_path.glob("*_image.nii.gz")):
                if not str(img_file).endswith(".nii.gz"):
                    continue

                # Look for corresponding label file
                label_file = img_file.parent / img_file.name.replace(
                    "_image.nii.gz", "_label.nii.gz"
                )

                if not label_file.exists():
                    continue  # Skip if no label file

                if self.slice_2d:
                    slices = self._read_2d_slices(img_file, label_file)
                    for slice_idx in range(slices):
                        image_files.append((img_file, slice_idx))
                        seg_files.append((label_file, slice_idx))
                else:
                    image_files.append(img_file)
                    seg_files.append(label_file)

        return image_files, seg_files

    def _read_2d_slices(self, img_file, label_file):
        """Load 2D slices from a 3D volume by extracting individual slices."""
        try:
            reader = NibabelReader()
            data = reader.read(img_file)

            slices = data.shape[2]

            return slices

        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            import traceback

            traceback.print_exc()


class MMWHS(BaseDataset):
    def __init__(
        self,
        location,
        domain: str,
        slice_2d: bool = False,
        transform=None,
        seg_transform=None,
        batch_size=1,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        # tiny in-memory LRU cache for repeated volume reads when slicing
        volume_cache_size: int = 16,
    ):
        """
        MMWHS Dataset with proper train/validation/test splitting.

        Args:
            location: Base path to the dataset
            domain: Either 'CT' or 'MR'
            slice_2d: If True, loads 2D slices; if False, loads 3D volumes
            transform: Transform to apply to images
            seg_transform: Transform to apply to segmentations
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            random_seed: Random seed for reproducible splits
        """
        super().__init__()
        self.domain = domain
        self.slice_2d = slice_2d

        # Validate split ratios
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Train, validation, and test ratios must sum to 1.0"

        # Create a full dataset to discover samples
        full_dataset = PyTorchMMWHS(
            base_path=str(location),
            domain=domain,
            indices=None,
            transform=transform,
            seg_transform=seg_transform,
            slice_2d=slice_2d,
            volume_cache_size=volume_cache_size,
        )

        # Get total number of samples
        total_samples = len(full_dataset)

        # Calculate split sizes
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)

        # Create reproducible random indices
        import random

        random.seed(random_seed)
        indices = list(range(total_samples))
        random.shuffle(indices)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Build split datasets using PyTorchMMWHS (ImageDataset emits dict samples)
        self.train_dataset = PyTorchMMWHS(
            base_path=str(location),
            domain=domain,
            indices=train_indices,
            transform=transform,
            seg_transform=seg_transform,
            slice_2d=slice_2d,
            volume_cache_size=volume_cache_size,
        )

        self.val_dataset = PyTorchMMWHS(
            base_path=str(location),
            domain=domain,
            indices=val_indices,
            transform=transform,
            seg_transform=seg_transform,
            slice_2d=slice_2d,
            volume_cache_size=volume_cache_size,
        )

        self.test_dataset = PyTorchMMWHS(
            base_path=str(location),
            domain=domain,
            indices=test_indices,
            transform=transform,
            seg_transform=seg_transform,
            slice_2d=slice_2d,
            volume_cache_size=volume_cache_size,
        )

        # Create DataLoaders
        train_loader_kwargs = {
            "shuffle": True,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True if torch.cuda.is_available() else False,
            "drop_last": True,
            "collate_fn": meta_safe_collate,
        }
        if num_workers and num_workers > 0:
            train_loader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                }
            )
        self.train_loader = DataLoader(self.train_dataset, **train_loader_kwargs)

        val_loader_kwargs = {
            "shuffle": False,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate_fn": meta_safe_collate,
        }
        if num_workers and num_workers > 0:
            val_loader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                }
            )
        self.val_loader = DataLoader(self.val_dataset, **val_loader_kwargs)

        test_loader_kwargs = {
            "shuffle": False,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate_fn": meta_safe_collate,
        }
        if num_workers and num_workers > 0:
            test_loader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                }
            )
        self.test_loader = DataLoader(self.test_dataset, **test_loader_kwargs)

        self.classnames = list(mmwhs_labels.values())
        self.num_classes = len(mmwhs_labels)

    def visualize_3d(self, sample):
        self._visualize_3d(
            sample,
            rotate=1 if self.domain in ["MR", "MRI"] else 0,
            flip_axis=0 if self.domain in ["MR", "MRI"] else None,
        )

    def visualize_sample_slice(self, sample):
        self._visualize_sample_slice(
            sample, rotate=1 if self.domain in ["MR", "MRI"] else 0, flip_axis=None
        )

    def _get_organ_legend(self, seg_slice):
        legend = {}

        set1 = cm.get_cmap("Set1", 8)
        # Skip background (label 0) in legend
        color_idx = 0
        for label_val, organ_name in mmwhs_labels.items():
            if label_val == 0:
                continue
            if np.any(seg_slice == label_val):
                legend[organ_name] = set1(color_idx)
                color_idx += 1

        return legend

    def encode(self, labels):
        """
        Encode the labels to their corresponding values.
        """
        encoded_labels = torch.zeros_like(labels, dtype=torch.int64)
        for i, label_val in enumerate(mmwhs_labels.keys()):
            encoded_labels[labels == i] = label_val
        return encoded_labels

    def decode(self, labels):
        """
        Decode the labels to their original values.
        """
        decoded_labels = torch.zeros_like(labels, dtype=torch.int64)
        for i, label_val in enumerate(mmwhs_labels.keys()):
            decoded_labels[labels == label_val] = i
        return decoded_labels
