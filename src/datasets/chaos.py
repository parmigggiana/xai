from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from matplotlib import cm
from monai.data import DataLoader

from src.datasets.common import BaseDataset
from src.ImageDataset import ImageDataset
from src.ITKReader2D import ITKReader2D
from src.utils import simple_collate_fn
from src.volumetricPNGReader import VolumetricPNGReader

chaos_labels_mr = [
    "Background",
    "Liver",
    "Right kidney",
    "Left kidney",
    "Spleen",
]

chaos_labels_ct = [
    "Background",
    "Liver",
]


class PyTorchCHAOS(ImageDataset):
    """
    CHAOS Dataset for CT and MRI volumes using MONAI's ImageDataset.

    This class scans both Train_Sets and Test_Sets directories but only keeps
    samples that have corresponding labels (segmentation files). The actual
    train/val/test splitting is handled by the parent CHAOS class.

    Args:
        base_path (str): Root directory of the dataset.
        domain (str): Either 'CT' or 'MRI'.
        indices (list, optional): List of indices to use for this dataset split.
        transform (callable, optional): Transform to apply to the images.
        seg_transform (callable, optional): Transform to apply to the segmentations.
        slice_2d (bool): If True, loads 2D slices; if False, loads 3D volumes.
        liver_only (bool): If True, filters to liver-only labels for MR.
    """

    def __init__(
        self,
        base_path: str,
        domain: str,
        slice_2d: bool = False,
        indices: Optional[list] = None,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        liver_only: bool = False,
    ) -> None:
        domain = domain.upper()
        assert domain in ["MRI", "MR", "CT"], "Domain must be 'MRI', 'MR', or 'CT'."

        self.base_path = base_path
        self.domain = domain
        self.slice_2d = slice_2d
        self.liver_only = liver_only
        self.indices = indices

        domain_dir = "CT" if self.domain == "CT" else "MR"

        # Scan both Train_Sets and Test_Sets directories
        self.train_data_path = (
            Path(self.base_path) / "CHAOS_Train_Sets" / "Train_Sets" / domain_dir
        )
        self.test_data_path = (
            Path(self.base_path) / "CHAOS_Test_Sets" / "Test_Sets" / domain_dir
        )

        # Load all available samples with labels and prepare file lists for ImageDataset
        image_files, seg_files = self._load_all_file_lists()

        # If indices are provided, filter the files to only include those indices
        if self.indices is not None:
            image_files = [image_files[i] for i in self.indices]
            seg_files = [seg_files[i] for i in self.indices]

        # Initialize ImageDataset with file lists
        super().__init__(
            image_files=image_files,
            seg_files=seg_files,
            transform=transform,
            seg_transform=seg_transform,
            reader=ITKReader2D(),  # Use our fixed reader
            seg_reader=VolumetricPNGReader(),
            image_only=False,
            transform_with_metadata=True,
        )

    def _load_all_file_lists(self):
        """Load file lists from both Train_Sets and Test_Sets directories, keeping only samples with labels."""
        image_files = []
        seg_files = []

        # Scan both train and test directories
        for data_path in [self.train_data_path, self.test_data_path]:
            if not data_path.exists():
                continue

            for patient_id in sorted(data_path.iterdir()):
                if not patient_id.is_dir():
                    continue

                img_path, seg_path = self._get_paths(patient_id, self.domain)

                # Check if both image and segmentation paths exist
                if not img_path.exists() or not seg_path.exists():
                    continue

                if self.slice_2d:
                    self._load_2d_slices(img_path, seg_path, image_files, seg_files)
                else:
                    self._load_3d_volume(img_path, seg_path, image_files, seg_files)

        return image_files, seg_files

    def _load_2d_slices(self, img_path, seg_path, image_files, seg_files):
        """Load 2D slices and only keep those with corresponding labels."""
        img_file_list = sorted(img_path.glob("*.dcm"))
        seg_file_list = sorted(seg_path.glob("*.png"))

        for img_file, seg_file in zip(img_file_list, seg_file_list):
            if (
                img_file.suffix == ".dcm"
                and seg_file.suffix == ".png"
                and seg_file.exists()
            ):
                image_files.append(str(img_file))
                seg_files.append(str(seg_file))

    def _load_3d_volume(self, img_path, seg_path, image_files, seg_files):
        """Load 3D volume only if segmentation directory has PNG files (indicating labels exist)."""
        # Check if segmentation directory has any PNG files
        seg_files_in_dir = list(seg_path.glob("*.png"))
        if len(seg_files_in_dir) > 0:  # Only include if there are segmentation files
            image_files.append(str(img_path))
            seg_files.append(str(seg_path))

    def _get_paths(self, patient_id: Path, domain: str) -> Tuple[Path, Path]:
        if domain == "CT":
            img_path = patient_id / "DICOM_anon"
            seg_path = patient_id / "Ground"
        else:
            img_path = patient_id / "T2SPIR" / "DICOM_anon"
            seg_path = patient_id / "T2SPIR" / "Ground"
        return img_path, seg_path


class CHAOS(BaseDataset):
    def __init__(
        self,
        location,
        domain: str,
        slice_2d: bool = True,
        liver_only: bool = False,
        transform=None,
        seg_transform=None,
        batch_size=1,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    ):
        """
        CHAOS Dataset with proper train/validation/test splitting.

        Args:
            location: Base path to the dataset
            domain: Either 'CT' or 'MRI'
            slice_2d: If True, loads 2D slices; if False, loads 3D volumes
            liver_only: If True, filters to liver-only labels for MR
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
        self.liver_only = liver_only

        # Validate split ratios
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Train, validation, and test ratios must sum to 1.0"

        # Create a full dataset to get all available samples
        full_dataset = PyTorchCHAOS(
            base_path=str(location),
            domain=domain,
            slice_2d=slice_2d,
            indices=None,  # Get all samples
            transform=transform,
            seg_transform=seg_transform,
            liver_only=liver_only,
        )

        # Get total number of samples
        total_samples = len(full_dataset)

        # Calculate split sizes
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size  # Remaining samples

        print(f"Dataset {domain} total samples: {total_samples}")
        print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Create reproducible random indices
        import random

        random.seed(random_seed)
        indices = list(range(total_samples))
        random.shuffle(indices)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Create datasets for each split
        self.train_dataset = PyTorchCHAOS(
            base_path=str(location),
            domain=domain,
            slice_2d=slice_2d,
            indices=train_indices,
            transform=transform,
            seg_transform=seg_transform,
            liver_only=liver_only,
        )

        self.val_dataset = PyTorchCHAOS(
            base_path=str(location),
            domain=domain,
            slice_2d=slice_2d,
            indices=val_indices,
            transform=transform,
            seg_transform=seg_transform,
            liver_only=liver_only,
        )

        self.test_dataset = PyTorchCHAOS(
            base_path=str(location),
            domain=domain,
            slice_2d=slice_2d,
            indices=test_indices,
            transform=transform,
            seg_transform=seg_transform,
            liver_only=liver_only,
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )

        if self.domain == "CT":
            self.classnames = chaos_labels_ct
        else:
            self.classnames = chaos_labels_mr
        self.num_classes = len(self.classnames)

    def visualize_3d(self, sample):
        self._visualize_3d(
            sample,
            rotate=0,
            flip_axis=(
                (3, 2)
                if (self.domain == "CT" and not self.train_dataset.slice_2d)
                else 2
            ),
        )

    def visualize_sample_slice(self, sample, rotate=0, flip_axis=1):
        return self._visualize_sample_slice(sample, rotate, flip_axis=flip_axis)

    def _get_organ_legend(self, seg_slice):
        legend = {}
        set1 = cm.get_cmap("Set1", 8)  # Set1 is qualitative, 8 distinct colors

        if self.domain in ["MR", "MRI"]:
            organ_list = [
                ("Liver", (55, 70, 63), 0),
                ("Right kidney", (110, 135, 126), 1),
                ("Left kidney", (175, 200, 189), 2),
                ("Spleen", (240, 255, 252), 3),
            ]
            for organ_name, (min_val, max_val, _), color_idx in organ_list:
                if np.any((seg_slice >= min_val) & (seg_slice <= max_val)):
                    legend[organ_name] = set1(color_idx)
        elif self.domain == "CT":
            legend["Liver"] = set1(0)

        return legend

    def encode(self, labels):
        """Encode labels to match the dataset encoding."""
        if self.domain in ["MR", "MRI"]:
            return labels * 63
        elif self.domain == "CT":
            return torch.where(labels > 0, 1, 0)

    def decode(self, labels):
        """Decode labels to match the original dataset encoding."""
        if self.domain in ["MR", "MRI"]:
            return labels // 63
        elif self.domain == "CT":
            return torch.where(labels > 0, 1, 0)
