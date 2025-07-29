from email.mime import image
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from matplotlib import cm
from monai.data import DataLoader, ITKReader, MetaTensor, PILReader

from src.datasets.common import BaseDataset
from src.datasets.custom_imageDataset import ImageDataset
from src.datasets.volumetricPNGReader import VolumetricPNGReader

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

    Args:
        base_path (str): Root directory of the dataset.
        domain (str): Either 'CT' or 'MRI'.
        split (str): Either 'train' or 'test'.
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
        split: str = "train",
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        liver_only: bool = False,
    ) -> None:
        domain = domain.upper()
        split = split.lower()
        assert domain in ["MRI", "MR", "CT"], "Domain must be 'MRI', 'MR', or 'CT'."
        assert split in ["train", "test"], "Split must be 'train' or 'test'."

        self.base_path = base_path
        self.domain = domain
        self.split = split
        self.slice_2d = slice_2d
        self.liver_only = liver_only

        split_dir = "Test_Sets" if self.split == "test" else "Train_Sets"
        domain_dir = "CT" if self.domain == "CT" else "MR"

        self.data_path = (
            Path(self.base_path) / f"CHAOS_{split_dir}" / split_dir / domain_dir
        )

        # Load samples and prepare file lists for ImageDataset
        image_files, seg_files = self._load_file_lists()

        print(f"Loaded {image_files} and {seg_files} for {self.domain} {self.split}")

        # Initialize ImageDataset with file lists
        super().__init__(
            image_files=image_files,
            seg_files=seg_files if self.split == "train" else None,
            transform=transform,
            seg_transform=seg_transform,
            reader=ITKReader(),
            seg_reader=VolumetricPNGReader(),
            image_only=self.split != "train",  # Only load images for test set
        )

    def _load_file_lists(self):
        """Load file lists for ImageDataset initialization."""
        image_files = []
        seg_files = []

        for patient_id in sorted(self.data_path.iterdir()):
            if not patient_id.is_dir():
                continue

            img_path, seg_path = self._get_paths(patient_id, self.domain)
            if self.slice_2d:
                img_file_list = sorted(img_path.glob("*.dcm"))
                seg_file_list = sorted(seg_path.glob("*.png"))

                for img_file, seg_file in zip(img_file_list, seg_file_list):
                    if img_file.suffix == ".dcm" and seg_file.suffix == ".png":
                        image_files.append(str(img_file))
                        if self.split == "train":
                            seg_files.append(str(seg_file))
            else:
                image_files.append(str(img_path))
                if self.split == "train":
                    seg_files.append(str(seg_path))

        return image_files, seg_files

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
        slice_2d: bool = False,
        liver_only: bool = False,
        transform=None,
        seg_transform=None,
        batch_size=1,
        num_workers=0,
    ):
        """
        CHAOS Test does not have labels, so we only use it for inference.
        """
        super().__init__()
        self.domain = domain
        self.slice_2d = slice_2d
        self.liver_only = liver_only

        self.train_dataset = PyTorchCHAOS(
            base_path=str(location),
            domain=domain,
            slice_2d=slice_2d,
            split="train",
            transform=transform,
            seg_transform=seg_transform,
            liver_only=liver_only,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.test_dataset = PyTorchCHAOS(
            base_path=str(location),
            domain=domain,
            slice_2d=slice_2d,
            split="test",
            transform=transform,
            seg_transform=seg_transform,
            liver_only=liver_only,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        if self.domain == "CT":
            self.num_classes = len(chaos_labels_ct)
        else:
            self.num_classes = len(chaos_labels_mr)

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

    def visualize_sample_slice(self, sample):
        return self._visualize_sample_slice(sample, 0, flip_axis=1)

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
