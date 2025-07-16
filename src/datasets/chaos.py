from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pydicom
import torch
from matplotlib import cm
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from src.datasets.common import BaseDataset

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


class PyTorchCHAOS(VisionDataset):
    """
    CHAOS Dataset for CT and MRI volumes.

    Args:
        base_path (str): Root directory of the dataset.
        domain (str): Either 'CT' or 'MRI'.
        split (str): Either 'train' or 'test'.
        transform (callable, optional): Transform to apply to the samples.
    """

    def __init__(
        self,
        base_path: str,
        domain: str,
        slice_2d: bool = False,
        split: str = "train",
        transform: Optional[Callable] = None,
        liver_only: bool = False,
    ) -> None:
        domain = domain.upper()
        split = split.lower()
        assert domain in ["MRI", "MR", "CT"], "Domain must be 'MRI', 'MR', or 'CT'."
        assert split in ["train", "test"], "Split must be 'train' or 'test'."

        self.base_path = base_path
        self.domain = domain
        self.split = split
        self.transform = transform
        self.slice_2d = slice_2d
        self.liver_only = liver_only

        split_dir = "Test_Sets" if self.split == "test" else "Train_Sets"
        domain_dir = "CT" if self.domain == "CT" else "MR"

        self.data_path = (
            Path(self.base_path) / f"CHAOS_{split_dir}" / split_dir / domain_dir
        )
        self.samples = self._load_samples()
        self.class_to_idx = {
            cls: i
            for i, cls in enumerate(
                chaos_labels_ct if self.domain == "CT" else chaos_labels_mr
            )
        }

    def _load_samples(self):
        samples = []
        for patient_id in sorted(self.data_path.iterdir()):
            if not patient_id.is_dir():
                continue
            if self.slice_2d:
                img_path, seg_path = self._get_paths(patient_id, self.domain)
                img_files = sorted(img_path.glob("*.dcm"))
                seg_files = sorted(seg_path.glob("*.png"))
                for img_file, seg_file in zip(img_files, seg_files):
                    if img_file.suffix == ".dcm" and seg_file.suffix == ".png":
                        samples.append((img_file, seg_file))
            else:
                samples.append(patient_id)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_paths(self, patient_id: Path, domain: str) -> Tuple[Path, Path]:
        if domain == "CT":
            img_path = patient_id / "DICOM_anon"
            seg_path = patient_id / "Ground"
        else:
            img_path = patient_id / "T2SPIR" / "DICOM_anon"
            seg_path = patient_id / "T2SPIR" / "Ground"
        return img_path, seg_path

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if self.slice_2d:
            img_file, seg_file = self.samples[idx]
            img = pydicom.dcmread(img_file).pixel_array.astype(np.float32)
            seg = (
                np.array(Image.open(seg_file)).astype(np.int64)
                if self.split == "train"
                else None
            )
            if self.liver_only and self.domain == "MR":
                # Filter out non-liver labels
                seg = np.where((seg >= 55) & (seg <= 70), seg, 0)
            # print(img.shape)
        else:
            patient_id = self.samples[idx]
            img_path, seg_path = self._get_paths(patient_id, self.domain)

            img_files = sorted(img_path.glob("*.dcm"))
            seg_files = sorted(seg_path.glob("*.png"))
            img_slices = [
                pydicom.dcmread(img_file).pixel_array.astype(np.float32)
                for img_file in img_files
                if img_file.suffix == ".dcm"
            ]

            seg_slices = [
                np.array(Image.open(seg_file)).astype(np.int64)
                for seg_file in seg_files
                if seg_file.suffix == ".png"
            ]
            if self.liver_only and self.domain == "MR":
                # Filter out non-liver labels
                seg_slices = [
                    np.where((seg >= 55) & (seg <= 70), seg, 0) for seg in seg_slices
                ]
            img = np.stack(img_slices, axis=-1)
            if self.split == "train":
                seg = np.stack(seg_slices, axis=-1)
            else:
                seg = None
            # img and seg are (W, H, D)
            # Ensure (C, D, H, W)
            img = img.transpose(2, 0, 1)  # (W, H, D) -> (D, H, W)
            img = img[np.newaxis, ...]  # Add channel dimension
            if seg is not None:
                seg = seg.transpose(2, 0, 1)
                seg = seg[np.newaxis, ...]

        sample = {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(seg) if seg is not None else None,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class CHAOS(BaseDataset):
    def __init__(
        self,
        location,
        domain: str,
        slice_2d: bool = False,
        liver_only: bool = False,
        preprocess=None,
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
            location,
            domain,
            slice_2d,
            "train",
            preprocess,
            liver_only=liver_only,  # , download=True
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PyTorchCHAOS(
            location,
            domain,
            slice_2d,
            "test",
            preprocess,
            liver_only=liver_only,  # , download=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if self.test_dataset[0]["label"] is None:
            self.test_loader = None

        # Set up class information
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]

        if self.domain == "CT":
            self.num_classes = len(chaos_labels_ct)
        else:
            self.num_classes = len(chaos_labels_mr)

    def get_model(self):
        """Return a Medical3DSegmenter with semantic guidance for CHAOS dataset."""
        from src.semantic_segmentation import (
            Medical3DSegmenter,
            CHAOS_CLASS_DESCRIPTIONS,
        )

        class_descriptions = CHAOS_CLASS_DESCRIPTIONS[self.domain]
        num_classes = len(class_descriptions)

        model = Medical3DSegmenter(
            encoder_type="swin_unetr",
            num_classes=num_classes,
            class_descriptions=class_descriptions,
            pretrained=True,
        )
        model.dataset = self
        return model

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

    def de_encode(self, labels):
        """De-encode labels to match the original dataset encoding."""
        if self.domain in ["MR", "MRI"]:
            return labels // 63
        elif self.domain == "CT":
            return torch.where(labels > 0, 1, 0)
