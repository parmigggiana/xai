from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from matplotlib import cm
from monai.data import DataLoader, ITKReader

from src.datasets.common import BaseDataset
from src.datasets.custom_imageDataset import ImageDataset

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


class PyTorchMMWHS(ImageDataset):
    """
    MM-WHS Dataset for CT and MRI volumes using MONAI's ImageDataset.

    Args:
        base_path (str): Root directory of the dataset.
        domain (str): Either 'CT' or 'MR'.
        split (str): Either 'train' or 'test'.
        transform (callable, optional): Transform to apply to the images.
        seg_transform (callable, optional): Transform to apply to the segmentations.
        slice_2d (bool): If True, slices the 3D volumes into 2D slices. Defaults to False (3D).
    """

    def __init__(
        self,
        base_path: str,
        domain: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        slice_2d: bool = False,
    ) -> None:
        domain = domain.lower()
        split = split.lower()
        assert domain in ["ct", "mr"], "Domain must be 'CT' or 'MR'."
        assert split in ["train", "test"], "Split must be 'train' or 'test'."

        self.base_path = base_path
        self.domain = domain
        self.split = split
        self.slice_2d = slice_2d

        self.data_path_images = (
            Path(self.base_path)
            / "MM-WHS 2017 Dataset"
            / "MM-WHS 2017 Dataset"
            / f"{self.domain}_{self.split}"
        )

        if self.split == "train":
            self.data_path_labels = self.data_path_images
        else:
            self.data_path_labels = (
                Path(self.base_path)
                / "MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic"
                / "nii"
            )

        # Load file lists for ImageDataset
        image_files, seg_files = self._load_file_lists()

        # Initialize ImageDataset with file lists
        super().__init__(
            image_files=image_files,
            seg_files=seg_files if self.split == "train" else None,
            transform=transform,
            seg_transform=seg_transform,
            image_only=False,
            transform_with_metadata=True,
        )

    def _load_file_lists(self):
        """Load file lists for ImageDataset initialization."""
        image_files = []
        seg_files = []

        for img_file in sorted(self.data_path_images.glob("*_image.nii.gz")):
            if not str(img_file).endswith(".nii.gz"):
                continue

            label_file = None
            if self.split == "train":
                label_file = img_file.parent / img_file.name.replace(
                    "_image.nii.gz", "_label.nii.gz"
                )
                if not label_file.exists():
                    raise FileNotFoundError(f"Label file {label_file} does not exist.")

            if self.slice_2d:
                # For 2D slicing, we need to know the number of slices
                try:
                    itk_reader = ITKReader()
                    img_data = itk_reader.read(str(img_file))
                    img_array, _ = itk_reader.get_data(img_data)
                    num_slices = img_array.shape[2]  # Z dimension
                    for slice_n in range(num_slices):
                        # For 2D ImageDataset, we'll need custom handling
                        # For now, add the 3D file and handle slicing in transforms
                        image_files.append(str(img_file))
                        if self.split == "train" and label_file:
                            seg_files.append(str(label_file))
                        break  # Add each volume only once for now
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
            else:
                image_files.append(str(img_file))
                if self.split == "train" and label_file:
                    seg_files.append(str(label_file))

        return image_files, seg_files


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
    ):
        """
        MMWHS Test does not have labels, so we only use it for inference.
        """
        super().__init__()
        self.train_dataset = PyTorchMMWHS(
            base_path=str(location),
            domain=domain,
            split="train",
            transform=transform,
            seg_transform=seg_transform,
            slice_2d=slice_2d,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.test_dataset = PyTorchMMWHS(
            base_path=str(location),
            domain=domain,
            split="test",
            transform=transform,
            seg_transform=seg_transform,
            slice_2d=slice_2d,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.domain = domain
        self.slice_2d = slice_2d
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
        for i, (label_val, organ_name) in enumerate(mmwhs_labels):
            if np.any(seg_slice == label_val):
                legend[organ_name] = set1(i + 1)  # Skip the first color (background)

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
