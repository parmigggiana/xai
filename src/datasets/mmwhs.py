from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from matplotlib import cm
from torchvision.datasets.vision import VisionDataset

from src.datasets.common import BaseDataset

mmwhs_labels_ct = ["Background", "Heart", "Liver", "Spleen", "Kidneys"]
mmwhs_labels_mr = ["Background", "Heart", "Liver", "Spleen", "Kidneys"]


class PyTorchMMWHS(VisionDataset):
    """
    MM-WHS Dataset for CT and MRI volumes.

    Args:
        base_path (str): Root directory of the dataset.
        domain (str): Either 'CT' or 'MRI'.
        split (str): Either 'train' or 'test'.
        transform (callable, optional): Transform to apply to the samples.
        slice_2d (bool): If True, slices the 3D volumes into 2D slices. Defaults to False (3D).
    """

    def __init__(
        self,
        base_path: str,
        domain: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        slice_2d: bool = False,
    ) -> None:
        domain = domain.lower()
        split = split.lower()
        assert domain in ["ct", "mr"], "Domain must be 'CT' or 'MR'."
        assert split in ["train", "test"], "Split must be 'train' or 'test'."

        self.base_path = base_path
        self.domain = domain
        self.split = split
        self.transform = transform
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

        self.samples = self._load_samples()
        self.class_to_idx = {
            cls: i
            for i, cls in enumerate(
                mmwhs_labels_ct if self.domain == "ct" else mmwhs_labels_mr
            )
        }

    def _load_samples(self):
        samples = []
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
                    img_header = nib.load(str(img_file)).header
                    num_slices = img_header.get_data_shape()[2]  # Z dimension
                    for slice_n in range(num_slices):
                        samples.append(
                            {
                                "image_path": img_file,
                                "slice_n": slice_n,
                                "label_path": label_file,
                            }
                        )
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
            else:
                samples.append(
                    {
                        "image_path": img_file,
                        "label_path": label_file,
                    }
                )

        return samples

    def _load_image(self, img_file):
        img_nii = nib.load(str(img_file))
        return img_nii.get_fdata().astype(np.float32)

    def _load_label(self, img_file):
        label_file = img_file.parent / img_file.name.replace(
            "_image.nii.gz", "_label.nii.gz"
        )
        if label_file.exists():
            label_nii = nib.load(str(label_file))
            return label_nii.get_fdata().astype(np.int64)
        else:
            raise FileNotFoundError(f"Label file {label_file} does not exist.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = self.samples[idx]
        img_data = self._load_image(sample["image_path"])
        label_data = (
            self._load_label(sample["label_path"])
            if sample["label_path"] is not None
            else None
        )
        if self.slice_2d:
            data = {
                "image": torch.from_numpy(img_data[:, :, sample["slice_n"]]),
                "label": (
                    torch.from_numpy(label_data[:, :, sample["slice_n"]])
                    if label_data is not None
                    else None
                ),
            }
        else:
            img_data = img_data.transpose(2, 0, 1)  # (W, H, D) -> (D, H, W)
            img_data = img_data[np.newaxis, ...]  # Add channel dimension
            if label_data is not None:
                label_data = label_data.transpose(2, 0, 1)
                label_data = label_data[np.newaxis, ...]
            data = {
                "image": torch.from_numpy(img_data),
                "label": (
                    torch.from_numpy(label_data) if label_data is not None else None
                ),
            }

        if self.transform:
            data = self.transform(data)

        return data


class MMWHS(BaseDataset):
    def __init__(
        self,
        location,
        domain: str,
        slice_2d: bool = False,
        preprocess=None,
        batch_size=1,
        num_workers=0,
    ):
        """
        MMWHS Test does not have labels, so we only use it for inference.
        """
        super().__init__()
        self.train_dataset = PyTorchMMWHS(
            location, domain, "train", preprocess, slice_2d=slice_2d
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            # collate_fn=lambda x: x[0] if len(x) == 1 else x,
        )

        self.test_dataset = PyTorchMMWHS(
            location, domain, "test", preprocess, slice_2d=slice_2d
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            # collate_fn=lambda x: x[0] if len(x) == 1 else x,
        )

        if self.test_dataset[0]["label"] is None:
            self.test_loader = None

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]
        self.num_classes = len(self.classnames)
        self.domain = domain
        self.slice_2d = slice_2d

    def get_model(self):
        """Return a Medical3DSegmenter with semantic guidance for MMWHS dataset."""
        from src.semantic_segmentation import (
            Medical3DSegmenter,
            MMWHS_CLASS_DESCRIPTIONS,
        )

        class_descriptions = MMWHS_CLASS_DESCRIPTIONS[self.domain.upper()]
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
            rotate=1 if self.domain in ["MR", "MRI"] else 0,
            flip_axis=0 if self.domain in ["MR", "MRI"] else None,
        )

    def visualize_sample_slice(self, sample):
        self._visualize_sample_slice(
            sample, rotate=1 if self.domain in ["MR", "MRI"] else 0, flip_axis=None
        )

    def _get_organ_legend(self, seg_slice):
        legend = {}
        mmwhs_labels = [
            (500, "Left ventricle blood cavity", 0),
            (600, "Right ventricle blood cavity", 1),
            (420, "Left atrium blood cavity", 2),
            (550, "Right atrium blood cavity", 3),
            (205, "Myocardium of the left ventricle", 4),
            (820, "Ascending aorta", 5),
            (850, "Pulmonary artery", 6),
        ]
        set1 = cm.get_cmap("Set1", 8)
        for label_val, organ_name, color_idx in mmwhs_labels:
            if np.any(seg_slice == label_val):
                legend[organ_name] = set1(color_idx)

        return legend

    def de_encode(self, labels):
        mmwhs_labels = {
            500: 0,
            600: 1,
            420: 2,
            550: 3,
            205: 4,
            820: 5,
            850: 6,
        }

        de_encoded_labels = np.zeros_like(labels, dtype=np.int64)
        for original_label, new_label in mmwhs_labels.items():
            de_encoded_labels[labels == original_label] = new_label
        return de_encoded_labels
