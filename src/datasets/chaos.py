from pathlib import Path
from typing import Callable, Optional, Any, Tuple
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import pydicom

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
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        domain = domain.upper()
        split = split.lower()
        assert domain in ["MRI", "MR", "CT"], "Domain must be 'MRI', 'MR', or 'CT'."
        assert split in ["train", "test"], "Split must be 'train' or 'test'."

        self.base_path = base_path
        self.domain = domain
        self.split = split
        self.transform = transform

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
            samples.append(patient_id)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        patient_id = self.samples[idx]
        if self.domain == "CT":
            img_path = patient_id / "DICOM_anon"
            seg_path = patient_id / "Ground"
        else:
            img_path = patient_id / "T2SPIR" / "DICOM_anon"
            seg_path = patient_id / "T2SPIR" / "Ground"

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
        img = np.stack(img_slices, axis=-1)
        if self.split == "train":
            seg = np.stack(seg_slices, axis=-1)
        else:
            seg = None

        sample = {
            "image": torch.from_numpy(img)[None],
            "label": torch.from_numpy(seg)[None] if seg is not None else None,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class CHAOS:
    def __init__(
        self,
        location,
        domain: str,
        preprocess=None,
        batch_size=1,
        num_workers=16,
    ):
        """
        CHAOS Test does not have labels, so we only use it for inference.
        """

        self.train_dataset = PyTorchCHAOS(
            location, domain, "train", preprocess  # , download=True
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],
        )

        self.test_dataset = PyTorchCHAOS(
            location, domain, "test", preprocess  # , download=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]
