import glob
import os

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CHAOSCTDataLoader(DataLoader):
    """
    Custom DataLoader for CHAOS CT volumes.
    Expects on disk:
      {base_path}/CHAOS_Train_Sets/Train_Sets/CT/{patient_id}/*.nii.gz
      {base_path}/CHAOS_Train_Sets/Train_Sets/Segmentation/{patient_id}/*.nii.gz
    """

    def __init__(
        self,
        dataset: Dataset,
        base_path: str,
        split: str = "Train_Sets",
        transform=None,
    ):
        self.base_path = base_path
        self.split = split
        self.transform = transform
        ct_dir = os.path.join(base_path, f"CHAOS_{split}", split, "CT")
        self.patient_ids = sorted(
            [d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d))]
        )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        img_pattern = os.path.join(
            self.base_path,
            f"CHAOS_{self.split}",
            self.split,
            "CT",
            pid,
            "*.nii.gz",
        )
        seg_pattern = os.path.join(
            self.base_path,
            f"CHAOS_{self.split}",
            self.split,
            "Segmentation",
            pid,
            "*.nii.gz",
        )
        img_file = glob.glob(img_pattern)[0]
        seg_file = glob.glob(seg_pattern)[0]

        img = nib.load(img_file).get_fdata().astype(np.float32)
        seg = nib.load(seg_file).get_fdata().astype(np.int64)

        sample = {
            "image": torch.from_numpy(img)[None],  # add channel dim
            "label": torch.from_numpy(seg)[None],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class CHAOSMRIDataLoader(DataLoader):
    """
    Custom DataLoader for CHAOS MRI volumes.
    """

    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        if hasattr(dataset, "transform") and dataset.transform is not None:
            self.transform = dataset.transform
        self.patient_ids = sorted(
            [
                d
                for d in os.listdir(
                    os.path.join(dataset.base_path, "CHAOS_Train_Sets/Train_Sets/MR")
                )
                if os.path.isdir(os.path.join(dataset.base_path, d))
            ]
        )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):

        # Get image and label paths from dataset
        img_path = self.dataset.data[idx]
        seg_path = self.dataset.labels[idx]

        # Load image and segmentation using nibabel
        # Load all slices from the directory
        img_files = sorted(glob.glob(os.path.join(os.path.dirname(img_path), "*")))
        seg_files = sorted(glob.glob(os.path.join(os.path.dirname(seg_path), "*")))

        # Load image slices
        img_slices = []
        for img_file in img_files:
            if img_file.endswith(".dcm"):
                img_data = nib.load(img_file).get_fdata().astype(np.float32)
                img_slices.append(img_data)
            else:
                raise ValueError(f"Unsupported MRI file format: {img_file}")

        # Load segmentation slices
        seg_slices = []
        for seg_file in seg_files:
            if seg_file.endswith(".png"):
                seg_slice = np.array(Image.open(seg_file)).astype(np.int64)
                seg_slices.append(seg_slice)
            else:
                raise ValueError(f"Unsupported segmentation file format: {seg_file}")

        # Stack slices into 3D arrays
        img = np.stack(img_slices, axis=-1)
        seg = np.stack(seg_slices, axis=-1)

        sample = {
            "image": torch.from_numpy(img)[None],  # add channel dim
            "label": torch.from_numpy(seg)[None],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class MMWHSCTDataLoader:
    """
    Custom DataLoader for MMHS DataLoader.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """

    pass


class MMWHSMRIDataLoader:
    """
    Custom DataLoader for MMHS DataLoader.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """

    pass


class APISCTDataLoader:
    """
    Custom DataLoader for APIS DataLoader.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """

    pass


class APISMRIDataLoader:
    """
    Custom DataLoader for APIS DataLoader.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """

    pass
