import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class CHAOSCTDataLoader(Dataset):
    """
    Custom Dataset for CHAOS CT volumes.
    Expects on disk:
      {base_path}/CHAOS_Train_Sets/Train_Sets/CT/{patient_id}/*.nii.gz
      {base_path}/CHAOS_Train_Sets/Train_Sets/Segmentation/{patient_id}/*.nii.gz
    """
    def __init__(self, base_path: str, split: str = "Train_Sets", transform=None):
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
            "image": torch.from_numpy(img)[None],   # add channel dim
            "label": torch.from_numpy(seg)[None],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class CHAOSMRIDataLoader(Dataset):
    """
    Custom Dataset for CHAOS MRI volumes.
    Expects on disk:
      {base_path}/CHAOS_Train_Sets/Train_Sets/MR/{patient_id}/*.nii.gz
      {base_path}/CHAOS_Train_Sets/Train_Sets/Segmentation/{patient_id}/*.nii.gz
    """
    def __init__(self, base_path: str, split: str = "Train_Sets", transform=None):
        self.base_path = base_path
        self.split = split
        self.transform = transform
        mr_dir = os.path.join(base_path, f"CHAOS_{split}", split, "MR")
        self.patient_ids = sorted(
            [d for d in os.listdir(mr_dir) if os.path.isdir(os.path.join(mr_dir, d))]
        )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        img_pattern = os.path.join(
            self.base_path,
            f"CHAOS_{self.split}",
            self.split,
            "MR",
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
            "image": torch.from_numpy(img)[None],   # add channel dim
            "label": torch.from_numpy(seg)[None],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class MMWHSCTDataLoader(base_path, dataset, batch_size=1, shuffle=True, num_workers=0, is_train=True):
    """
    Custom DataLoader for MMHS dataset.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """
   pass

class MMWHSMRIDataLoader(base_path, dataset, batch_size=1, shuffle=True, num_workers=0, is_train=True):
    """
    Custom DataLoader for MMHS dataset.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """
   pass

class APISCTDataLoader(base_path, dataset, batch_size=1, shuffle=True, num_workers=0, is_train=True):
    """
    Custom DataLoader for APIS dataset.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """
    pass

class APISMRIDataLoader(base_path, dataset, batch_size=1, shuffle=True, num_workers=0, is_train=True):
    """
    Custom DataLoader for APIS dataset.

    Inherits from torch.nn.DataLoader to provide custom functionality if needed.
    """
    pass