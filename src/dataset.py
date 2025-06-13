from abc import abstractmethod
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import torch
from monai.data import CacheDataset
from PIL import Image


class BaseDataset(CacheDataset):
    def __init__(
        self, base_path: str, domain: str, split: str = "train", *args, **kwargs
    ):
        domain = domain.upper()
        split = split.lower()
        assert domain in [
            "MRI",
            "MR",
            "CT",
        ], f"Domain {domain} is invalid. It must be either 'MRI'/'MR' or 'CT'."
        assert split in [
            "train",
            "test",
        ], f"Split {split} is invalid. It must be either 'train' or 'test'."

        self.base_path = base_path
        self.split = split
        self.domain = domain

        data = self._load_data()
        super().__init__(data, *args, **kwargs)

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError("Subclasses must implement this method.")


class CHAOSDataset(BaseDataset):

    def _load_data(self):
        split_dir = "Test_Sets" if self.split == "test" else "Train_Sets"
        domain = "CT" if self.domain == "CT" else "MR"

        data_path = Path(self.base_path) / f"CHAOS_{split_dir}" / split_dir / domain

        data = []

        for patient_id in sorted(data_path.iterdir()):
            if not patient_id.is_dir():
                # print("Skipping non-directory:", patient_id)
                continue
            # print("Processing patient:", patient_id)

            if domain == "CT":
                img_path = patient_id / "DICOM_anon"
                seg_path = patient_id / "Ground"
            else:  # Ignoring T1DUAL for now
                img_path = patient_id / "T2SPIR" / "DICOM_anon"
                seg_path = patient_id / "T2SPIR" / "Ground"

            # print(f"Loading data from {img_path} and labels from {seg_path}")

            img_files = sorted(img_path.glob("*.dcm"))
            seg_files = sorted(seg_path.glob("*.png"))
            # print("img_files", img_files)
            # print("seg_files", seg_files)
            # Load image slices
            img_slices = []
            for img_file in img_files:
                if img_file.suffix == ".dcm":
                    dicom_data = pydicom.dcmread(img_file)
                    img_data = dicom_data.pixel_array.astype(np.float32)
                    img_slices.append(img_data)
                else:
                    raise ValueError(f"Unsupported file format: {img_file}")

            # Load segmentation slices
            seg_slices = []
            for seg_file in seg_files:
                if seg_file.suffix == ".png":
                    seg_slice = np.array(Image.open(seg_file)).astype(np.int64)
                    seg_slices.append(seg_slice)
                else:
                    raise ValueError(
                        f"Unsupported segmentation file format: {seg_file}"
                    )

            # Stack slices into 3D arrays
            img = np.stack(img_slices, axis=-1)
            seg = np.stack(seg_slices, axis=-1)

            sample = {
                "image": torch.from_numpy(img)[None],  # add channel dim
                "label": torch.from_numpy(seg)[None],
            }
            if hasattr(self, "transform") and self.transform:
                sample = self.transform(sample)

            data.append(sample)

        return data


class MMWHSDataset(BaseDataset):
    def _load_data(self):
        """
        data/MM-WHS
        └── MM-WHS 2017 Dataset
            └── MM-WHS 2017 Dataset
            ├── ct_test/         # imgs, .nii.gz
            ├── ct_train/        # imgs + labels, .nii.gz
            ├── mr_test/         # imgs, .nii.gz
            └── mr_train/        # imgs + labels, .nii.gz
        └── MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic
            └── nii/                # labels, .nii.gz

        """
        domain = "ct" if self.domain == "CT" else "mr"

        data_path_images = (
            Path(self.base_path) / "MM-WHS 2017 Dataset" / f"{domain}_{self.split}"
        )

        if self.split == "train":
            data_path_labels = data_path_images
        else:
            data_path_labels = (
                Path(self.base_path)
                / "MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic"
                / "nii"
            )

        data = []
        # print(
        #     f"Loading MM-WHS dataset from {data_path_images} and labels from {data_path_labels}"
        # )
        # print(list(data_path_images.iterdir()))
        for img_file in sorted(data_path_images.glob("*1_image.nii.gz")):
            if not str(img_file).endswith(".nii.gz"):
                print(f"Skipping non-NIfTI file: {img_file}")
                continue

            # Load image volume
            img_nii = nib.load(str(img_file))
            img_data = img_nii.get_fdata().astype(np.float32)

            # Load label volume if available
            if self.split == "train":
                label_file = img_file.parent / img_file.name.replace(
                    "_image.nii.gz", "_label.nii.gz"
                )
                if label_file.exists():
                    label_nii = nib.load(str(label_file))
                    label_data = label_nii.get_fdata().astype(np.int64)
                else:
                    raise FileNotFoundError(f"Label file {label_file} does not exist.")
            else:
                label_file = data_path_labels / img_file.replace(
                    "_image.nii.gz", "_label_encrypt_1mm.nii.gz"
                )
                if label_file.exists():
                    label_nii = nib.load(str(label_file))
                    label_data = label_nii.get_fdata().astype(np.int64)
                else:
                    raise FileNotFoundError(f"Label file {label_file} does not exist.")

            sample = {
                "image": torch.from_numpy(img_data),
                "label": torch.from_numpy(label_data),
            }

            # print(sample)
            if hasattr(self, "transform") and self.transform:
                sample = self.transform(sample)

            data.append(sample)

        return data
        # raise NotImplementedError("MM-WHS dataset loading is not implemented yet.")
