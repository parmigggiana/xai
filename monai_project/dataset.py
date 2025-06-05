from pathlib import Path

import numpy as np
import pydicom
import torch
from monai.data import CacheDataset
from PIL import Image


class CHAOSDataset(CacheDataset):
    def __init__(
        self,
        base_path: str,
        domain: str,
        split: str = "Train_Sets",
        *args,
        **kwargs,
    ):
        """
        Initializes the CHAOS MRI Dataset.

        Args:
            data (list): List of MRI data samples.
            labels (list): List of corresponding labels for the data samples.
        """
        assert domain.upper() in [
            "MRI",
            "MR",
            "CT",
        ], "Domain must be either 'MRI'/'MR' or 'CT'."
        assert split.lower() in [
            "train",
            "test",
        ], "Split must be either 'train' or 'test'."

        self.base_path = base_path
        self.split = "Test_Sets" if split.lower() == "test" else "Train_Sets"
        self.domain = "CT" if domain.upper() == "CT" else "MR"

        # Load data and labels from the specified directory
        data = self._load_data()
        super().__init__(data, *args, **kwargs)

    def _load_data(self):
        """
        Loads MRI data and labels from the specified directory.

        Returns:
            list: List of dictionaries containing 'image' and 'label' tensors.
        """

        data_path = (
            Path(self.base_path) / f"CHAOS_{self.split}" / self.split / self.domain
        )

        data = []

        for patient_id in sorted(data_path.iterdir()):
            if not patient_id.is_dir():
                # print("Skipping non-directory:", patient_id)
                continue
            # print("Processing patient:", patient_id)

            if self.domain == "CT":
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
