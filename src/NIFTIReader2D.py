"""
NIFTIReader2D - A specialized reader for loading specific 2D slices from NIfTI files.

This reader inherits from ITKReader and extends it to load specific slices by index
instead of loading entire volumes. It's designed for efficient 2D slice-based processing
of 3D medical images.
"""

from os import PathLike
import numpy as np
from typing import Dict, Any, Sequence, Tuple
from monai.data import NibabelReader


class NIFTIReader2D(NibabelReader):
    def __init__(self, **kwargs):
        """Initialize the NIFTIReader2D with ITKReader parameters."""
        super().__init__(**kwargs)

    def read(
        self, data: Sequence[Tuple[PathLike, int]] | Tuple[PathLike, int], **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read a specific slice from a NIfTI file.

        Args:
            file_path (str or Path): Path to the NIfTI file.
            slice_index (int): Index of the slice to read.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The image data for the specified slice and metadata.
        """
        # Load the entire volume first
        path_data = [d[0] for d in data] if isinstance(data, Sequence) else [data[0]]
        slice_idx = [d[1] for d in data] if isinstance(data, Sequence) else data[1]

        volume_data, meta_data = super().read(path_data, **kwargs)

        # Extract the specific slice
        if volume_data.ndim != 3:
            raise ValueError(
                "Expected 3D volume data, but got data with shape: {}".format(
                    volume_data.shape
                )
            )

        slice_data = volume_data[slice_idx, :, :]

        return slice_data, meta_data
