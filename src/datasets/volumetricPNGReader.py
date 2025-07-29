"""
Custom volumetric image reader for CHAOS dataset.

This module provides a custom image reader that can load PNG image series
as 3D volumes, similar to how DICOM series are handled by ITKReader.
"""

import numpy as np
from pathlib import Path
from typing import Iterable, Union, Tuple, Dict, Sequence
from PIL import Image

from monai.data.image_reader import ImageReader
from monai.utils import ensure_tuple, optional_import

nib, _ = optional_import("nibabel")

# Constants
PNG_VOLUME_PREFIX = "PNG_VOLUME:"


class VolumetricPNGReader(ImageReader):
    """
    Custom image reader for loading PNG image series as 3D volumes.

    This reader inherits from MONAI's ImageReader and can handle:
    1. Individual PNG files (2D slices)
    2. Directories containing PNG files (reconstructed as 3D volumes)
    3. Special CHAOS_PNG_VOLUME: prefixed paths for volume reconstruction

    The reader works similarly to how DICOM series are handled, but for PNG files.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def verify_suffix(
        self, filename: Union[Sequence[Union[str, Path]], str, Path]
    ) -> bool:
        """
        Verify that the filename has a supported suffix or is a directory with PNG files.

        Args:
            filename: path to file or directory, or sequence of paths

        Returns:
            bool: True if the file(s) can be handled by this reader
        """
        suffixes = [".png", ".PNG"]

        if isinstance(filename, str):
            if filename.startswith(PNG_VOLUME_PREFIX):
                return True
            filename = Path(filename)

        if isinstance(filename, Path):
            if filename.is_dir():
                # Check if directory contains PNG files
                return any(
                    f.suffix.lower() in [".png"]
                    for f in filename.iterdir()
                    if f.is_file()
                )
            else:
                return filename.suffix in suffixes

        # Handle sequence of files
        filename = ensure_tuple(filename)
        return all(Path(f).suffix in suffixes for f in filename)

    def read(
        self, data: Union[Sequence[Union[str, Path]], str, Path], **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Read PNG file(s) and return numpy array and metadata.

        Args:
            data: path to PNG file, directory with PNG files, or CHAOS_PNG_VOLUME: prefixed path
            **kwargs: additional keyword arguments

        Returns:
            Tuple[np.ndarray, Dict]: image array and metadata dictionary
        """
        if isinstance(data, str) and data.startswith(PNG_VOLUME_PREFIX):
            # Handle special .PNG volume marker
            directory_path = Path(data.replace(PNG_VOLUME_PREFIX, ""))
            return self._read_png_volume(directory_path)

        if (
            isinstance(data, Sequence)
            and len(data) == 1
            and isinstance(data[0], (str, Path))
        ):
            data = data[0]

        if isinstance(data, (str, Path)):
            data_path = Path(data)

            if data_path.is_dir():
                # Directory containing PNG files
                return self._read_png_volume(data_path)
            else:
                # Single PNG file
                return self._read_single_png(data_path)

        # Sequence of PNG files
        data_paths = [Path(f) for f in ensure_tuple(data)]
        return self._read_png_sequence(data_paths)

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and metadata from the loaded image.

        Args:
            img: Tuple of (array, metadata) from read() method

        Returns:
            Tuple[np.ndarray, Dict]: image array and metadata
        """
        if isinstance(img, tuple) and len(img) == 2:
            return img[0], img[1]
        else:
            # Fallback if img is just the array
            return img, {}

    def _read_single_png(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read a single PNG file."""
        try:
            # Load PNG using PIL
            pil_image = Image.open(file_path)
            img_array = np.array(pil_image)

            # Convert to grayscale if needed
            if len(img_array.shape) == 3 and img_array.shape[2] > 1:
                img_array = img_array[:, :, 0]  # Take first channel

            # Ensure 3D array (H, W, 1) for consistency
            if len(img_array.shape) == 2:
                img_array = img_array[:, :, np.newaxis]

            metadata = {
                "filename_or_obj": str(file_path),
                "original_shape": img_array.shape,
                "spatial_shape": img_array.shape[:2],
                "format": "PNG",
                "dtype": str(img_array.dtype),
            }

            return img_array, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to read PNG file {file_path}: {str(e)}")

    def _read_png_sequence(self, file_paths: Sequence[Path]) -> Tuple[np.ndarray, Dict]:
        """Read a sequence of PNG files and stack them into a 3D volume."""
        if not file_paths:
            raise ValueError("No PNG files provided")

        # Sort files to ensure consistent ordering
        sorted_paths = sorted(file_paths)

        # Read first image to get dimensions
        first_img, _ = self._read_single_png(sorted_paths[0])
        height, width = first_img.shape[:2]

        # Create 3D volume array
        volume = np.zeros((height, width, len(sorted_paths)), dtype=first_img.dtype)

        # Load all slices
        for i, file_path in enumerate(sorted_paths):
            slice_img, _ = self._read_single_png(file_path)
            if len(slice_img.shape) == 3:
                slice_img = slice_img[:, :, 0]  # Take first channel if 3D
            volume[:, :, i] = slice_img

        metadata = {
            "filename_or_obj": [str(p) for p in sorted_paths],
            "original_shape": volume.shape,
            "spatial_shape": volume.shape,
            "format": "PNG_VOLUME",
            "dtype": str(volume.dtype),
            "num_slices": len(sorted_paths),
        }

        return volume, metadata

    def _read_png_volume(self, directory_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read all PNG files from a directory and reconstruct as 3D volume."""
        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Find all PNG files in the directory
        png_files = sorted(
            [
                f
                for f in directory_path.iterdir()
                if f.is_file() and f.suffix.lower() == ".png"
            ]
        )

        if not png_files:
            raise FileNotFoundError(
                f"No PNG files found in directory: {directory_path}"
            )
        return self._read_png_sequence(png_files)


class VolumetricImageLoader:
    """
    Custom loader that can handle both DICOM directories and PNG volumes.

    This class acts as a dispatcher, choosing the appropriate reader based on
    the input data type and format.
    """

    def __init__(self):
        from monai.data.image_reader import ITKReader

        self.dicom_reader = ITKReader()
        self.png_reader = VolumetricPNGReader()

    def __call__(self, data: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load image data using the appropriate reader.

        Args:
            data: path to file or directory, or special marker string

        Returns:
            Tuple[np.ndarray, Dict]: image array and metadata
        """
        if isinstance(data, str):
            if data.startswith(PNG_VOLUME_PREFIX):
                # Use PNG reader for CHAOS volumes
                return self.png_reader.read(data)

            data_path = Path(data)
        else:
            data_path = Path(data)

        if data_path.is_dir():
            # Check if directory contains DICOM or PNG files
            dicom_files = list(data_path.glob("*.dcm")) + list(data_path.glob("*.DCM"))
            png_files = list(data_path.glob("*.png")) + list(data_path.glob("*.PNG"))

            if dicom_files:
                # Use DICOM reader
                return self.dicom_reader.read(str(data_path))
            elif png_files:
                # Use PNG reader
                return self.png_reader.read(data_path)
            else:
                raise ValueError(
                    f"No supported image files found in directory: {data_path}"
                )

        else:
            # Single file - determine type by extension
            if data_path.suffix.lower() in [".png"]:
                return self.png_reader.read(data_path)
            else:
                # Assume DICOM or other ITK-supported format
                return self.dicom_reader.read(str(data_path))
