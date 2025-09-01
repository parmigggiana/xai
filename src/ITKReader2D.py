from monai.data import ITKReader


class ITKReader2D(ITKReader):
    """
    Custom ITKReader that ensures proper dimensionality for 2D slices.
    Fixes the issue where single DICOM slices get depth dimension of size 0.
    """

    def __init__(self, *args, **kwargs):
        # kwargs.setdefault("affine_lps_to_ras", False) # Keep raw storage orientation to match CHAOS PNG masks (no LPS->RAS reorientation)
        super().__init__(*args, **kwargs)

    def __call__(self, filename, reader=None):
        """Read image and ensure proper 2D slice dimensions."""
        # Use parent ITKReader to load the image
        img_array, meta = super().__call__(filename, reader)

        # If we have a problematic depth dimension (size 0), fix it by reshaping
        if len(img_array.shape) >= 3 and img_array.shape[-1] == 0:
            # Create new shape with depth dimension set to 1
            new_shape = list(img_array.shape)
            new_shape[-1] = 1  # Set depth dimension to 1
            img_array = img_array.reshape(new_shape)

        return img_array, meta
