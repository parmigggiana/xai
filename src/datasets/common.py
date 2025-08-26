try:
    import napari

    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    napari = None

import numpy as np
import torch
from matplotlib import cm

from src.semantic_segmentation import MedicalSegmenter


class BaseDataset:
    def __init__(self):
        self.name = type(self).__name__

    @classmethod
    def _get_organ_legend(cls, seg_slice):

        print(f"Warning: No specific legend for dataset {type(cls)}.")
        set1 = cm.get_cmap("Set1", 8)  # Set1 is qualitative, 8 distinct colors
        legend = {}
        unique_labels = np.unique(seg_slice)
        unique_labels = unique_labels[unique_labels > 0]
        for idx, label in enumerate(unique_labels):
            legend[label] = set1(idx % set1.N)
        return legend

    def get_model(self, encoder_type="swin_unetr"):
        """
        Args:
            encoder_type (str): Type of encoder to use ('swin_unetr', 'resnet', or 'clipseg')

        Returns:
            MedicalSegmenter or CLIPSeg: Model with semantic guidance capabilities
        """

        # Original logic for other encoder types
        model = MedicalSegmenter(
            encoder_type=encoder_type,
            num_classes=self.num_classes,
            pretrained=True,
            dataset=self,
        )
        return model

    def visualize_3d(self, sample):
        """
        Visualize a 3D volumetric image sample and its segmentation mask using 3D rendering.

        Args:
            sample (dict): contains 'image' and 'label'.
        """
        if not NAPARI_AVAILABLE:
            print("⚠️  napari is not available. Install napari to use 3D visualization:")
            print("   pip install 'napari[pyqt6,optional]==0.6.2a1'")
            print("   Falling back to 2D slice visualization...")
            self.visualize_sample_slice(sample)
            return

        self._visualize_3d(sample)

    @torch.no_grad()
    def _visualize_3d(
        self,
        sample,
        rotate: int = 0,
        flip_axis: int = None,
    ):
        """
        Visualize a 3D volumetric image sample and its segmentation mask using 3D rendering.

        Args:
            dataloader (DataLoader): yields batches with 'image' and 'label'.
            sample_index (int): index of the batch to visualize.
            device (torch.device, optional): computation device.
            dataset_name (str): name of the dataset for legend labeling.
        """
        if not NAPARI_AVAILABLE:
            print("⚠️  napari is not available for 3D visualization")
            return

        img, seg = sample["image"], sample["label"]
        # print(img.shape)
        if img.ndim < 3:
            return

        # Rotate for correct orientation
        img = np.rot90(img, k=rotate, axes=(0, 1))
        seg = np.rot90(seg, k=rotate, axes=(0, 1))

        if flip_axis is not None:
            if isinstance(flip_axis, int):
                flip_axis = (flip_axis,)
            for axis in flip_axis:
                img = np.flip(img, axis=axis)
                seg = np.flip(seg, axis=axis)

        # Create a Napari viewer
        viewer = napari.Viewer()

        # Scale z-axis to make layers appear taller in 3D view
        z_scale = 1.0
        if self.domain in ["MR", "MRI"]:
            z_scale = 5.0  # Increase this value to make layers taller

        scale = (1.0, 1.0, z_scale) if img.ndim == 3 else (1.0, 1.0, 1.0, z_scale)

        # Add image and segmentation layers with scaling
        viewer.add_image(
            img, name="Image", colormap="gray", blending="additive", scale=scale
        )
        viewer.add_labels(seg, name="Segmentation", opacity=0.5, scale=scale)

        # Start the Napari event loop
        napari.run()

    def visualize_sample_slice(
        self,
        sample,
    ):
        """
        Visualize a volumetric image sample and its segmentation mask (center slice).
        """
        self._visualize_sample_slice(sample)

    @torch.no_grad()
    def _visualize_sample_slice(
        self,
        sample,
        rotate: int = 0,
        flip_axis: int = None,
    ) -> None:
        """
        Visualize a volumetric image sample and its segmentation mask (center slice).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap

        img, seg = sample["image"], sample["label"]

        # Convert to numpy if tensors
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        if hasattr(seg, "cpu"):
            seg = seg.cpu().numpy()

        # Handle different input shapes
        if img.ndim == 3:
            if img.shape[0] <= 3:  # Likely (C, H, W) format
                img_slice = img
                seg_slice = seg.squeeze() if seg.ndim > 2 else seg
            else:  # (H, W, D) format
                z = img.shape[-1] // 2
                img_slice = img[..., z]
                seg_slice = seg[..., z]
        elif img.ndim == 4:
            z = img.shape[-1] // 2
            img_slice = img[0, ..., z]
            seg_slice = seg[0, ..., z]
        elif img.ndim == 2:
            img_slice = img
            seg_slice = seg
        elif img.ndim == 5:  # (B, C, D, W, H)
            z = img.shape[2] // 2
            img_slice = img[0, 0, z, ...]
            seg_slice = seg[0, 0, z, ...]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # Ensure seg_slice is at least 2D for rotation
        if seg_slice.ndim < 2:
            print(
                f"Warning: seg_slice is {seg_slice.ndim}D with shape {seg_slice.shape}. Skipping rotation."
            )
            rotate = 0  # Skip rotation for 1D arrays

        # Apply rotation exactly once (if requested)
        if rotate != 0:
            # Rotate image slice
            if img_slice.ndim == 3 and img_slice.shape[0] in (1, 3):
                # (C, H, W): rotate per-channel
                channels = []
                for c in range(img_slice.shape[0]):
                    channels.append(np.rot90(img_slice[c], k=rotate))
                img_slice = np.stack(channels, axis=0)
            elif img_slice.ndim >= 2:
                # 2D (H, W) or other 2D-like
                img_slice = np.rot90(img_slice, k=rotate)

            # Rotate segmentation slice
            if seg_slice.ndim >= 2:
                seg_slice = np.rot90(seg_slice, k=rotate)
            else:
                print(
                    f"Warning: Cannot rotate seg_slice with shape {seg_slice.shape} (ndim={seg_slice.ndim})"
                )

        # Create the plot
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Apply flipping if specified and data is at least 2D
        if flip_axis is not None and seg_slice.ndim >= 2:
            if isinstance(flip_axis, int):
                flip_axis = (flip_axis,)

            print(f"IMAGE SHAPE: {img_slice.shape}, SEG SHAPE: {seg_slice.shape}")
            for axis in flip_axis:
                if axis < img_slice.ndim:
                    img_slice = np.flip(img_slice, axis=axis)
                if axis < seg_slice.ndim:
                    seg_slice = np.flip(seg_slice, axis=axis)

        # Display the image slice
        if img_slice.ndim == 3:
            if img_slice.shape[-1] == 3:  # Already (H, W, 3)
                ax1.imshow(img_slice)
            elif img_slice.shape[0] == 3:  # Still (3, H, W), transpose it
                img_display = np.transpose(img_slice, (1, 2, 0))
                ax1.imshow(img_display)
            else:  # Other multi-channel, take first channel
                ax1.imshow(img_slice[0], cmap="gray")
        elif img_slice.ndim == 2:  # 2D grayscale
            ax1.imshow(img_slice, cmap="gray")
        else:  # 1D or other cases
            print(f"Cannot display image slice with shape {img_slice.shape}")
            ax1.text(
                0.5,
                0.5,
                f"Cannot display\nshape: {img_slice.shape}",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        ax1.set_title("Image Slice")
        ax1.axis("off")

        # Display overlay
        if img_slice.ndim == 3:
            # Use grayscale background for overlay
            if img_slice.shape[-1] == 3:  # (H, W, 3)
                img_gray = np.dot(img_slice, [0.299, 0.587, 0.114])
            elif img_slice.shape[0] == 3:  # (3, H, W)
                img_display = np.transpose(img_slice, (1, 2, 0))
                img_gray = np.dot(img_display, [0.299, 0.587, 0.114])
            else:
                img_gray = (
                    img_slice[0] if img_slice.shape[0] > 1 else img_slice.squeeze()
                )
            ax2.imshow(img_gray, cmap="gray")
        elif img_slice.ndim == 2:
            ax2.imshow(img_slice, cmap="gray")
        else:
            ax2.text(
                0.5,
                0.5,
                f"Cannot display\nshape: {img_slice.shape}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        # Create masked overlay only if seg_slice is 2D
        if seg_slice.ndim >= 2:
            masked_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)

            # Get legend and display overlay with colors
            legend = self._get_organ_legend(seg_slice)
            if legend:
                legend_colors = ListedColormap([legend[label] for label in legend])
                ax2.imshow(masked_overlay, cmap=legend_colors, alpha=0.4)

                legend_elements = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="s",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                        label=label,
                    )
                    for label, color in legend.items()
                ]
                ax2.legend(
                    handles=legend_elements,
                    loc="upper right",
                    frameon=True,
                    facecolor="white",
                )
        else:
            ax2.text(
                0.5,
                0.5,
                f"Cannot display segmentation\nshape: {seg_slice.shape}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        ax2.set_title("Overlay Segmentation")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()
