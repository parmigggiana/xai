import napari
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap

from src.semantic_segmentation import Medical3DSegmenter


class BaseDataset:
    def __init__(self):
        self.name = type(self).__name__

    def _get_organ_legend(self, seg_slice):

        print(f"Warning: No specific legend for dataset {type(self)}.")
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
            encoder_type (str): Type of encoder to use ('swin_unetr' or 'resnet')

        Returns:
            Medical3DSegmenter: Model with semantic guidance capabilities
        """
        model = Medical3DSegmenter(
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

        img, seg = sample["image"], sample["label"]
        # Handle (H, W, D) or (C, H, W, D)

        if img.ndim == 3:
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

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        img_slice = np.rot90(img_slice, k=rotate)
        seg_slice = np.rot90(seg_slice, k=rotate)

        if flip_axis is not None:
            if isinstance(flip_axis, int):
                flip_axis = (flip_axis,)
            # print(f"Flipping along axes: {flip_axis}")
            for axis in flip_axis:
                # print(f"Flipping along axis {axis}")
                img_slice = np.flip(img_slice, axis=axis)
                seg_slice = np.flip(seg_slice, axis=axis)

        ax1.imshow(img_slice, cmap="gray")
        ax1.set_title("Image Slice")
        ax1.axis("off")

        ax2.imshow(img_slice, cmap="gray")
        masked_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)

        legend = self._get_organ_legend(seg_slice)
        legend_colors = ListedColormap([legend[label] for label in legend])
        legend_labels = list(legend.keys())
        ax2.imshow(masked_overlay, cmap=legend_colors, alpha=0.4)
        if legend_labels:
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
        ax2.set_title("Overlay Segmentation")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()
