"""
Interface module for evaluating volumetric segmentation performance using MONAI.
"""

from typing import Optional

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from napari.utils.colormaps import DirectLabelColormap
from torch.utils.data import DataLoader


def _get_organ_legend(seg_slice, dataset_name):
    """
    Utility function to get legend labels and colors for organs present in the segmentation.
    Handles different conventions for CHAOS_MRI, CHAOS_CT, MM-WHS (CT/MRI), and other datasets.
    Uses the Set1 qualitative colormap (max 8-9 distinct colors).
    """
    legend = {}
    set1 = cm.get_cmap("Set1", 8)  # Set1 is qualitative, 8 distinct colors

    if dataset_name in ["CHAOS_MRI", "CHAOS_MR"]:
        organ_list = [
            ("Liver", (55, 70, 63), 0),
            ("Right kidney", (110, 135, 126), 1),
            ("Left kidney", (175, 200, 189), 2),
            ("Spleen", (240, 255, 252), 3),
        ]
        for organ_name, (min_val, max_val, _), color_idx in organ_list:
            if np.any((seg_slice >= min_val) & (seg_slice <= max_val)):
                legend[organ_name] = set1(color_idx)
    elif dataset_name == "CHAOS_CT":
        if np.any(seg_slice > 0):
            legend["Liver"] = set1(0)
    elif dataset_name in [
        "MMWHS",
        "MM-WHS",
        "MMWHS_CT",
        "MM-WHS_CT",
        "MMWHS_MRI",
        "MM-WHS_MRI",
    ]:
        mmwhs_labels = [
            (500, "Left ventricle blood cavity", 0),
            (600, "Right ventricle blood cavity", 1),
            (420, "Left atrium blood cavity", 2),
            (550, "Right atrium blood cavity", 3),
            (205, "Myocardium of the left ventricle", 4),
            (820, "Ascending aorta", 5),
            (850, "Pulmonary artery", 6),
        ]
        for label_val, organ_name, color_idx in mmwhs_labels:
            if np.any(seg_slice == label_val):
                legend[organ_name] = set1(color_idx)
    else:
        # Generic: just show unique nonzero labels
        print(f"Warning: No specific legend for dataset {dataset_name}.")
        unique_labels = np.unique(seg_slice)
        unique_labels = unique_labels[unique_labels > 0]
        for idx, label in enumerate(unique_labels):
            legend[label] = set1(idx % set1.N)
    return legend


def visualize_sample_slice(
    sample,
    device: Optional[torch.device] = None,
    dataset_name: str = "CHAOS_MRI",
    rotate: int = 0,
    flip_axis: int = -1,  # -1 for no flip, 0 for vertical, 1 for horizontal
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
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    img_slice = np.rot90(img_slice, k=rotate)
    seg_slice = np.rot90(seg_slice, k=rotate)

    if flip_axis != -1:
        img_slice = np.flip(img_slice, axis=flip_axis)
        seg_slice = np.flip(seg_slice, axis=flip_axis)

    ax1.imshow(img_slice, cmap="gray")
    ax1.set_title("Image Slice")
    ax1.axis("off")

    ax2.imshow(img_slice, cmap="gray")
    overlay = np.zeros_like(seg_slice, dtype=np.float32)
    mask = seg_slice > 0
    overlay[mask] = seg_slice[mask]
    masked_overlay = np.ma.masked_where(overlay == 0, overlay)

    legend = _get_organ_legend(seg_slice, dataset_name)
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


def visualize_sample(
    sample,
    device: Optional[torch.device] = None,
    dataset_name: str = "CHAOS_MRI",
) -> None:
    """
    Visualize all slices of a volumetric image sample and its segmentation mask.
    """
    img, seg = sample["image"], sample["label"]
    num_slices = img.shape[-1]
    cols = min(4, num_slices)
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    # Ensure axes is always a 2D array for consistent indexing
    if rows != cols:
        axes = np.array(axes).reshape(1, -1)

    seg_max = seg.max()

    for z in range(num_slices):
        row = z // cols
        col = z % cols
        # Handle (H, W, D) or (C, H, W, D)
        if img.ndim == 3:
            img_slice = img[..., z]
            seg_slice = seg[..., z]
        elif img.ndim == 4:
            img_slice = img[0, ..., z]
            seg_slice = seg[0, ..., z]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        ax = axes[row, col]
        ax.imshow(img_slice, cmap="gray")
        overlay = np.zeros_like(seg_slice, dtype=np.float32)
        mask = seg_slice > 0
        overlay[mask] = seg_slice[mask]
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)

        cmap = cm.get_cmap("Set1").copy()
        cmap.set_bad(alpha=0)
        ax.imshow(masked_overlay, cmap=cmap, alpha=0.4, vmin=0, vmax=seg_max)
        ax.set_title(f"Slice {z}")
        ax.axis("off")

    if seg_max > 0:
        legend_labels, legend_colors = _get_organ_legend(seg, seg_max, dataset_name)
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
                for label, color in zip(legend_labels, legend_colors)
            ]
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=len(legend_labels),
                frameon=True,
                facecolor="white",
            )

    for z in range(num_slices, rows * cols):
        row = z // cols
        col = z % cols
        ax = axes[row, col]
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_3d(
    sample,
    dataset_name: str,
    device: Optional[torch.device] = None,
    rotate: int = 0,
    flip_axis: int = -1,  # -1 for no flip, 0 for vertical, 1 for horizontal
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
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rotate for correct orientation
    img = np.rot90(img, k=rotate, axes=(0, 1))
    seg = np.rot90(seg, k=rotate, axes=(0, 1))

    if flip_axis != -1:
        img = np.flip(img, axis=flip_axis + 1)
        seg = np.flip(seg, axis=flip_axis + 1)

    # Prepare legend info
    seg_max = seg.max()
    legend = _get_organ_legend(seg, seg_max, dataset_name)
    # Prepare label color mapping for napari
    # Create a matplotlib ListedColormap for segmentation labels

    unique_labels = np.unique(seg)
    label_color_dict = {}
    for idx, label in enumerate(unique_labels):
        if label == 0:
            label_color_dict[int(label)] = (0, 0, 0, 0)  # transparent for background
        else:
            # Use Set1 qualitative colormap with 8 colors (exclude the last color)
            set1 = cm.get_cmap("Set1", 9)
            n_colors = 8  # Exclude the last color
            color_idx = (idx - 1) % n_colors  # idx-1 because label==0 is background
            color = set1(color_idx)
            # Convert RGBA from 0-1 to 0-255 for napari
            color = tuple(int(255 * c) for c in color)
            label_color_dict[int(label)] = color
    seg_cmap = DirectLabelColormap(color_dict=label_color_dict)

    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add image and segmentation layers
    viewer.add_image(img, name="Image", colormap="gray", blending="additive")
    viewer.add_labels(seg, name="Segmentation", opacity=0.5, colormap=seg_cmap)

    # Start the Napari event loop
    napari.run()
