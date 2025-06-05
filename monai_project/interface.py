"""
Interface module for evaluating volumetric segmentation performance using MONAI.
"""

from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import os


def evaluate_segmentation_performance(
    dataset_name: str,
    dataloader: DataLoader,
    base_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a segmentation model on 3D medical imaging data.

    Args:
        dataset_name (str): Name of the dataset (for logging).
        dataloader (DataLoader): PyTorch DataLoader providing batches of {'image': Tensor, 'label': Tensor}.
        base_model (torch.nn.Module, optional): Model to evaluate. Defaults to pretrained Swin-UNETR.
        device (torch.device, optional): PyTorch device. Defaults to CUDA if available.
        save_path (str, optional): Path to save/load the classification head.

    Returns:
        Dict[str, float]: Dictionary with metrics: Dice, Hausdorff Distance.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize default model
    if base_model is None:
        # Pretrained Swin-UNETR
        base_model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            drop_rate=0.0,
        )
        # TODO: load pretrained weights if available
    # Use base_model as encoder only and freeze its weights
    encoder = base_model.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Prepare classification head: load or save from disk if save_path is given
    num_channels = getattr(base_model, "out_channels", 2)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        head_file = os.path.join(save_path, f"{dataset_name}_head.pth")
        if os.path.exists(head_file):
            head = torch.nn.Conv3d(num_channels, num_channels, kernel_size=1)
            head.load_state_dict(torch.load(head_file, map_location=device))
            head = head.to(device)
        else:
            head = torch.nn.Conv3d(num_channels, num_channels, kernel_size=1).to(device)
            torch.save(head.state_dict(), head_file)
    else:
        head = torch.nn.Conv3d(num_channels, num_channels, kernel_size=1).to(device)
    classification_head = head

    # Build full model: encoder + classification head
    model = torch.nn.Sequential(encoder, classification_head).to(device)
    model.eval()

    # Define metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

    # Inference and metric computation
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = sliding_window_inference(
                inputs=images, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
            )

            # Apply argmax to get discrete segmentation
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            # Compute metrics
            dice_metric(y_pred=preds, y=labels)
            hausdorff_metric(y_pred=preds, y=labels)

    # Aggregate metrics
    dice_score = dice_metric.aggregate().item()
    hausdorff_dist = hausdorff_metric.aggregate().item()

    # Reset metrics for potential use
    dice_metric.reset()
    hausdorff_metric.reset()

    return {
        "dataset": dataset_name,
        "dice": dice_score,
        "hausdorff_distance": hausdorff_dist,
    }


def visualize_sample_slice(
    dataloader: DataLoader,
    sample_index: int = 0,
    device: Optional[torch.device] = None,
) -> None:
    """
    Visualize a volumetric image sample and its segmentation mask (center slice).

    Args:
        dataloader (DataLoader): yields batches with 'image' and 'label'.
        sample_index (int): index of the batch to visualize.
        device (torch.device, optional): computation device.
    """
    import matplotlib.pyplot as plt

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch the specified sample
    # print(dataloader)
    for idx, batch in enumerate(dataloader):
        # print(f"Processing batch {idx + 1}/{len(dataloader)}")
        if idx == sample_index:
            img = batch["image"][0].to(device).cpu().numpy()
            seg = batch["label"][0].to(device).cpu().numpy()
            break

    # Select center slice along last axis (depth dimension)
    z = img.shape[-1] // 2
    # print(img.shape, seg.shape, z)

    img_slice = img[0, ..., z]  # Remove channel dimension
    seg_slice = seg[0, ..., z]  # Remove channel dimension

    # Plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_slice, cmap="gray")
    ax1.set_title("Image Slice")
    ax1.axis("off")

    ax2.imshow(img_slice, cmap="gray")
    ax2.imshow(seg_slice, cmap="jet", alpha=0.5, vmin=0, vmax=seg_slice.max())
    ax2.set_title("Overlay Segmentation")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_sample(
    dataloader: DataLoader,
    sample_index: int = 0,
    device: Optional[torch.device] = None,
) -> None:
    """
    Visualize all slices of a volumetric image sample and its segmentation mask.

    Args:
        dataloader (DataLoader): yields batches with 'image' and 'label'.
        sample_index (int): index of the batch to visualize.
        device (torch.device, optional): computation device.
    """
    import matplotlib.pyplot as plt

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch the specified sample
    for idx, batch in enumerate(dataloader):
        # print(f"Processing batch {idx + 1}/{len(dataloader)}")
        if idx == sample_index:
            img = batch["image"][0].to(device).cpu().numpy()
            seg = batch["label"][0].to(device).cpu().numpy()
            break

    # print(f"Image shape: {img.shape}, Segmentation shape: {seg.shape}")

    # Get number of slices
    num_slices = img.shape[-1]

    # Calculate grid size for subplots
    cols = min(4, num_slices)  # Max 4 columns
    rows = (num_slices + cols - 1) // cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Handle case where there's only one row
    if rows == 1:
        axes = axes.reshape(1, -1) if num_slices > 1 else [axes]

    # Plot all slices
    for z in range(num_slices):
        row = z // cols
        col = z % cols

        img_slice = img[0, ..., z]  # Remove channel dimension
        seg_slice = seg[0, ..., z]  # Remove channel dimension

        ax = axes[row, col] if rows > 1 else axes[col]

        # Show image with segmentation overlay
        ax.imshow(img_slice, cmap="gray")
        ax.imshow(seg_slice, cmap="jet", alpha=0.5, vmin=0, vmax=seg_slice.max())
        ax.set_title(f"Slice {z}")
        ax.axis("off")

    # Hide unused subplots
    for z in range(num_slices, rows * cols):
        row = z // cols
        col = z % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_3d(
    dataloader: DataLoader,
    sample_index: int = 0,
    device: Optional[torch.device] = None,
):
    """
    Visualize a 3D volumetric image sample and its segmentation mask using 3D rendering.

    Args:
        dataloader (DataLoader): yields batches with 'image' and 'label'.
        sample_index (int): index of the batch to visualize.
        device (torch.device, optional): computation device.
    """
    import napari

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch the specified sample
    for idx, batch in enumerate(dataloader):
        if idx == sample_index:
            img = batch["image"][0].to(device).cpu().numpy()
            seg = batch["label"][0].to(device).cpu().numpy()
            break

    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add image and segmentation layers
    viewer.add_image(img, name="Image", colormap="gray", blending="additive")
    viewer.add_labels(seg, name="Segmentation", opacity=0.5)

    # Start the Napari event loop
    napari.run()
