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
            drop_rate=0.0
        )
        # TODO: load pretrained weights if available
    # Use base_model as encoder only and freeze its weights
    encoder = base_model.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Prepare classification head: load or save from disk if save_path is given
    num_channels = getattr(base_model, 'out_channels', 2)
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
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model
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


def visualize_sample(
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
    for idx, batch in enumerate(dataloader):
        if idx == sample_index:
            img = batch['image'][0].to(device).cpu().numpy()
            seg = batch['label'][0].to(device).cpu().numpy()
            break

    # Select center slice along first axis
    z = img.shape[0] // 2
    img_slice = img[z, ...]
    seg_slice = seg[z, ...]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_slice, cmap='gray')
    ax1.set_title('Image Slice')
    ax1.axis('off')

    ax2.imshow(img_slice, cmap='gray')
    ax2.imshow(seg_slice, cmap='jet', alpha=0.5)
    ax2.set_title('Overlay Segmentation')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

