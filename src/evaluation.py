import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
from torch.utils.data import DataLoader

from src.old_modeling import Classifier


def evaluate_segmentation_performance(
    dataloader: DataLoader,
    model: Classifier,
    device: torch.device = None,
) -> tuple[float, float]:
    """
    Evaluate a segmentation model on 3D medical imaging data.

    Args:
        dataloader (DataLoader): PyTorch DataLoader providing batches of {'image': Tensor, 'label': Tensor}.
        base_model (torch.nn.Module, optional): Model to evaluate. Defaults to pretrained Swin-UNETR.
        device (torch.device, optional): PyTorch device. Defaults to CUDA if available.

    Returns:
        tuple[float, float]: Dice score and Hausdorff distance for the segmentation.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.freeze()

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

    return dice_score, hausdorff_dist
