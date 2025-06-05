import os
import torch

import torch.nn
from torch.utils.data import Dataset

from monai_project.dataset import CHAOSMRIDataset, CHAOSCTDataset


def get_classification_head(
    dataset_name: str, num_classes: int, save_path: str
) -> torch.nn.Module:
    """
    Get the classification head for a given dataset.

    Args:
        dataset_name (str): Name of the dataset.
        num_classes (int): Number of classes for classification.

    Returns:
        torch.nn.Module: Classification head.
    """

    # Check if classification head exists at save_path
    if os.path.exists(save_path):
        # Load existing classification head
        classification_head = torch.load(save_path)
    else:
        # Create new classification head
        classification_head = torch.nn.Linear(
            512, num_classes
        )  # Assuming 512 input features

        # Save the classification head
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(classification_head, save_path)

    return classification_head


def finetune(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    save_path: str,
    device: torch.device = None,
):
    """
    Finetune a model on a given dataset.

    Args:
        model (torch.nn.Module): Model to finetune.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        save_path (str): Path to save the finetuned model.
        device (torch.device, optional): Device to use for training. Defaults to CUDA if available.

    Returns:
        torch.nn.Module: Finetuned model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = (
        torch.nn.CrossEntropyLoss()
    )  # For pixel-wise classification in semantic segmentation

    patience = 3
    best_loss = float("inf")
    patience_counter = 0

    best_model_state = None
    epoch = 0

    while True:
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            batch_loss = loss(output, target)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        epoch += 1

    # Load best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the finetuned model checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    return model


def get_dataset(dataset_name: str, is_train: bool = True) -> Dataset:
    """
    Get the dataset path for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        is_train (bool): Whether to get the training or validation dataset path.

    Returns:
        str: Path to the dataset.
    """
    match dataset_name:
        case "CHAOS_MRI":
            base_path = "data/CHAOS"
            split = "Train_Sets" if is_train else "Test_Sets"
            return CHAOSMRIDataset(base_path=base_path, split=split)
        case "CHAOS_CT":
            base_path = "data/CHAOS"
            split = "Train_Sets" if is_train else "Test_Sets"
            return CHAOSCTDataset(base_path=base_path, split=split)
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")
