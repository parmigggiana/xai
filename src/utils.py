from pathlib import Path

import torch
import torch.nn
from torch.utils.data import Dataset

from src.dataset import CHAOSDataset, MMWHSDataset
from src.modeling import ClassificationHead
import re
import zipfile
import torch
from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet
import urllib.request
import urllib.parse
from torchvision.models import ResNet


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
    if Path(save_path).exists():
        # Load existing classification head
        classification_head = torch.load(save_path)
    else:
        # Create new classification head
        classification_head = torch.nn.Linear(
            512, num_classes
        )  # Assuming 512 input features

        # Save the classification head
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
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
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    return model


def get_dataset(
    dataset_name: str, domain: str, is_train: bool = True, base_path="data/"
) -> Dataset:
    """
    Get the dataset path for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        is_train (bool): Whether to get the training or validation dataset path.

    Returns:
        str: Path to the dataset.
    """
    dataset_path = Path(base_path) / dataset_name
    split = "train" if is_train else "test"
    dataset_path.mkdir(parents=True, exist_ok=True)
    download_and_extract_dataset(dataset_name, base_path)
    match dataset_name:
        case "CHAOS":
            return CHAOSDataset(base_path=dataset_path, domain=domain, split=split)
        case "MM-WHS":
            raise MMWHSDataset(base_path=dataset_path, domain=domain, split=split)
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")


def download_and_extract_dataset(dataset: str, base_path: str = "data/"):
    index_url = f"https://xai.balzov.com/{dataset}/"
    base_path = Path(base_path) / dataset
    with urllib.request.urlopen(index_url) as response:
        html = response.read().decode("utf-8")
        zip_files = re.findall(r'href="([^"]+\.zip)"', html)
    for zip_file in zip_files:
        # Decode URL encoding in file names
        decoded_zip_file = urllib.parse.unquote(zip_file)
        zip_path = base_path / decoded_zip_file
        extract_dir = base_path / Path(decoded_zip_file).with_suffix("")
        if not zip_path.exists():
            zip_url = index_url + zip_file
            print(f"Downloading {zip_url} to {zip_path}...")

            def reporthook(
                block_num, block_size, total_size, zip_file=decoded_zip_file
            ):
                downloaded = block_num * block_size
                percent = (
                    min(100, downloaded * 100 / total_size) if total_size > 0 else 0
                )
                print(
                    f"\rDownloading {zip_file}: {percent:.2f}% ({downloaded // (1024 * 1024)}MB/{total_size // (1024 * 1024)}MB)",
                    end="",
                )

            urllib.request.urlretrieve(zip_url, zip_path, reporthook)
            print()  # Newline after download
        # Unzip if not already extracted
        if not extract_dir.exists():
            print(f"Extracting {zip_path} to {extract_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)


def get_classification_head(
    dataset_name: str, domain: str, save_path: str
) -> ClassificationHead:
    """
    Get the classification head for a given dataset.

    Args:
        dataset_name (str): Name of the dataset.
        num_classes (int): Number of classes for classification.

    Returns:
        torch.nn.Module: Classification head.
    """
    # raise NotImplementedError(
    #     f"Classification head for dataset {dataset_name} is not implemented."
    # )
    match (dataset_name.upper(), domain.upper()):
        case ("CHAOS", "CT"):
            out_channels = 2
        case ("CHAOS", "MR"):
            out_channels = 5
        case _:
            raise ValueError(
                f"Unsupported dataset {dataset_name} or domain {domain} for classification head."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get in_channels from ResNet output channels (usually 512 for ResNet)
    in_channels = 512

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        head_file = Path(save_path) / f"{dataset_name}_head.pth"

    if head_file and head_file.exists():
        head = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)
        head.load_state_dict(torch.load(head_file, map_location=device))
        head = head.to(device)
    else:
        head = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1).to(device)
        # Initialize weights and biases
        torch.nn.init.xavier_uniform_(head.weight)
        if head.bias is not None:
            torch.nn.init.zeros_(head.bias)
        # TODO
        torch.save(head.state_dict(), head_file)

    return head


def get_baseline_encoder(base_path: str = "", depth: int = 10) -> torch.nn.Module:
    """
    Get the baseline encoder for the model.

    Args:
        base_path (str): Base path to load the encoder from.

    Returns:
        torch.nn.Module: Baseline encoder.
    """

    encoder_path = Path(base_path) / f"baseline_{depth}.pth"
    encoder = ResNet(depth)

    if encoder_path.exists():
        state_dict = torch.load(encoder_path, map_location="cpu")
        encoder.load_state_dict(state_dict)
    else:
        state_dict = get_pretrained_resnet_medicalnet(depth)
        encoder.load_state_dict(state_dict)
        Path(base_path).mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), encoder_path)
    return encoder
