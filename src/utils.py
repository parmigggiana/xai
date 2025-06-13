import os
import pickle
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset

from src.dataset import CHAOSDataset, MMWHSDataset


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


######### Added #########


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
            return MMWHSDataset(base_path=dataset_path, domain=domain, split=split)
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
