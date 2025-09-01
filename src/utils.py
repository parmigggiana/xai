import os
import pickle
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn


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


def download_and_extract_dataset(dataset: str, base_path: str = "data/"):
    index_url = f"https://xai.balzov.com/{dataset}/"
    base_path = Path(base_path) / dataset
    Path(base_path).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(index_url) as response:
        html = response.read().decode("utf-8")
        zip_files = re.findall(r'href="([^"]+\.zip)"', html)
    for zip_file in zip_files:
        # Decode URL encoding in file names
        decoded_zip_file = urllib.parse.unquote(zip_file)
        zip_path = base_path / decoded_zip_file
        extract_dir = base_path / Path(decoded_zip_file).with_suffix("")
        if not extract_dir.exists():
            if not zip_path.exists():
                zip_url = index_url + zip_file
                print(f"Downloading {zip_url} to {zip_path}...")

                # Only use progress reporting when running interactively
                if sys.stdout.isatty():

                    def reporthook(
                        block_num, block_size, total_size, zip_file=decoded_zip_file
                    ):
                        downloaded = block_num * block_size
                        percent = (
                            min(100, downloaded * 100 / total_size)
                            if total_size > 0
                            else 0
                        )
                        print(
                            f"\rDownloading {zip_file}: {percent:.2f}% ({downloaded // (1024 * 1024)}MB/{total_size // (1024 * 1024)}MB)",
                            end="",
                        )

                    urllib.request.urlretrieve(zip_url, zip_path, reporthook)
                    print()  # Newline after download
                else:
                    urllib.request.urlretrieve(zip_url, zip_path)
                    print(f"Download completed: {decoded_zip_file}")

            print(f"Extracting {zip_path} to {extract_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            # Remove the zip file after extraction
            zip_path.unlink(missing_ok=True)


def print_memory_usage(stage=""):
    """Print current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
        print(
            f"{stage} - RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB (cached: {gpu_cached:.1f}MB)"
        )
    else:
        print(f"{stage} - RAM: {memory_mb:.1f}MB")


# MONAI-native dict-based transform constructors
def build_monai_dict_transforms(
    dataset_name: str, domain: str, spatial_size: int, is_training: bool, use_3d: bool
):
    from monai import transforms as T
    import torch

    # Decode func per dataset
    def get_decode():
        from src.datasets.mmwhs import mmwhs_labels

        if dataset_name.upper() == "CHAOS":
            if domain.upper() in ("MR", "MRI"):
                return lambda labels: labels // 63
            return lambda labels: torch.where(labels > 0, 1.0, 0.0)
        if dataset_name.upper() == "MMWHS":

            def decode(labels):
                decoded = torch.zeros_like(labels, dtype=torch.float32)
                for i, val in enumerate(mmwhs_labels.keys()):
                    decoded[labels == val] = i
                return decoded

            return decode
        return lambda x: x

    keys_img = ["image"]
    keys_lbl = ["label"]
    decode = get_decode()

    t: list = []
    # loaders
    # Leave readers to MONAI auto-dispatch based on file extension; users can override if needed
    # Channels/orientation
    if use_3d:
        t += [
            T.LoadImaged(keys=keys_img + keys_lbl),
            T.EnsureChannelFirstd(keys=keys_img + keys_lbl, channel_dim="no_channel"),
            T.Orientationd(keys=keys_img + keys_lbl, axcodes="RAS"),
        ]
    else:
        t += [
            T.LoadImaged(keys=keys_img + keys_lbl),
            T.Lambdad(
                keys=keys_img + keys_lbl,
                func=lambda x: (
                    x.squeeze(-1) if hasattr(x, "shape") and x.ndim >= 3 else x
                ),
            ),
            T.EnsureChannelFirstd(keys=keys_img + keys_lbl, channel_dim="no_channel"),
        ]

    # Intensity
    if domain.upper() == "CT":
        t += [
            T.ScaleIntensityRanged(
                keys=keys_img, a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            )
        ]
    else:
        t += [T.NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True)]

    if is_training:
        t += [
            T.RandGaussianNoised(keys=keys_img, prob=0.2, std=0.05),
            T.RandAdjustContrastd(keys=keys_img, prob=0.2, gamma=(0.9, 1.1)),
        ]

    # Label decode
    t += [T.Lambdad(keys=keys_lbl, func=decode)]

    # Resize + type
    t += [
        T.Resized(
            keys=keys_img, spatial_size=spatial_size, size_mode="longest", mode="area"
        ),
        T.Resized(
            keys=keys_lbl,
            spatial_size=spatial_size,
            size_mode="longest",
            mode="nearest",
        ),
        T.EnsureTyped(keys=keys_img, dtype=torch.float32, track_meta=True),
        T.EnsureTyped(keys=keys_lbl, dtype=torch.float32, track_meta=False),
    ]

    return T.Compose(t)


# Collate helpers
def meta_safe_collate(batch):
    """Collate that converts MetaTensor -> Tensor before stacking, dropping metadata.

    Supports batches of tuples like (image, label) or single tensors.
    """
    from torch.utils.data._utils.collate import default_collate

    def to_tensor(x):
        try:
            return x.as_tensor() if hasattr(x, "as_tensor") else x
        except Exception:
            return x

    if isinstance(batch[0], tuple):
        # transpose list of tuples into tuple of lists
        columns = list(zip(*batch))
        collated = []
        for col in columns:
            items = [to_tensor(x) for x in col]
            collated.append(default_collate(items))
        return tuple(collated) if len(collated) > 1 else collated[0]
    elif isinstance(batch[0], dict):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [to_tensor(sample[k]) for sample in batch]
            out[k] = default_collate(vals)
        return out
    else:
        items = [to_tensor(x) for x in batch]
        return default_collate(items)


################################### Testing Sampler ####################################

import random

import numpy as np
from torch.utils.data import Sampler


class DiversitySampler(Sampler):
    """Custom sampler that ensures diverse samples within each batch"""

    def __init__(self, dataset, batch_size, seed=42):
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # If dataset has metadata we can use for diversity assessment
        self.has_metadata = hasattr(dataset, "__getitem__") and hasattr(
            dataset[0], "__len__"
        )
        if self.has_metadata and len(dataset[0]) >= 3:
            print("DiversitySampler: Using metadata for enhanced diversity")

    def __iter__(self):
        # Create shuffled indices
        indices = list(range(self.dataset_size))
        self.rng.shuffle(indices)

        # Organize indices to maximize batch diversity
        num_batches = self.dataset_size // self.batch_size

        # Strategy: interleave samples rather than taking consecutive ones
        reorganized = []
        for i in range(self.batch_size):
            for j in range(num_batches):
                idx = j * self.batch_size + i
                if idx < self.dataset_size:
                    reorganized.append(indices[idx])

        # Add any remaining indices
        remaining = self.dataset_size - len(reorganized)
        if remaining > 0:
            reorganized.extend(indices[-remaining:])

        return iter(reorganized)

    def __len__(self):
        return self.dataset_size
