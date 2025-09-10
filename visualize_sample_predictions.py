"""
Visualize a single sample per dataset-domain combination with the same preprocessing as local.ipynb.
Shows image, ground-truth, and predictions from:
- baseline checkpoint
- finetuned checkpoint

Easily extendable: register more model variants in MODEL_VARIANTS to visualize additional predictions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from monai import transforms

from src.datasets.common import BaseDataset

# Local imports
from src.datasets.registry import get_dataset
from src.task_vector import TaskVector

# -------------------------
# Config mirroring local.ipynb
# -------------------------
# Resolve paths relative to this file so running from any CWD works
ROOT_DIR = Path(__file__).resolve().parent
DATASET_NAMES = ["CHAOS", "MMWHS"]
DOMAINS = ["CT", "MR"]
DATA_PATH = ROOT_DIR / "data"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints"
print(f"Data path: {DATA_PATH}")
print(f"Checkpoint path: {CHECKPOINT_PATH}")
USE_3D = False
BATCH_SIZE = 16
SPATIAL_SIZE = 128
NUM_WORKERS = 0
ALPHA_TV = 1

# Preprocessing stats as in the notebook
NORM_STATS = {
    ("MMWHS", "MR"): (186.5875, 258.5917),
    ("MMWHS", "CT"): (-745.0086, 1042.7251),
    ("CHAOS", "MR"): (90.8292, 168.8922),
    ("CHAOS", "CT"): (-478.1732, 476.7163),
}


def get_decode_func(dataset_name: str, domain: str):
    from src.datasets.mmwhs import mmwhs_labels

    decode = None
    if dataset_name == "CHAOS":
        if domain in ["MR", "MRI"]:

            def decode(labels):
                return labels // 63

        elif domain == "CT":

            def decode(labels):
                return torch.where(labels > 0, 1.0, 0.0)

    elif dataset_name == "MMWHS":

        def decode(labels):
            decoded_labels = torch.zeros_like(labels, dtype=torch.float32)
            for i, label_val in enumerate(mmwhs_labels.keys()):
                decoded_labels[labels == label_val] = i
            return decoded_labels

    if decode is None:

        def decode(labels):
            return labels

    return decode


def get_preprocessing(dataset_name: str, domain: str, is_training=True):
    decode_func = get_decode_func(dataset_name, domain)
    mean_std = NORM_STATS.get((dataset_name, domain))
    mean, std = mean_std if mean_std is not None else (None, None)

    # Image-specific transforms
    if USE_3D:
        image_transforms = [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Orientation(axcodes="RAS"),
        ]
    else:
        image_transforms = [
            transforms.Lambda(lambda x: x.squeeze(-1)),
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
        ]

    # Resize early to reduce compute
    image_transforms.append(
        transforms.Resize(
            spatial_size=SPATIAL_SIZE,
            size_mode="longest",
            mode="area",
            anti_aliasing=True,
        )
    )

    # Convert to tensor and ensure float32 for stable CPU ops
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.EnsureType(dtype=torch.float32),
        ]
    )

    # Normalize (still in float32)
    if mean is not None and std is not None:
        image_transforms.append(
            transforms.NormalizeIntensity(
                subtrahend=float(mean),
                divisor=float(std),
                channel_wise=False,
            )
        )

    # Augmentations (training only) — run in float32 on CPU
    if is_training:
        image_transforms.extend(
            [
                transforms.RandGaussianNoise(prob=0.15, std=0.05),
                transforms.RandAdjustContrast(prob=0.15, gamma=(0.95, 1.05)),
            ]
        )

    # Repeat to 3 channels only at the end (2D only)
    if not USE_3D:
        image_transforms.append(transforms.RepeatChannel(repeats=3))

    image_transform = transforms.Compose(image_transforms)

    # Segmentation transforms
    if not USE_3D:
        seg_transforms = [
            transforms.Lambda(lambda x: x.squeeze(-1)),
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
        ]
    else:
        seg_transforms = [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Orientation(axcodes="RAS"),
        ]

    seg_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.EnsureType(dtype=torch.long),
            transforms.Lambda(
                lambda x: decode_func(x)
            ),  # decode after tensor conversion
            transforms.Resize(
                spatial_size=SPATIAL_SIZE, size_mode="longest", mode="nearest"
            ),
            # transforms.EnsureType(dtype=torch.float32),
        ]
    )

    seg_transform = transforms.Compose(seg_transforms)
    return image_transform, seg_transform


# -------------------------
# Model variants registry
# -------------------------
# Each entry maps a label to a function that receives (dataset, encoder_type) and returns a model with weights loaded.
# Start with baseline and finetuned variants; extend by appending more items.


def _checkpoint_for(dataset_name: str, domain: str, tag: str) -> Path:
    dim = "3d" if USE_3D else "2d"
    return CHECKPOINT_PATH / f"{dataset_name}_{domain}_{dim}_{tag}.pth"


def build_model_variant(label: str, dataset: BaseDataset, encoder_type: str):
    # label should be either "baseline" or "finetuned" to match filenames created in local.ipynb
    ckpt = _checkpoint_for(dataset.name, dataset.domain, label)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for {label}: {ckpt}")

    model = dataset.get_model(encoder_type=encoder_type)

    # Load encoder-only checkpoints saved in notebook (torch.save(model.encoder, ...))
    # We'll attach it back to the current model.
    enc = torch.load(
        ckpt,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )
    model.encoder = enc
    model.eval()
    model.freeze()
    return model


MODEL_VARIANTS = {
    "baseline": build_model_variant,
    "finetuned": build_model_variant,
}


# -------------------------
# Task vector utilities (Part 1 parity)
# -------------------------


def _tv_key(dataset_name: str, domain: str) -> str:
    return f"{dataset_name}_{domain}"


def _checkpoint_for(dataset_name: str, domain: str, tag: str) -> Path:
    dim = "3d" if USE_3D else "2d"
    return CHECKPOINT_PATH / f"{dataset_name}_{domain}_{dim}_{tag}.pth"


def build_all_task_vectors() -> (
    tuple[dict[str, TaskVector], dict[str, TaskVector], dict[str, TaskVector]]
):
    """Build per-(dataset,domain) task vectors and the Part 1 composites.

    Returns:
        (tvs, composite_by_dataset, composite_by_domain)
    """
    tvs: dict[str, TaskVector] = {}
    # individual task vectors from baseline/finetuned
    for dset in DATASET_NAMES:
        for dom in DOMAINS:
            base = _checkpoint_for(dset, dom, "baseline")
            fine = _checkpoint_for(dset, dom, "finetuned")
            if base.exists() and fine.exists():
                try:
                    tvs[_tv_key(dset, dom)] = TaskVector(base, fine)
                except Exception as e:
                    print(f"  ⚠️ Failed to build TV for {dset} {dom}: {e}")

    # composites like in Part 1 of local.ipynb
    composite_by_dataset: dict[str, TaskVector] = {}
    for dset in DATASET_NAMES:
        key_mr = _tv_key(dset, "MR")
        key_ct = _tv_key(dset, "CT")
        if key_mr in tvs and key_ct in tvs:
            composite_by_dataset[dset] = tvs[key_mr] + tvs[key_ct]

    composite_by_domain: dict[str, TaskVector] = {}
    # e.g., MR = CHAOS_MR + MMWHS_MR
    for dom in DOMAINS:
        k1 = _tv_key("CHAOS", dom)
        k2 = _tv_key("MMWHS", dom)
        if k1 in tvs and k2 in tvs:
            composite_by_domain[dom] = tvs[k1] + tvs[k2]

    return tvs, composite_by_dataset, composite_by_domain


def pick_one_sample(dataset: BaseDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick a random (image, label) pair from available loaders.

    Prefers val, then train, then test. Randomly selects both the batch and the index within the batch
    when possible (i.e., when loader has a defined length). Falls back to the first yielded batch otherwise.
    """
    loader = dataset.val_loader or dataset.train_loader or dataset.test_loader
    if loader is None:
        raise RuntimeError("No dataloaders available in dataset")

    # Try to pick a random batch if len(loader) is available
    try:
        num_batches = len(loader)  # may raise TypeError if undefined
    except Exception:
        num_batches = None

    if isinstance(num_batches, int) and num_batches > 0:
        target_batch = random.randrange(num_batches)
        for i, batch in enumerate(loader):
            if i == target_batch:
                break
    else:
        # Fallback: just take the first batch
        batch = next(iter(loader))

    img = batch.get("image")
    lab = batch.get("label")
    if img is None or lab is None:
        raise RuntimeError("Batch missing 'image' or 'label'")

    # Random index within the batch
    bsz = getattr(img, "shape", None)[0] if hasattr(img, "shape") else None
    if not isinstance(bsz, int) or bsz <= 0:
        idx = 0
    else:
        idx = random.randrange(bsz)

    return img[idx], lab[idx]


def to_numpy(x: torch.Tensor):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return x.numpy()


def visualize_triptych(
    image,
    label,
    preds_by_model: Dict[str, torch.Tensor],
    title: str,
    out_dir: Path,
    filename: str | None = None,
):
    """Create composite figure (image, GT, predictions) and save to file.

    Args:
        image: input image tensor
        label: ground-truth label tensor
        preds_by_model: mapping variant name -> prediction tensor
        title: figure title
        out_dir: directory to save image
        filename: override filename (without path). If None, derived from title.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 2 + len(preds_by_model)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    img_np = to_numpy(image)
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        img_np = img_np[0]

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("image")
    axes[0].axis("off")

    lab_np = to_numpy(label.squeeze(0))
    axes[1].imshow(lab_np, cmap="viridis")
    axes[1].set_title("ground truth")
    axes[1].axis("off")

    col = 2
    for name, pred in preds_by_model.items():
        p_np = to_numpy(pred.squeeze(0))
        axes[col].imshow(p_np, cmap="viridis")
        axes[col].set_title(name)
        axes[col].axis("off")
        col += 1

    fig.suptitle(title)
    fig.tight_layout()

    # Derive filename
    if filename is None:
        safe = (
            title.lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("—", "-")
            .replace("--", "-")
        )
        filename = f"{safe}.png"
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def run(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_type = "swin_unetr" if args.use_3d else "clipseg"

    # Build task vectors once (if checkpoints are available)
    _, comp_by_dataset, comp_by_domain = build_all_task_vectors()

    for dataset_name in args.datasets:
        for domain in args.domains:
            print(f"\n== {dataset_name} / {domain} ==")
            image_transform, seg_transform = get_preprocessing(
                dataset_name, domain, is_training=False
            )

            ds: BaseDataset = get_dataset(
                dataset_name=dataset_name,
                base_path=DATA_PATH,
                domain=domain,
                transform=image_transform,
                seg_transform=seg_transform,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                slice_2d=not USE_3D,
            )

            # pick a single sample (image, label) and move to device
            image, label = pick_one_sample(ds)
            if hasattr(image, "as_tensor"):
                image = image.as_tensor()
            if hasattr(label, "as_tensor"):
                label = label.as_tensor()
            image = image.to(device)
            label = label.to(device)

            # collect predictions from registered variants
            preds_by_model: Dict[str, torch.Tensor] = {}
            for variant_name, builder in MODEL_VARIANTS.items():
                try:
                    model = builder(variant_name, ds, encoder_type)
                    model = model.to(device)
                    with torch.inference_mode():
                        logits = model(image.unsqueeze(0))
                        pred = torch.argmax(logits, dim=1, keepdim=True)[0]
                    preds_by_model[variant_name] = pred
                except FileNotFoundError as e:
                    print(f"  Skipping {variant_name}: {e}")

            # Additional variants from Part 1 composite task vectors
            # 1) dataset composite: {dataset}: TV(dataset_MR) + TV(dataset_CT)
            comp_ds = comp_by_dataset.get(dataset_name)
            if comp_ds is not None:
                base_ckpt = _checkpoint_for(dataset_name, domain, "baseline")
                if base_ckpt.exists():
                    try:
                        enc_mod = comp_ds.apply_to(base_ckpt, scaling_coef=ALPHA_TV)
                        model = ds.get_model(encoder_type=encoder_type).to(device)
                        model.encoder = enc_mod
                        model.eval()
                        model.freeze()
                        with torch.inference_mode():
                            logits = model(image.unsqueeze(0))
                            pred = torch.argmax(logits, dim=1, keepdim=True)[0]
                        preds_by_model[f"tv_{dataset_name}"] = pred
                    except Exception as e:
                        print(f"  Skipping tv_{dataset_name}: {e}")

            # 2) domain composite: {domain}: TV(CHAOS_domain) + TV(MMWHS_domain)
            comp_dom = comp_by_domain.get(domain)
            if comp_dom is not None:
                base_ckpt = _checkpoint_for(dataset_name, domain, "baseline")
                if base_ckpt.exists():
                    try:
                        enc_mod = comp_dom.apply_to(base_ckpt, scaling_coef=ALPHA_TV)
                        model = ds.get_model(encoder_type=encoder_type).to(device)
                        model.encoder = enc_mod
                        model.eval()
                        model.freeze()
                        with torch.inference_mode():
                            logits = model(image.unsqueeze(0))
                            pred = torch.argmax(logits, dim=1, keepdim=True)[0]
                        preds_by_model[f"tv_{domain}"] = pred
                    except Exception as e:
                        print(f"  Skipping tv_{domain}: {e}")

            title = f"{dataset_name} / {domain} — {encoder_type}"
            # Build deterministic filename: dataset_domain_encoder.png
            file_stub = f"{dataset_name}_{domain}_{encoder_type}".lower()
            visualize_triptych(
                image,
                label,
                preds_by_model,
                title,
                out_dir=args.out_dir,
                filename=f"{file_stub}.png",
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize sample predictions from checkpoints"
    )
    p.add_argument(
        "--datasets", nargs="*", default=DATASET_NAMES, choices=DATASET_NAMES
    )
    p.add_argument("--domains", nargs="*", default=DOMAINS, choices=DOMAINS)
    p.add_argument(
        "--use-3d",
        action="store_true",
        dest="use_3d",
        help="Use 3D encoder (swin_unetr)",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="Disable dataset in-memory cache"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "figures",
        help="Directory to save visualization figures",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
