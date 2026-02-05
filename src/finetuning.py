from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Tuple, Union

import torch

from src.datasets.common import BaseDataset
from src.datasets.registry import get_dataset
from src.utils import download_and_extract_dataset, get_preprocessing


@dataclass(frozen=True)
class FinetuneConfig:
    dataset_names: Iterable[str]
    domains: Iterable[str]
    training_epochs: Mapping[Tuple[str, str], int]
    data_path: Union[str, Path] = "data/"
    checkpoint_path: Union[str, Path] = "checkpoints/"
    outputs_path: Union[str, Path] = "outputs/"
    use_3d: bool = True
    batch_size: int = 4
    spatial_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    num_workers: int = 0
    debug: bool = False
    memory_trace: bool = False
    profile: Union[bool, str] = False
    encoder_type: Optional[str] = None


def update_metrics(
    outputs_path: Union[str, Path], name: str, new_metrics: dict
) -> None:
    outputs_path = Path(outputs_path)
    metrics_file = outputs_path / "metrics.json"

    if not metrics_file.exists():
        metrics = {}
    else:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

    metrics[name] = new_metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


def _setup_tracemalloc(memory_trace: bool) -> Optional[Callable[[str], None]]:
    if not memory_trace:
        return None

    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start()

    def print_memory_snapshot(label: str = "") -> None:
        if not tracemalloc.is_tracing():
            print("⚠️ tracemalloc not started")
            return

        current, peak = tracemalloc.get_traced_memory()
        print(
            f"\n=== Memory Snapshot: {label} ===\n"
            f"Current: {current / 1024 / 1024:.1f} MB | Peak: {peak / 1024 / 1024:.1f} MB"
        )

    return print_memory_snapshot


def _checkpoint_stem(dataset_name: str, domain: str, use_3d: bool) -> str:
    return f"{dataset_name}_{domain}_{'3d' if use_3d else '2d'}"


def _finetuned_checkpoint_path(
    checkpoint_path: Path, dataset_name: str, domain: str, use_3d: bool
) -> Path:
    return (
        checkpoint_path
        / f"{_checkpoint_stem(dataset_name, domain, use_3d)}_finetuned.pth"
    )


def _baseline_checkpoint_path(
    checkpoint_path: Path, dataset_name: str, domain: str, use_3d: bool
) -> Path:
    return (
        checkpoint_path
        / f"{_checkpoint_stem(dataset_name, domain, use_3d)}_baseline.pth"
    )


def _freeze_encoder_for_3d(model) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.out.parameters():
        p.requires_grad = True
    for p in model.encoder.decoder1.parameters():
        p.requires_grad = True


def _finetune_one(
    *,
    config: FinetuneConfig,
    dataset_name: str,
    domain: str,
    encoder_type: str,
    profile_dir: Path,
    data_path: Path,
    checkpoint_path: Path,
    outputs_path: Path,
    memory_snapshot_fn: Optional[Callable[[str], None]],
) -> None:
    if (dataset_name, domain) not in config.training_epochs:
        print(
            f"⚠️ No training epochs specified for {dataset_name} in {domain} domain. Skipping."
        )
        return

    epochs = config.training_epochs[(dataset_name, domain)]
    download_and_extract_dataset(dataset_name, str(data_path))

    image_transform, seg_transform = get_preprocessing(
        dataset_name,
        domain,
        is_training=True,
        track_memory=config.memory_trace,
        use_3d=config.use_3d,
        spatial_size=config.spatial_size,
        debug=config.debug,
        memory_trace=config.memory_trace,
        memory_snapshot_fn=memory_snapshot_fn,
    )

    finetuned_checkpoint = _finetuned_checkpoint_path(
        checkpoint_path, dataset_name, domain, config.use_3d
    )
    if finetuned_checkpoint.exists():
        print(
            f"Finetuned model for {dataset_name} in {domain} domain with {'3d' if config.use_3d else '2d'} images already exists at {finetuned_checkpoint}. Skipping finetuning."
        )
        return

    print(
        f"Finetuning on {dataset_name} dataset in {domain} domain with {'3d' if config.use_3d else '2d'} images "
    )

    dataset: BaseDataset = get_dataset(
        dataset_name=dataset_name,
        domain=domain,
        transform=image_transform,
        seg_transform=seg_transform,
        base_path=data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        slice_2d=not config.use_3d,
    )

    if not isinstance(dataset, BaseDataset):
        raise TypeError(
            f"Expected dataset to be an instance of BaseDataset, got {type(dataset)}"
        )

    model = dataset.get_model(base_model=encoder_type)

    baseline_checkpoint = _baseline_checkpoint_path(
        checkpoint_path, dataset_name, domain, config.use_3d
    )
    torch.save(model.encoder, baseline_checkpoint)

    if config.use_3d:
        _freeze_encoder_for_3d(model)

    model.finetune(
        epochs=epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        debug=config.debug,
        profile=config.profile,
        profile_dir=str(profile_dir),
    )

    torch.save(model.encoder, finetuned_checkpoint)

    model_metrics = model.evaluate(profile=config.profile, profile_dir=str(profile_dir))
    update_metrics(
        outputs_path,
        f"{dataset_name}_{domain}_{'3d' if config.use_3d else '2d'}_finetuned",
        model_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_finetuning(config: FinetuneConfig) -> None:
    """Run finetuning across dataset/domain combinations.

    This encapsulates what used to live in finetune.py: download, preprocessing,
    dataset/model creation, finetuning, checkpointing, evaluation, metrics logging.
    """

    checkpoint_path = Path(config.checkpoint_path)
    outputs_path = Path(config.outputs_path)
    data_path = Path(config.data_path)

    profile_dir = outputs_path / "profiling"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    encoder_type = config.encoder_type
    if encoder_type is None:
        encoder_type = "swin_unetr" if config.use_3d else "clipseg"

    memory_snapshot_fn = _setup_tracemalloc(config.memory_trace)

    for dataset_name in config.dataset_names:
        for domain in config.domains:
            _finetune_one(
                config=config,
                dataset_name=dataset_name,
                domain=domain,
                encoder_type=encoder_type,
                profile_dir=profile_dir,
                data_path=data_path,
                checkpoint_path=checkpoint_path,
                outputs_path=outputs_path,
                memory_snapshot_fn=memory_snapshot_fn,
            )
