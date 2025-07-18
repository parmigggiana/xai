import copy
import inspect
import sys
from pathlib import Path

import torch
from torch.utils.data.dataset import random_split

from src.datasets.chaos import CHAOS
from src.datasets.common import BaseDataset
from src.datasets.mmwhs import MMWHS

# from src.datasets.apis import APIS
# from src.datasets.oasis import OASIS

registry = {
    name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(obj, BaseDataset) and obj is not BaseDataset
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(
    dataset,
    new_dataset_class_name,
    batch_size,
    num_workers,
    val_fraction,
    max_val_samples=None,
    seed=0,
):
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(
    dataset_name, base_path, preprocess=None, batch_size=128, num_workers=16, **kwargs
) -> BaseDataset:
    assert (
        dataset_name in registry
    ), f"Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}"
    dataset_class = registry[dataset_name]
    if isinstance(base_path, str):
        base_path = Path(base_path)
    location = base_path / dataset_name
    dataset = dataset_class(
        preprocess=preprocess,
        location=location,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs,
    )
    return dataset
