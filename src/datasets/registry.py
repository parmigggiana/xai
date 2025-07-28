import inspect
import sys
from pathlib import Path

from src.datasets.chaos import CHAOS
from src.datasets.common import BaseDataset
from src.datasets.mmwhs import MMWHS

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


def get_dataset(
    dataset_name, base_path, preprocess=None, batch_size=4, num_workers=0, **kwargs
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
