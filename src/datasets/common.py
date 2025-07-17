import collections
import glob
import os
import random

import napari
import numpy as np
import torch

# import torch.nn as nn
import torchvision.datasets as datasets
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

# from src.modeling_seg import ImageEncoder, ImageSegmenter  # , SegmentationHead


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f"Flipping labels with probability {self.flip_label_prob}")
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes - 1)
                    self.samples[i] = (self.samples[i][0], new_label)

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {"images": image, "labels": label, "image_paths": self.samples[index][0]}


class BaseDataset:
    def __init__(self):
        self.name = type(self).__name__

    def _get_organ_legend(self, seg_slice):

        print(f"Warning: No specific legend for dataset {type(self)}.")
        set1 = cm.get_cmap("Set1", 8)  # Set1 is qualitative, 8 distinct colors
        legend = {}
        unique_labels = np.unique(seg_slice)
        unique_labels = unique_labels[unique_labels > 0]
        for idx, label in enumerate(unique_labels):
            legend[label] = set1(idx % set1.N)
        return legend

    # def build_segmentation_head(
    #     self,
    #     feature_dim,
    #     dataset_name,
    #     data_location,
    #     device,
    #     num_classes=None,
    #     classnames=None,
    # ):
    #     """Build a segmentation head for volumetric or 2D segmentation tasks."""
    #     if num_classes is None:
    #         if classnames is None:
    #             # Import here to avoid circular imports
    #             from src.datasets.registry import get_dataset

    #             dataset = get_dataset(dataset_name, None, location=data_location)
    #             num_classes = len(dataset.classnames)
    #         else:
    #             num_classes = len(classnames)

    #     if getattr(self, "slice_2d", False):
    #         # 2D segmentation head
    #         segmentation_head = nn.Sequential(
    #             nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 128, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(128),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(128, num_classes, kernel_size=1),
    #         ).to(device)
    #     else:
    #         # 3D segmentation head
    #         segmentation_head = nn.Sequential(
    #             nn.Conv3d(feature_dim, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm3d(256),
    #             nn.ReLU(inplace=True),
    #             nn.Conv3d(256, 128, kernel_size=3, padding=1),
    #             nn.BatchNorm3d(128),
    #             nn.ReLU(inplace=True),
    #             nn.Conv3d(128, num_classes, kernel_size=1),
    #         ).to(device)

    #     return segmentation_head

    # def get_segmentation_head(self, save_path, cache_dir=None):
    #     filename = os.path.join(
    #         save_path,
    #         f"head_{self.name}_{self.domain}_{'2d' if self.slice_2d else '3d'}.pt",
    #     )
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     if self.slice_2d:
    #         base_encoder = "ViT-B-32"
    #     else:
    #         base_encoder = "RN50"

    #     feature_dim = ImageEncoder(
    #         model=base_encoder, keep_lang=True, cache_dir=cache_dir
    #     ).model.visual.output_dim

    #     if os.path.exists(filename):
    #         print(
    #             f"Segmentation head for {self.name} {self.domain} {'2D' if self.slice_2d else '3D'} exists at {filename}"
    #         )
    #         # Build the segmentation head and load the state dict
    #         segmentation_head = self.build_segmentation_head(
    #             feature_dim,
    #             self.name,
    #             data_location="data/",
    #             device=device,
    #             num_classes=self.num_classes,
    #         )
    #         segmentation_head.load_state_dict(torch.load(filename, map_location=device))
    #         return segmentation_head

    #     print(
    #         f"Did not find segmentation head for {self.name} {self.domain} {'2D' if self.slice_2d else '3D'} at {filename}, building one from scratch."
    #     )

    #     segmentation_head = self.build_segmentation_head(
    #         feature_dim,
    #         self.name,
    #         data_location="data/",
    #         device=device,
    #         num_classes=self.num_classes,
    #     )

    #     os.makedirs(save_path, exist_ok=True)
    #     torch.save(segmentation_head.state_dict(), filename)
    #     return segmentation_head

    # def get_model(self, save_path):
    #     # Build a ImageSegmenter
    #     image_encoder = ImageEncoder(
    #         model="ViT-B-32", keep_lang=False, cache_dir=".cache/"
    #     )
    #     segmentation_head = self.get_segmentation_head(save_path=save_path)
    #     model = ImageSegmenter(
    #         image_encoder=image_encoder,
    #         segmentation_head=segmentation_head,
    #         dataset=self,
    #     )
    #     return model

    def get_hybrid_model(self, encoder_type="swin_unetr", use_semantic_head=True):
        """
        Return a hybrid Medical3DSegmenter with semantic guidance.
        This is an alternative to get_model that uses the semantic_segmentation approach.

        Args:
            encoder_type (str): Type of encoder to use ('swin_unetr' or 'resnet')
            use_semantic_head (bool): Whether to use semantic guidance

        Returns:
            Medical3DSegmenter: Model with semantic guidance capabilities
        """
        from src.semantic_segmentation import Medical3DSegmenter

        # Get class descriptions if available, otherwise use generic ones
        # Only pass class_descriptions if use_semantic_head is True
        if use_semantic_head:
            class_descriptions = getattr(
                self, "classnames", [f"Class {i}" for i in range(self.num_classes)]
            )
        else:
            class_descriptions = None

        model = Medical3DSegmenter(
            encoder_type=encoder_type,
            num_classes=self.num_classes,
            class_descriptions=class_descriptions,
            pretrained=True,
            dataset=self,
        )
        return model

    def visualize_3d(self, sample):
        """
        Visualize a 3D volumetric image sample and its segmentation mask using 3D rendering.

        Args:
            sample (dict): contains 'image' and 'label'.
        """
        self._visualize_3d(sample)

    @torch.no_grad()
    def _visualize_3d(
        self,
        sample,
        rotate: int = 0,
        flip_axis: int = None,
    ):
        """
        Visualize a 3D volumetric image sample and its segmentation mask using 3D rendering.

        Args:
            dataloader (DataLoader): yields batches with 'image' and 'label'.
            sample_index (int): index of the batch to visualize.
            device (torch.device, optional): computation device.
            dataset_name (str): name of the dataset for legend labeling.
        """

        img, seg = sample["image"], sample["label"]
        # print(img.shape)
        if img.ndim < 3:
            return

        # Rotate for correct orientation
        img = np.rot90(img, k=rotate, axes=(0, 1))
        seg = np.rot90(seg, k=rotate, axes=(0, 1))

        if flip_axis is not None:
            if isinstance(flip_axis, int):
                flip_axis = (flip_axis,)
            for axis in flip_axis:
                img = np.flip(img, axis=axis)
                seg = np.flip(seg, axis=axis)

        # Create a Napari viewer
        viewer = napari.Viewer()

        # Scale z-axis to make layers appear taller in 3D view
        z_scale = 1.0
        if self.domain in ["MR", "MRI"]:
            z_scale = 5.0  # Increase this value to make layers taller

        scale = (1.0, 1.0, z_scale) if img.ndim == 3 else (1.0, 1.0, 1.0, z_scale)

        # Add image and segmentation layers with scaling
        viewer.add_image(
            img, name="Image", colormap="gray", blending="additive", scale=scale
        )
        viewer.add_labels(seg, name="Segmentation", opacity=0.5, scale=scale)

        # Start the Napari event loop
        napari.run()

    def visualize_sample_slice(
        self,
        sample,
    ):
        """
        Visualize a volumetric image sample and its segmentation mask (center slice).
        """
        self._visualize_sample_slice(sample)

    @torch.no_grad()
    def _visualize_sample_slice(
        self,
        sample,
        rotate: int = 0,
        flip_axis: int = None,
    ) -> None:
        """
        Visualize a volumetric image sample and its segmentation mask (center slice).
        """
        import matplotlib.pyplot as plt

        img, seg = sample["image"], sample["label"]
        # Handle (H, W, D) or (C, H, W, D)
        if img.ndim == 3:
            z = img.shape[-1] // 2
            img_slice = img[..., z]
            seg_slice = seg[..., z]
        elif img.ndim == 4:
            z = img.shape[-1] // 2
            img_slice = img[0, ..., z]
            seg_slice = seg[0, ..., z]
        elif img.ndim == 2:
            img_slice = img
            seg_slice = seg
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        img_slice = np.rot90(img_slice, k=rotate)
        seg_slice = np.rot90(seg_slice, k=rotate)

        if flip_axis is not None:
            if isinstance(flip_axis, int):
                flip_axis = (flip_axis,)
            print(f"Flipping along axes: {flip_axis}")
            for axis in flip_axis:
                print(f"Flipping along axis {axis}")
                img_slice = np.flip(img_slice, axis=axis)
                seg_slice = np.flip(seg_slice, axis=axis)

        ax1.imshow(img_slice, cmap="gray")
        ax1.set_title("Image Slice")
        ax1.axis("off")

        ax2.imshow(img_slice, cmap="gray")
        overlay = np.zeros_like(seg_slice, dtype=np.float32)
        mask = seg_slice > 0
        overlay[mask] = seg_slice[mask]
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)

        legend = self._get_organ_legend(seg_slice)
        legend_colors = ListedColormap([legend[label] for label in legend])
        legend_labels = list(legend.keys())
        ax2.imshow(masked_overlay, cmap=legend_colors, alpha=0.4)
        if legend_labels:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=label,
                )
                for label, color in legend.items()
            ]
            ax2.legend(
                handles=legend_elements,
                loc="upper right",
                frameon=True,
                facecolor="white",
            )
        ax2.set_title("Overlay Segmentation")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {"images": batch[0], "labels": batch[1]}
    elif len(batch) == 3:
        batch = {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(
        image_encoder, device_ids=[x for x in range(torch.cuda.device_count())]
    )
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch["images"].cuda())

            all_data["features"].append(features.cpu())

            for key, val in batch.items():
                if key == "images":
                    continue
                if hasattr(val, "cpu"):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = "train" if is_train else "val"
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f"{image_encoder.cache_dir}/{dname}/{split}"
        cached_files = glob.glob(f"{cache_dir}/*")
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f"Getting features from {cache_dir}")
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f"Did not find cached features at {cache_dir}. Building from scratch.")
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print("Not caching because no cache directory was passed.")
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching data at {cache_dir}")
            for name, val in data.items():
                torch.save(val, f"{cache_dir}/{name}.pt")
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data["features"])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data["features"] = torch.from_numpy(data["features"]).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(
            feature_dataset, batch_size=args.batch_size, shuffle=is_train, num_workers=1
        )
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader
