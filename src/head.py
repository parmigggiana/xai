"""
Adapted from https://github.com/mlfoundations/task_vectors/ by Ilalrco et al.
"""

import os

import open_clip
import torch
from tqdm import tqdm

from src.datasets.registry import get_dataset
from src.datasets.templates import get_templates
from src.modeling import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, template, data_location, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(dataset_name, None, location=data_location)
    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(args, dataset):
    filename = os.path.join(args.save, f"head_{dataset}.pt")
    if os.path.exists(filename):
        print(f"Classification head for {args.model} on {dataset} exists at {filename}")
        return ClassificationHead.load(filename)
    print(
        f"Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch."
    )
    model = ImageEncoder(args, keep_lang=True).model
    template = get_templates(dataset)
    classification_head = build_classification_head(
        model, dataset, template, args.data_location, args.device
    )
    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head


def get_segmentation_head(dataset, domain, slice_2d, save_path, cache_dir):
    filename = os.path.join(
        save_path, f"head_{dataset}_{domain}_{'2d' if slice_2d else '3d'}.pt"
    )
    if os.path.exists(filename):
        print(
            f"Segmentation head for {dataset} {domain} {'2D' if slice_2d else '3D'} exists at {filename}"
        )
        return ClassificationHead.load(filename)
    print(
        f"Did not find segmentation head for {dataset} {domain} {'2D' if slice_2d else '3D'} at {filename}, building one from scratch."
    )

    if slice_2d:
        base_encoder = "ViT-B-32"
    else:
        base_encoder = "RN50"

    model = ImageEncoder(model=base_encoder, keep_lang=True, cache_dir=cache_dir).model
    template = get_templates(dataset)
    segmentation_head = build_segmentation_head()

    os.makedirs(save_path, exist_ok=True)
    segmentation_head.save(filename)
    return segmentation_head
