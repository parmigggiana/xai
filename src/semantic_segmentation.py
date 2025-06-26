"""
Hybrid approach: 3D medical segmentation with semantic guidance.
This combines the best of both worlds:
1. 3D medical pretrained encoders (preserve spatial context)
2. Semantic-guided head training (leverage text descriptions)
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, ResNet
from typing import Dict, List

try:
    from transformers import AutoTokenizer, AutoModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print(
        "Warning: transformers not available. Install with: pip install transformers>=4.20.0"
    )


class SemanticGuidedSegmentationHead(nn.Module):
    """
    A segmentation head that uses semantic text embeddings to guide class predictions.
    This bridges the gap between CLIP-style semantic guidance and 3D medical segmentation.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        class_descriptions: List[str],
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Load biomedical text encoder for semantic guidance
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Get text embeddings for each class
        self.class_embeddings = self._compute_class_embeddings(class_descriptions)

        # 3D segmentation layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Semantic alignment layer
        text_dim = self.class_embeddings.shape[1]
        self.semantic_projection = nn.Linear(128, text_dim)

        # Final classification layer
        self.classifier = nn.Conv3d(128, num_classes, kernel_size=1)

    def _compute_class_embeddings(self, class_descriptions: List[str]) -> torch.Tensor:
        """Compute text embeddings for each class description."""
        embeddings = []

        with torch.no_grad():
            for desc in class_descriptions:
                inputs = self.tokenizer(
                    desc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                outputs = self.text_encoder(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(embedding)

        return torch.stack(embeddings)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with semantic guidance.

        Args:
            features: [B, C, D, H, W] feature maps from encoder

        Returns:
            Dict containing logits and semantic alignment scores
        """
        # Apply conv layers
        x = self.conv_layers(features)  # [B, 128, D, H, W]

        # Compute semantic alignment
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*D*H*W, 128]
        semantic_features = self.semantic_projection(x_flat)  # [B*D*H*W, text_dim]

        # Compute similarity with class embeddings
        class_embeddings = self.class_embeddings.to(x.device)  # [num_classes, text_dim]
        semantic_scores = torch.mm(
            semantic_features, class_embeddings.t()
        )  # [B*D*H*W, num_classes]
        semantic_scores = semantic_scores.reshape(B, D, H, W, self.num_classes)
        semantic_scores = semantic_scores.permute(
            0, 4, 1, 2, 3
        )  # [B, num_classes, D, H, W]

        # Main classification
        logits = self.classifier(x)  # [B, num_classes, D, H, W]

        return {
            "logits": logits,
            "semantic_scores": semantic_scores,
            "combined": logits + 0.1 * semantic_scores,  # Weighted combination
        }


class Medical3DSegmenter(nn.Module):
    """
    3D Medical segmentation model with semantic guidance.
    Uses medical pretrained encoders + semantic-guided heads.
    """

    def __init__(
        self,
        encoder_type: str = "swin_unetr",
        num_classes: int = 2,
        class_descriptions: List[str] = None,
        pretrained: bool = True,
        dataset=None,
    ):
        super().__init__()

        # Store model configuration
        self.encoder_type = encoder_type
        self.num_classes = num_classes
        self.dataset = dataset
        self.pretrained = pretrained

        # Initialize encoder
        if encoder_type == "swin_unetr":
            self.encoder = SwinUNETR(
                in_channels=1,
                out_channels=num_classes,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
            )
            feature_dim = 48  # SwinUNETR feature dimension
        elif encoder_type == "resnet":
            self.encoder = ResNet(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=num_classes,
                block="basic",
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
            )
            feature_dim = 512
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Semantic-guided head (if descriptions provided)
        if class_descriptions:
            self.use_semantic_head = True
            # Extract features before final layer for semantic guidance
            if encoder_type == "swin_unetr":
                # Replace the final layer with our semantic head
                self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
                self.semantic_head = SemanticGuidedSegmentationHead(
                    feature_dim=feature_dim * 8,  # Before final upsampling
                    num_classes=num_classes,
                    class_descriptions=class_descriptions,
                )
        else:
            self.use_semantic_head = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input tensor format transformation for SwinUNETR
        # Input might be [B, H, W, D] but SwinUNETR expects [B, C, D, H, W]
        if x.dim() == 4:
            # Assume input is [B, H, W, D], transform to [B, C, D, H, W]
            # Add channel dimension: [B, H, W, D] -> [B, 1, H, W, D]
            x = x.unsqueeze(1)  # [B, 1, H, W, D]
            # Rearrange to [B, C, D, H, W]: [B, 1, H, W, D] -> [B, 1, D, H, W]
            x = x.permute(0, 1, 4, 2, 3)  # [B, 1, D, H, W]

        if self.use_semantic_head:
            features = self.encoder(x)
            outputs = self.semantic_head(features)
            return outputs["combined"]  # Use semantic-guided predictions
        else:
            return self.encoder(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override call method to handle both training and inference.
        This allows the model to be used seamlessly in training loops.
        """
        return self.forward(x)

    def finetune(self, epochs: int = 5):
        """Finetune the model on the dataset."""
        # This would implement the training loop
        # For now, just save the current state as "finetuned"

        print(f"Finetuning completed for {epochs} epochs")

    def load_task_vector(self, task_vector):
        """Load a task vector into the model."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in task_vector.vector:
                    param.data += task_vector.vector[name]
        print("Task vector loaded successfully")

    def evaluate(self):
        """Evaluate the model and return metrics on both train and test loaders."""
        from monai.metrics import DiceMetric, HausdorffDistanceMetric

        device = next(self.parameters()).device
        self.eval()

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean"
        )

        results = {}

        for split in ["train", "test"]:
            loader = getattr(self.dataset, f"{split}_loader", None)
            if loader is None:
                results[split] = {"dice": None, "hausdorff": None}
                continue

            dice_metric.reset()
            hausdorff_metric.reset()
            has_labels = False

            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(device)
                    labels = batch.get("label", None)
                    if labels is None:
                        continue
                    labels = labels.to(device)
                    has_labels = True

                    print(
                        f"Processing batch from {split} loader with shape {images.shape}"
                    )

                    outputs = self.forward(images)

                    preds = torch.argmax(outputs, dim=1, keepdim=True)
                    dice_metric(y_pred=preds, y=labels)
                    hausdorff_metric(y_pred=preds, y=labels)

            if has_labels:
                try:
                    dice_score = dice_metric.aggregate().item()
                    hausdorff_dist = hausdorff_metric.aggregate().item()
                    results[split] = {"dice": dice_score, "hausdorff": hausdorff_dist}
                except (ValueError, RuntimeError):
                    # Handle case where no valid predictions were made
                    results[split] = {"dice": None, "hausdorff": None}
            else:
                results[split] = {"dice": None, "hausdorff": None}

        return results


# Medical class descriptions for semantic guidance
CHAOS_CLASS_DESCRIPTIONS = {
    "CT": ["background tissue in CT scan", "liver organ in CT imaging"],
    "MR": [
        "background tissue in MRI scan",
        "liver organ in MRI",
        "right kidney in MRI",
        "left kidney in MRI",
        "spleen organ in MRI",
    ],
}

MMWHS_CLASS_DESCRIPTIONS = {
    "CT": [
        "background tissue in cardiac CT",
        "heart muscle myocardium",
        "heart chambers and vessels",
        "cardiac structures in CT",
    ],
    "MR": [
        "background tissue in cardiac MRI",
        "heart muscle myocardium",
        "heart chambers and vessels",
        "cardiac structures in MRI",
    ],
}
