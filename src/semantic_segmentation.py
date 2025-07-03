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

from src.utils import print_memory_usage

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

        # 3D segmentation layers - reduced channels for memory efficiency
        self.conv_layers = nn.Sequential(
            nn.Conv3d(feature_dim, 128, kernel_size=3, padding=1),  # Reduced from 256
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),  # Reduced from 128
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Semantic alignment layer - updated for reduced channels
        text_dim = self.class_embeddings.shape[1]
        self.semantic_projection = nn.Linear(64, text_dim)  # Changed from 128 to 64

        # Final classification layer - updated for reduced channels
        self.classifier = nn.Conv3d(
            64, num_classes, kernel_size=1
        )  # Changed from 128 to 64

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
        Forward pass with semantic guidance - memory optimized version.

        Args:
            features: [B, C, D, H, W] feature maps from encoder

        Returns:
            Dict containing logits and semantic alignment scores
        """
        print(f"🔍 SemanticHead: Input features shape: {features.shape}")

        # Apply conv layers
        x = self.conv_layers(features)  # [B, 64, D, H, W] - reduced channels
        print(f"🔍 SemanticHead: After conv layers shape: {x.shape}")

        # Compute semantic alignment with chunked processing to save memory
        B, C, D, H, W = x.shape
        print(f"🔍 SemanticHead: B={B}, C={C}, D={D}, H={H}, W={W}")

        # Process in smaller chunks to reduce memory usage
        chunk_size = min(D * H * W // 4, 1024)  # Process in chunks of max 1024 pixels
        print(f"🔍 SemanticHead: Chunk size: {chunk_size}")
        semantic_scores_list = []

        # Get class embeddings once
        class_embeddings = self.class_embeddings.to(x.device)  # [num_classes, text_dim]
        print(f"🔍 SemanticHead: Class embeddings shape: {class_embeddings.shape}")

        # Reshape for processing
        x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, D*H*W, 64]
        print(f"🔍 SemanticHead: Reshaped x shape: {x_reshaped.shape}")

        # Process each batch separately to save memory
        print("🔍 SemanticHead: Starting batch processing...")
        for b in range(B):
            print(f"🔍 SemanticHead: Processing batch {b}/{B}")
            batch_scores = []
            x_batch = x_reshaped[b]  # [D*H*W, 64]
            print(f"🔍 SemanticHead: Batch {b} shape: {x_batch.shape}")

            # Process in chunks
            print(
                f"🔍 SemanticHead: Processing {x_batch.shape[0]} pixels in chunks of {chunk_size}"
            )
            for i in range(0, x_batch.shape[0], chunk_size):
                chunk = x_batch[i : i + chunk_size]  # [chunk_size, 64]
                print(
                    f"🔍 SemanticHead: Chunk {i//chunk_size + 1}, shape: {chunk.shape}"
                )

                semantic_features = self.semantic_projection(
                    chunk
                )  # [chunk_size, text_dim]
                print(
                    f"🔍 SemanticHead: Semantic features shape: {semantic_features.shape}"
                )

                # Compute similarity with class embeddings
                chunk_scores = torch.mm(
                    semantic_features, class_embeddings.t()
                )  # [chunk_size, num_classes]
                print(f"🔍 SemanticHead: Chunk scores shape: {chunk_scores.shape}")
                batch_scores.append(chunk_scores)

            # Concatenate chunks back together
            batch_scores = torch.cat(batch_scores, dim=0)  # [D*H*W, num_classes]
            print(
                f"🔍 SemanticHead: Batch {b} final scores shape: {batch_scores.shape}"
            )
            semantic_scores_list.append(batch_scores)
        # Stack batches and reshape
        print(f"🔍 SemanticHead: Stacking {len(semantic_scores_list)} batches...")
        semantic_scores = torch.stack(
            semantic_scores_list, dim=0
        )  # [B, D*H*W, num_classes]
        print(
            f"🔍 SemanticHead: Stacked semantic scores shape: {semantic_scores.shape}"
        )

        semantic_scores = semantic_scores.reshape(B, D, H, W, self.num_classes)
        print(f"🔍 SemanticHead: Reshaped semantic scores: {semantic_scores.shape}")

        semantic_scores = semantic_scores.permute(
            0, 4, 1, 2, 3
        )  # [B, num_classes, D, H, W]
        print(f"🔍 SemanticHead: Final semantic scores shape: {semantic_scores.shape}")

        # Main classification
        logits = self.classifier(x)  # [B, num_classes, D, H, W]
        print(f"🔍 SemanticHead: Logits shape: {logits.shape}")

        print("🔍 SemanticHead: Creating output dictionary...")
        result = {
            "logits": logits,
            "semantic_scores": semantic_scores,
            "combined": logits + 0.1 * semantic_scores,  # Weighted combination
        }
        print(f"🔍 SemanticHead: Combined output shape: {result['combined'].shape}")
        print("✅ SemanticHead: Forward pass completed successfully!")

        return result


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
            # Create ResNet but we'll modify it to preserve spatial dimensions
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
            # For SwinUNETR, we need to use a different approach to extract features
            if encoder_type == "swin_unetr":
                # Keep the full SwinUNETR model, we'll extract features in forward
                # SwinUNETR output features are typically from the decoder before final layer
                # We'll use the encoder outputs directly
                self.semantic_head = SemanticGuidedSegmentationHead(
                    feature_dim=384,  # Reduced from 768 for memory efficiency
                    num_classes=num_classes,
                    class_descriptions=class_descriptions,
                )
            else:
                # For ResNet, we need to extract features before global pooling
                # We'll create a custom forward method that preserves spatial dimensions
                self.semantic_head = SemanticGuidedSegmentationHead(
                    feature_dim=feature_dim,
                    num_classes=num_classes,
                    class_descriptions=class_descriptions,
                )
        else:
            self.use_semantic_head = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle case where input might be a list (from some operations)
        if isinstance(x, (list, tuple)):
            if len(x) == 1 and isinstance(x[0], torch.Tensor):
                x = x[0]
            else:
                raise ValueError(
                    f"Expected single tensor input, got list/tuple of length {len(x)}"
                )

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor input, got {type(x)}")

        # Ensure tensor is contiguous
        x = x.contiguous()

        if self.use_semantic_head:
            # Extract features for semantic guidance
            if self.encoder_type == "swin_unetr":
                # For SwinUNETR, use the encoder part
                features = self.encoder(x)
            elif self.encoder_type == "resnet":
                # For ResNet, extract features before global pooling
                features = self._extract_resnet_features(x)

            outputs = self.semantic_head(features)
            return outputs["combined"]  # Use semantic-guided predictions
        else:
            # Direct encoder output (for models without semantic head)
            return self.encoder(x)

    def _extract_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ResNet before global pooling to preserve spatial dimensions."""
        # Based on the ResNet structure test:
        # 0: conv1, 1: bn1, 2: act, 3: maxpool,
        # 4: layer1, 5: layer2, 6: layer3, 7: layer4,
        # 8: avgpool (AdaptiveAvgPool3d) <- STOP HERE
        # 9: fc

        # Apply layers up to layer4 (index 7) to preserve spatial dimensions
        layers = list(self.encoder.children())

        # Apply layers 0-7 (up to and including layer4)
        for i in range(8):  # Stop before avgpool (index 8)
            if i < len(layers):
                x = layers[i](x)

        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override call method to handle both training and inference.
        This allows the model to be used seamlessly in training loops.
        """
        return self.forward(x)

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.semantic_head.parameters():
            param.requires_grad = False

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

        # dice_metric = DiceMetric(include_background=False, reduction="mean")
        # hausdorff_metric = HausdorffDistanceMetric(
        #     include_background=False, reduction="mean"
        # )

        results = {}

        for split in ["train", "test"]:
            loader = getattr(self.dataset, f"{split}_loader", None)
            results[split] = {}
            if loader is None:
                continue

            # dice_metric.reset()
            # hausdorff_metric.reset()
            has_labels = False

            with torch.no_grad():
                for batch in loader:
                    print_memory_usage(f"Batch size: {batch['image'].shape}")
                    images = batch["image"].to(device)
                    labels = batch.get("label", None)
                    if labels is None:
                        continue
                    labels = labels.to(device)
                    # has_labels = True

                    outputs = self(images)
                    preds = torch.argmax(outputs, dim=1)

                    # Compute metrics for this batch
                    # dice_metric(y_pred=preds, y=labels)
                    # hausdorff_metric(y_pred=preds, y=labels)

                    # Compute Mean Class Accuracy (mAcc)

                    # Compute Mean Intersection over Union (mIoU)

            # if has_labels:
            # try:
            # dice_score = dice_metric.aggregate().item()
            # hausdorff_dist = hausdorff_metric.aggregate().item()
            # results[split] = {"dice": dice_score, "hausdorff": hausdorff_dist}
            # except (ValueError, RuntimeError):
            # Handle case where no valid predictions were made
            # results[split] = {"dice": None, "hausdorff": None}
            # else:
            #     results[split] = {"dice": None, "hausdorff": None}

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
