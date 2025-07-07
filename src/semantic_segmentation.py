"""
Hybrid approach: 3D medical segmentation with semantic guidance.
This combines the best of both worlds:
1. 3D medical pretrained encoders (preserve spatial context)
2. Semantic-guided head training (leverage text descriptions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR, ResNet
from typing import Dict, List, OrderedDict, Tuple
import os

from monai.apps import download_url

import numpy as np
from tqdm import tqdm

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

        # Add upsampling decoder to restore original spatial dimensions
        self.upsampling_decoder = nn.Sequential(
            # Upsample by factor of 2 in each dimension
            nn.ConvTranspose3d(num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # Upsample by factor of 2 again
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # Upsample by factor of 2 again
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # Final layer to get correct number of classes
            nn.Conv3d(16, num_classes, kernel_size=3, padding=1),
        )

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
        # Apply conv layers
        x = self.conv_layers(features)  # [B, 64, D, H, W] - reduced channels

        # Compute semantic alignment with chunked processing to save memory
        B, C, D, H, W = x.shape

        # Process in smaller chunks to reduce memory usage
        chunk_size = min(D * H * W // 4, 1024)  # Process in chunks of max 1024 pixels
        semantic_scores_list = []

        # Get class embeddings once
        class_embeddings = self.class_embeddings.to(x.device)  # [num_classes, text_dim]

        # Reshape for processing
        x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, D*H*W, 64]

        # Process each batch separately to save memory
        for b in range(B):
            batch_scores = []
            x_batch = x_reshaped[b]  # [D*H*W, 64]

            # Process in chunks
            for i in range(0, x_batch.shape[0], chunk_size):
                chunk = x_batch[i : i + chunk_size]  # [chunk_size, 64]

                semantic_features = self.semantic_projection(
                    chunk
                )  # [chunk_size, text_dim]

                # Compute similarity with class embeddings
                chunk_scores = torch.mm(
                    semantic_features, class_embeddings.t()
                )  # [chunk_size, num_classes]
                batch_scores.append(chunk_scores)

            # Concatenate chunks back together
            batch_scores = torch.cat(batch_scores, dim=0)  # [D*H*W, num_classes]
            semantic_scores_list.append(batch_scores)

        # Stack batches and reshape
        semantic_scores = torch.stack(
            semantic_scores_list, dim=0
        )  # [B, D*H*W, num_classes]

        semantic_scores = semantic_scores.reshape(B, D, H, W, self.num_classes)

        semantic_scores = semantic_scores.permute(
            0, 4, 1, 2, 3
        )  # [B, num_classes, D, H, W]

        # Main classification
        logits = self.classifier(x)  # [B, num_classes, D, H, W]

        # Apply upsampling to restore original spatial dimensions
        upsampled_logits = self.upsampling_decoder(logits)

        # Upsample semantic scores to match logits dimensions
        upsampled_semantic_scores = nn.functional.interpolate(
            semantic_scores,
            size=upsampled_logits.shape[2:],  # Target spatial dimensions [D, H, W]
            mode="trilinear",
            align_corners=False,
        )

        # Combine semantic and convolutional predictions
        combined = 0.7 * upsampled_semantic_scores + 0.3 * upsampled_logits

        return {
            "semantic": upsampled_semantic_scores,
            "logits": upsampled_logits,
            "combined": combined,
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
            feature_dim = 768  # Adjusted: feature_dim = feature_size * 16

            # Load pretrained SwinViT weights if available
            if pretrained:
                self._load_swinvit_weights()

        elif encoder_type == "resnet":
            # Create ResNet but we'll modify it to preserve spatial dimensions
            self.encoder = ResNet(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=num_classes,
                block="basic",
                layers=[2, 2, 2, 2],  # ResNet-18 style (smaller)
                block_inplanes=[64, 128, 256, 512],
            )
            feature_dim = 512
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Semantic-guided head (if descriptions provided)
        if class_descriptions:
            self.use_semantic_head = True
            self.semantic_head = SemanticGuidedSegmentationHead(
                feature_dim=feature_dim,
                num_classes=num_classes,
                class_descriptions=class_descriptions,
            )
        else:
            self.use_semantic_head = False

    def _load_swinvit_weights(self):
        """Load pretrained SwinViT weights from data/model_swinvit.pt"""
        try:
            resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
            dst = "./data/ssl_pretrained_weights.pth"
            download_url(resource, dst)
            pretrained_path = os.path.normpath(dst)
            ssl_dict = torch.load(pretrained_path, weights_only=True)
            ssl_weights = ssl_dict["model"]

            # Generate new state dict so it can be loaded to MONAI SwinUNETR Model
            monai_loadable_state_dict = OrderedDict()
            model_prior_dict = self.encoder.state_dict()
            model_update_dict = model_prior_dict

            del ssl_weights["encoder.mask_token"]
            del ssl_weights["encoder.norm.weight"]
            del ssl_weights["encoder.norm.bias"]
            del ssl_weights["out.conv.conv.weight"]
            del ssl_weights["out.conv.conv.bias"]

            for key, value in ssl_weights.items():
                if key[:8] == "encoder.":
                    if key[8:19] == "patch_embed":
                        new_key = "swinViT." + key[8:]
                    else:
                        new_key = "swinViT." + key[8:18] + key[20:]
                    monai_loadable_state_dict[new_key] = value
                else:
                    monai_loadable_state_dict[key] = value

            model_update_dict.update(monai_loadable_state_dict)
            self.encoder.load_state_dict(model_update_dict, strict=True)
            model_final_loaded_dict = self.encoder.state_dict()

            # Safeguard test to ensure that weights got loaded successfully
            layer_counter = 0
            for k, _v in model_final_loaded_dict.items():
                if k in model_prior_dict:
                    layer_counter = layer_counter + 1

                    old_wts = model_prior_dict[k]
                    new_wts = model_final_loaded_dict[k]

                    old_wts = old_wts.to("cpu").numpy()
                    new_wts = new_wts.to("cpu").numpy()
                    diff = np.mean(np.abs(old_wts, new_wts))
                    print("Layer {}, the update difference is: {}".format(k, diff))
                    if diff == 0.0:
                        print("Warning: No difference found for layer {}".format(k))
            print(
                "Total updated layers {} / {}".format(
                    layer_counter, len(model_prior_dict)
                )
            )
            print("Pretrained Weights Succesfully Loaded !")

        except Exception as e:
            print(f"Error loading SwinViT weights: {e}")

    def _pad_input_for_swin_unetr(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pads the input tensor's depth to be divisible by 32 for SwinUNETR."""
        original_depth = x.shape[2]
        if self.encoder_type == "swin_unetr":
            depth = x.shape[2]
            if depth % 32 != 0:
                pad_depth = 32 - (depth % 32)
                padding = (0, 0, 0, 0, 0, pad_depth)
                x = F.pad(x, padding, "constant", 0)
        return x, original_depth

    def _crop_output_to_original_size(
        self, result: torch.Tensor, original_depth: int
    ) -> torch.Tensor:
        """Crops the output tensor back to the original depth if it was padded."""
        if self.encoder_type == "swin_unetr" and result.shape[2] != original_depth:
            result = result[:, :, :original_depth, :, :]
        return result

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

        x, original_depth = self._pad_input_for_swin_unetr(x)
        print(f"Input shape after padding: {x.shape}")
        if self.use_semantic_head:
            # Extract features for semantic guidance
            if self.encoder_type == "swin_unetr":
                features = self.encoder.swinViT(x)[-1]

            elif self.encoder_type == "resnet":
                features = self._extract_resnet_features(x)

            outputs = self.semantic_head(features)
            result = outputs["combined"]  # Use semantic-guided predictions
        else:
            # Direct encoder output (for models without semantic head)
            result = self.encoder(x)

        result = self._crop_output_to_original_size(result, original_depth)

        return result

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
        if self.use_semantic_head:
            for param in self.semantic_head.parameters():
                param.requires_grad = False

    def finetune(self, epochs: int = 5):
        """Finetune the model on the dataset."""
        # This would implement the training loop
        # For now, just save the current state as "finetuned"
        pass

    def load_task_vector(self, task_vector):
        """Load a task vector into the model."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in task_vector.vector:
                    param.data += task_vector.vector[name]

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
            results[split] = {}
            if loader is None:
                continue

            dice_metric.reset()
            hausdorff_metric.reset()
            has_labels = False

            with torch.no_grad():
                for batch in tqdm(loader):
                    images = batch["image"].to(device)
                    labels = batch.get("label", None)
                    if labels is None:
                        continue
                    labels = labels.to(device)
                    has_labels = True

                    outputs = self(images)
                    preds = torch.argmax(
                        outputs, dim=1, keepdim=True
                    )  # [B, 1, D, H, W]

                    # Compute metrics for this batch
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
