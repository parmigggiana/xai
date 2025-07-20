"""
Hybrid approach: 3D medical segmentation with semantic guidance.
This combines the best of both worlds:
1. 3D medical pretrained encoders (preserve spatial context)
2. Semantic-guided head training (leverage text descriptions)
"""

import os
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.apps import download_url
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.networks.nets.resnet import resnet50
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer

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

        # 3D segmentation layers with increased middle layer sizes (deeper and wider network)
        self.conv_layers = nn.Sequential(
            nn.Conv3d(feature_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
        )

        # Semantic alignment layer - updated for increased channels
        text_dim = self.class_embeddings.shape[1]
        self.semantic_projection = nn.Linear(1024, text_dim)

        # Final classification layer - updated for increased channels
        self.classifier = nn.Conv3d(1024, num_classes, kernel_size=1)

        # Reduced upsampling decoder (only one upsampling step)
        # self.upsampling_decoder = nn.Sequential(
        #     nn.ConvTranspose3d(num_classes, 32, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(32, num_classes, kernel_size=3, padding=1),
        # )
        self.upsampling_decoder = nn.Identity()

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

    def train_classification_head(
        self,
        classnames: List[str],
        templates: List[str],
        device: torch.device = None,
        logit_scale: float = 1.0,
    ):
        """
        Build classification head weights using text embeddings similar to CLIP zero-shot classification.

        Args:
            classnames: List of class names to build embeddings for
            templates: List of template strings with {} placeholder for class names
            device: Device to run computation on
            logit_scale: Scale factor for logits (similar to CLIP's logit_scale)

        Returns:
            Updated class_embeddings tensor that can be used for classification
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.text_encoder.eval()

        print("Building classification head for semantic segmentation.")
        with torch.no_grad():
            zeroshot_weights = []

            for classname in tqdm(classnames):
                texts = []
                # Apply each template to the classname
                for template in templates:
                    if "{}" in template:
                        texts.append(template.format(classname))
                    else:
                        # If no placeholder, just append template + classname
                        texts.append(f"{template} {classname}")

                # Tokenize all text variations for this class
                text_embeddings = []
                for text in texts:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = self.text_encoder(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()

                    # Normalize embedding
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    text_embeddings.append(embedding)

                # Average embeddings across templates for this class
                class_embedding = torch.stack(text_embeddings).mean(dim=0, keepdim=True)
                # Normalize again after averaging
                class_embedding = class_embedding / class_embedding.norm()

                zeroshot_weights.append(class_embedding)

            # Stack all class embeddings
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(
                device
            )  # [num_classes, 1, text_dim]
            zeroshot_weights = zeroshot_weights.squeeze(1)  # [num_classes, text_dim]

            # Apply logit scaling (similar to CLIP)
            if (
                abs(logit_scale - 1.0) > 1e-8
            ):  # Use tolerance for floating point comparison
                zeroshot_weights = zeroshot_weights * logit_scale

            # Transpose to match expected format [text_dim, num_classes] for matrix multiplication
            zeroshot_weights = zeroshot_weights.t()  # [text_dim, num_classes]

        # Update the class embeddings
        self.class_embeddings = zeroshot_weights.t()  # Store as [num_classes, text_dim]

        print(f"Classification head built with shape: {self.class_embeddings.shape}")
        return self.class_embeddings

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize encoder
        if encoder_type == "swin_unetr":
            self.encoder = SwinUNETR(
                # img_size=(256, 256, 256),  # Fixed input size for SwinUNETR
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
            self.encoder = resnet50(
                pretrained=pretrained,
                n_input_channels=1,
                feed_forward=False,
                shortcut_type="B",
                bias_downsample=False,
            )
            feature_dim = 2048
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

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        self.device = device
        self.encoder.to(device)
        if self.use_semantic_head:
            self.semantic_head.to(device)
        return self

    def _load_swinvit_weights(self):
        """Load pretrained SwinViT weights from data/model_swinvit.pt"""
        try:
            resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
            dst = "./data/ssl_pretrained_weights.pth"
            download_url(resource, dst)
            pretrained_path = os.path.normpath(dst)
            ssl_dict = torch.load(
                pretrained_path, weights_only=True, map_location=self.device
            )
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
                if key.startswith("encoder."):
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
                    # print("Layer {}, the update difference is: {}".format(k, diff))
                    if abs(diff) < 1e-8:  # Use tolerance for floating point comparison
                        print("Warning: No difference found for layer {}".format(k))
            print(
                "Total updated layers {} / {}".format(
                    layer_counter, len(model_prior_dict)
                )
            )
            print("Pretrained Weights Succesfully Loaded !")

        except Exception as e:
            print(f"Error loading SwinViT weights: {e}")

    def _preprocess_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Preprocess input by resampling to 256x256 while keeping depth the same.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Tuple of (preprocessed_tensor, original_spatial_size)
        """
        original_size = x.shape[2:]  # (D, H, W)

        # Only resample H and W dimensions to 256x256, keep D unchanged
        target_size = (original_size[0], 256, 256)  # (D, 256, 256)

        if original_size[1:] != (
            256,
            256,
        ):  # Only resample if H,W are not already 256x256
            print(f"Resampling from {original_size} to {target_size}")
            x = F.interpolate(
                x, size=target_size, mode="trilinear", align_corners=False
            )

        return x, original_size

    def _postprocess_output(
        self, result: torch.Tensor, original_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Postprocess output by resampling back to original spatial dimensions.

        Args:
            result: Model output tensor [B, C, D, H, W]
            original_size: Original spatial dimensions (D, H, W)

        Returns:
            Tensor resampled to original spatial dimensions
        """
        current_size = result.shape[2:]  # (D, H, W)

        if current_size != original_size:
            print(f"Resampling output from {current_size} to {original_size}")
            result = F.interpolate(
                result, size=original_size, mode="trilinear", align_corners=False
            )

        return result

    def _pad_input_for_swin_unetr(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pads the input tensor's depth to be divisible by 32 for SwinUNETR."""
        original_shape = (x.shape[2], x.shape[3], x.shape[4])
        if self.encoder_type == "swin_unetr":
            depth, height, width = x.shape[2], x.shape[3], x.shape[4]
            pad_depth = (32 - depth % 32) if depth % 32 != 0 else 0
            pad_height = (32 - height % 32) if height % 32 != 0 else 0
            pad_width = (32 - width % 32) if width % 32 != 0 else 0
            # F.pad uses (W_left, W_right, H_left, H_right, D_left, D_right)
            padding = (0, pad_width, 0, pad_height, 0, pad_depth)
            if pad_depth > 0 or pad_height > 0 or pad_width > 0:
                x = F.pad(x, padding, "constant", 0)
        return x, original_shape

    def _crop_output_to_original_size(
        self, result: torch.Tensor, original_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Crops the output tensor back to the original shape if it was padded."""
        if self.encoder_type == "swin_unetr":
            depth, height, width = original_shape
            result = result[:, :, :depth, :height, :width]
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

        # Preprocess input: resample to 256x256
        # x, original_size = self._preprocess_input(x)
        x, original_shape = self._pad_input_for_swin_unetr(x)

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

        result = self._crop_output_to_original_size(result, original_shape)

        # Postprocess output: resample to original size
        # result = self._postprocess_output(result, original_size)

        return result

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

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        if self.use_semantic_head:
            for param in self.semantic_head.parameters():
                param.requires_grad = True

    def finetune(
        self,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_best: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Memory-optimized finetune method with advanced training features.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            save_best: Whether to save the best model based on validation Dice score
            gradient_accumulation_steps: Steps to accumulate gradients (effective batch size multiplier)
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary with training history (losses, metrics)
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided to finetune the model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the correct device
        self.to(device)

        # Enable memory-efficient settings
        torch.backends.cudnn.benchmark = (
            True  # Optimize cudnn for consistent input sizes
        )

        print(f"🚀 Starting training for {epochs} epochs")
        print(f"   Device: {device}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Weight Decay: {weight_decay}")
        print(f"   Gradient Accumulation Steps: {gradient_accumulation_steps}")

        # Setup loss function, metrics, optimizer, and scaler
        loss_function, dice_metric, optimizer, scaler = self._setup_training_components(
            learning_rate, weight_decay
        )

        # Training history
        history = {"train_loss": [], "train_dice": []}

        best_train_dice = 0.0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\n📖 Epoch {epoch + 1}/{epochs}")

            # Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training phase
            self.train()
            train_losses = []

            train_pbar = tqdm(self.dataset.train_loader, desc="Training")
            for batch_idx, batch in enumerate(train_pbar):
                # Process each training batch with error handling
                outputs, loss_value, success = self._process_training_batch(
                    batch,
                    device,
                    loss_function,
                    optimizer,
                    scaler,
                    gradient_accumulation_steps,
                    max_grad_norm,
                    batch_idx,
                )

                if success:
                    train_losses.append(loss_value)

                    # Update progress bar
                    avg_loss = np.mean(
                        train_losses[-3:]
                    )  # Moving average of last 3 batches
                    train_pbar.set_postfix(
                        {
                            "Loss": f"{avg_loss:.4f}",
                        }
                    )

            # Record epoch results
            epoch_train_loss = np.mean(train_losses) if train_losses else float("inf")
            epoch_train_dice = 0.0
            # Compute training Dice score
            with torch.no_grad():
                for batch in tqdm(
                    self.dataset.train_loader, desc="Calculating Train Dice"
                ):
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)

                    # Forward pass
                    outputs = self.forward(images)
                    preds = torch.argmax(outputs, dim=1, keepdim=True)

                    # Compute Dice score
                    dice_metric(y_pred=preds, y=labels)

            dice_result = dice_metric.aggregate()

            # try to fix an error
            # Check if dice_result is a tuple is multiple metrics
            if isinstance(dice_result, tuple):
                epoch_train_dice = dice_result[0].mean().item()
            elif dice_result.numel() > 1:
                epoch_train_dice = dice_result.mean().item()
            else:
                epoch_train_dice = dice_result.item()
            dice_metric.reset()

            history["train_loss"].append(epoch_train_loss)
            history["train_dice"].append(epoch_train_dice)
            print(
                f"Epoch {epoch + 1} - Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f}"
            )

            # Save best model (move to CPU to save GPU memory)
            if save_best and epoch_train_dice > best_train_dice:
                best_train_dice = epoch_train_dice
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.state_dict().items()
                }

        # Restore best model if requested
        if save_best and best_model_state is not None:
            # Move back to device
            best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
            self.load_state_dict(best_model_state)

        print("\n✅ Training completed!")

        return history

    def _setup_training_components(self, learning_rate, weight_decay):
        """Setup loss function, metrics, optimizer, and scaler for training."""

        # Setup loss function (memory-efficient configuration)
        loss_function = DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            lambda_dice=0.65,
            lambda_ce=0.35,
        )

        # Setup metrics with robust configuration
        dice_metric = DiceMetric(
            include_background=False, reduction="mean_batch", get_not_nans=True
        )

        # Setup optimizer
        optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Setup gradient scaler for mixed precision
        scaler = torch.amp.GradScaler(self.device.type)

        return loss_function, dice_metric, optimizer, scaler

    def _process_training_batch(
        self,
        batch,
        device,
        loss_function,
        optimizer,
        scaler,
        gradient_accumulation_steps,
        max_grad_norm,
        batch_idx,
    ):
        """Process a single training batch with error handling."""
        try:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Ensure labels are in correct format [B, 1, D, H, W]
            if labels.dim() == 4:  # [B, D, H, W]
                labels = labels.unsqueeze(1)

            # Apply dataset-specific label decoding if available
            if hasattr(self.dataset, "de_encode"):
                labels = self.dataset.de_encode(labels)

            # Validate label range
            max_label = labels.max().item()
            if max_label >= self.num_classes:
                labels = torch.clamp(labels, 0, self.num_classes - 1)

            # Forward pass with mixed precision
            with torch.amp.autocast(device.type):
                outputs = self.forward(images)
                loss = loss_function(outputs, labels)
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

            # Clean up intermediate tensors
            del images, labels

            return outputs, loss.item() * gradient_accumulation_steps, True
        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM at batch {batch_idx}, clearing cache and continuing...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, 0.0, False
        # except Exception as e:
        #     print(f"⚠️ Error at batch {batch_idx}: {e}")
        #     return None, 0.0, False

    def load_task_vector(self, task_vector):
        """Load a task vector into the model."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in task_vector.vector:
                    param.data += task_vector.vector[name]

    def evaluate(self, batch_size_override=None):
        """
        Evaluate the model and return metrics on both train and test loaders.
        Memory-optimized version with aggressive memory management for large datasets like MMWHS.
        """
        import gc

        from monai.metrics import DiceMetric, HausdorffDistanceMetric

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.freeze()

        # Use smaller batch size for evaluation if dealing with memory issues
        original_batch_size = None
        if batch_size_override and hasattr(self.dataset, "train_loader"):
            original_batch_size = self.dataset.train_loader.batch_size
            print(
                f"Overriding batch size from {original_batch_size} to {batch_size_override} for evaluation"
            )

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean"
        )

        results = {}
        self.to(device)

        for split in ["train", "test"]:
            # Clear cache before each split
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            loader = getattr(self.dataset, f"{split}_loader", None)
            results[split] = {}
            if loader is None:
                continue

            print(f"🔍 Evaluating {split} split...")

            dice_metric.reset()
            hausdorff_metric.reset()
            has_labels = False

            # Process batches with aggressive memory management
            with torch.no_grad():

                for batch_idx, batch in enumerate(
                    tqdm(loader, desc=f"Evaluating {split}")
                ):
                    # Move data to device with non_blocking for efficiency
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch.get("label", None)

                    if labels is None:
                        del images
                        continue

                    labels = labels.to(device, non_blocking=True)
                    has_labels = True

                    # For very large images, process with reduced precision
                    with torch.amp.autocast(device.type):
                        outputs = self(images)

                    preds = torch.argmax(outputs, dim=1, keepdim=True)

                    # Compute metrics
                    dice_metric(y_pred=preds, y=labels)
                    hausdorff_metric(y_pred=preds, y=labels)

            # Aggregate results with error handling
            if has_labels:
                try:
                    dice_score = dice_metric.aggregate().item()
                    hausdorff_dist = hausdorff_metric.aggregate().item()
                    results[split] = {"dice": dice_score, "hausdorff": hausdorff_dist}
                    print(
                        f"✅ {split} - Dice: {dice_score:.4f}, Hausdorff: {hausdorff_dist:.4f}"
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"⚠️ Error aggregating metrics for {split}: {e}")
                    results[split] = {"dice": None, "hausdorff": None}
            else:
                results[split] = {"dice": None, "hausdorff": None}
        # Restore original batch size if it was overridden
        if original_batch_size and hasattr(self.dataset, "train_loader"):
            print(f"Restoring original batch size: {original_batch_size}")

        self.unfreeze()
        return results

    def _optimize_for_evaluation(self):
        """
        Apply memory optimizations specifically for evaluation mode.
        Useful for large datasets like MMWHS that can cause OOM errors.
        """
        # Enable memory efficient mode
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # Disable for variable input sizes
            torch.backends.cudnn.deterministic = True

        # If using semantic head, temporarily reduce some compute-heavy operations
        if self.use_semantic_head and hasattr(self.semantic_head, "conv_layers"):
            # Temporarily switch to eval mode for all components
            self.semantic_head.eval()

        print("🔧 Applied memory optimizations for evaluation")

    def evaluate_with_memory_management(self, max_batch_size=1):
        """
        Ultra-conservative evaluation method for datasets that cause persistent OOM.

        Args:
            use_cpu_offload: Move intermediate results to CPU to save GPU memory
            max_batch_size: Force maximum batch size (1 for most conservative)
        """
        import gc

        from monai.metrics import DiceMetric, HausdorffDistanceMetric

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔍 Starting memory-conservative evaluation on {device}")

        # Apply optimizations
        self._optimize_for_evaluation()
        self.eval()
        self.to(device)

        # Initialize metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean"
        )

        results = {}

        for split in ["train", "test"]:
            if not hasattr(self.dataset, f"{split}_loader"):
                results[split] = {"dice": None, "hausdorff": None}
                continue

            loader = getattr(self.dataset, f"{split}_loader")

            if loader is None:
                results[split] = {"dice": None, "hausdorff": None}
                continue

            dice_metric.reset()
            hausdorff_metric.reset()
            successful_batches = 0

            # Clear cache before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval {split}")):
                    try:
                        # Process one sample at a time if batch size > 1
                        images = batch["image"]
                        labels = batch.get("label")

                        if labels is None:
                            continue

                        # Process each sample in the batch individually if needed
                        batch_size = images.shape[0]

                        for sample_idx in range(min(batch_size, max_batch_size)):
                            # Extract single sample
                            sample_image = images[sample_idx : sample_idx + 1].to(
                                device, non_blocking=True
                            )
                            sample_label = labels[sample_idx : sample_idx + 1].to(
                                device, non_blocking=True
                            )

                            try:
                                # Forward pass with maximum memory conservation
                                with torch.amp.autocast(device.type):
                                    outputs = self(sample_image)

                                # Move predictions to CPU immediately if requested
                                preds = torch.argmax(outputs, dim=1, keepdim=True)

                                # Compute metrics
                                dice_metric(y_pred=preds, y=sample_label)
                                hausdorff_metric(y_pred=preds, y=sample_label)

                                successful_batches += 1

                                # Cleanup
                                del outputs, preds, sample_image, sample_label

                            except torch.cuda.OutOfMemoryError:
                                print(
                                    f"💥 OOM on sample {sample_idx} of batch {batch_idx}, skipping..."
                                )
                                # Emergency cleanup
                                for var_name in [
                                    "sample_image",
                                    "sample_label",
                                    "outputs",
                                    "preds",
                                ]:
                                    if var_name in locals():
                                        del locals()[var_name]
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                continue

                            # Clear cache after every sample for maximum safety
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        # Aggressive cleanup after each batch
                        del images, labels
                        if batch_idx % 5 == 0:  # Every 5 batches
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                    except Exception as e:
                        print(f"❌ Error processing batch {batch_idx}: {e}")
                        # Emergency cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue

            # Aggregate results
            if successful_batches > 0:
                try:
                    dice_score = dice_metric.aggregate().item()
                    hausdorff_dist = hausdorff_metric.aggregate().item()
                    results[split] = {"dice": dice_score, "hausdorff": hausdorff_dist}
                    print(
                        f"✅ {split} - Dice: {dice_score:.4f}, Hausdorff: {hausdorff_dist:.4f} ({successful_batches} successful batches)"
                    )
                except Exception as e:
                    print(f"⚠️ Error aggregating {split} metrics: {e}")
                    results[split] = {"dice": None, "hausdorff": None}
            else:
                results[split] = {"dice": None, "hausdorff": None}
                print(f"❌ No successful batches processed for {split}")

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
