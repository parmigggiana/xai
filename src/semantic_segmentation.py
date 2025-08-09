import os
from typing import OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.apps import download_url
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from tqdm import tqdm
from src.CLIPSeg import CLIPSeg
import gc
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class MedicalSegmenter(nn.Module):
    """
    Medical segmentation model with support for 3D and 2D inputs.
    """

    def __init__(
        self,
        encoder_type: str,
        num_classes: int,
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
                in_channels=1,
                out_channels=num_classes,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
            )
            # feature_dim = 768

            # Load pretrained SwinViT weights if available
            if pretrained:
                self._load_swinvit_weights()

            self.head = self.encoder.out
        elif encoder_type == "clipseg":

            model = CLIPSeg(
                classes=dataset.classnames, version="ViT-B/16", reduce_dim=64
            )
            model.load_state_dict(
                torch.load(
                    "data/rd64-uni-refined.pth",
                    map_location=torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    ),
                ),
                strict=False,
            )
            self.encoder = model
            self.head = model.head

        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. Supported: 'swin_unetr', 'clipseg'."
            )

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        self.device = device
        self.encoder.to(device)
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

    def _pad_input_for_swin_unetr(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pads the input tensor's depth to be divisible by 32 for SwinUNETR."""
        # Unpack shape: expect x to be [B, C, D, H, W] or [D, H, W]
        if x.dim() == 5 or x.dim() == 4:
            # [B, C, D, H, W] or [C, D, H, W]
            depth, height, width = x.shape[-3:]
        elif x.dim() == 3:
            # [D, H, W]
            depth, height, width = x.shape
        else:
            raise ValueError(f"Unsupported input shape for padding: {x.shape}")

        pad_depth = (32 - depth % 32) if depth % 32 != 0 else 0
        pad_height = (32 - height % 32) if height % 32 != 0 else 0
        pad_width = (32 - width % 32) if width % 32 != 0 else 0
        # F.pad uses (W_left, W_right, H_left, H_right, D_left, D_right)
        padding = (0, pad_width, 0, pad_height, 0, pad_depth)
        if pad_depth > 0 or pad_height > 0 or pad_width > 0:
            x = F.pad(x, padding, "constant", 0)
        return x, (depth, height, width)

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
        if self.encoder_type == "swin_unetr":
            x, original_shape = self._pad_input_for_swin_unetr(x)

        result = self.encoder(x)

        if self.encoder_type == "swin_unetr":
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

    def freeze_head(self):
        for param in self.head.parameters():
            param.requires_grad = False

    def freeze_body(self):
        self.freeze()
        for param in self.head.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def finetune(
        self,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_best: bool = True,
        max_grad_norm: float = 1.0,
    ):
        """
        Memory-optimized finetune method with advanced training features.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            save_best: Whether to save the best model based on validation Dice score
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

        # Setup loss function, metrics, optimizer, and scaler
        loss_function, dice_metric, optimizer, scaler = self._setup_training_components(
            learning_rate, weight_decay
        )

        # Training history
        history = {"val_loss": [], "val_dice": []}

        best_val_dice = 0.0
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

            # Validation phase
            epoch_val_loss = 0.0
            epoch_val_dice = 0.0

            with torch.no_grad():
                self.eval()  # Set to eval mode for validation
                for batch in tqdm(
                    self.dataset.val_loader, desc="Calculating Val Loss & Dice"
                ):
                    images = batch[0].to(device, non_blocking=True)
                    labels = batch[1].to(device, non_blocking=True)

                    # Ensure async transfers complete before proceeding
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    # Ensure labels are in correct format [B, 1, D, H, W]
                    if self.encoder_type == "swin_unetr" and labels.dim() == 4:
                        labels = labels.unsqueeze(1)

                    # If dataset has decode, apply it
                    # if hasattr(self.dataset, "decode"):
                    #     labels = self.dataset.decode(labels)

                    # Forward pass
                    with torch.amp.autocast(device.type):
                        outputs = self.forward(images)
                        epoch_val_loss += loss_function(outputs, labels) / len(
                            self.dataset.val_loader
                        )

                    preds = torch.argmax(outputs, dim=1, keepdim=True)
                    dice_metric(y_pred=preds, y=labels)

                # Aggregate validation metrics
                dice_result = dice_metric.aggregate()
                if isinstance(dice_result, tuple):
                    epoch_val_dice = dice_result[0].mean().item()
                elif hasattr(dice_result, "numel") and dice_result.numel() > 1:
                    epoch_val_dice = dice_result.mean().item()
                else:
                    epoch_val_dice = float(dice_result)
                dice_metric.reset()

            history["val_loss"].append(epoch_val_loss)
            history["val_dice"].append(epoch_val_dice)
            print(
                f"Epoch {epoch + 1} - Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}"
            )

            # Save best model based on validation Dice (move to CPU to save GPU memory)
            if save_best and epoch_val_dice > best_val_dice:
                best_val_dice = epoch_val_dice
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
        # Custom class weights: reduce background weight (assume background is excluded, so we add 1)
        class_weights = torch.ones(
            self.num_classes, dtype=torch.float32, device=self.device
        )
        # class_weights[0] = 0.2  # Reduce background weight (adjust as needed)

        loss_function = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.7,
            lambda_ce=0.3,
            weight=class_weights,
        )

        # Setup metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean")

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
        max_grad_norm,
        batch_idx,
    ):
        """Process a single training batch with error handling."""
        try:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            # Ensure labels are in correct format [B, 1, D, H, W]
            if self.encoder_type == "swin_unetr" and labels.dim() == 4:  # [B, D, H, W]
                labels = labels.unsqueeze(1)

            # Apply dataset-specific label decoding if available
            # if hasattr(self.dataset, "decode"):
            #     labels = self.dataset.decode(labels)

            # Validate label range
            # max_label = labels.max().item()
            # if max_label >= self.num_classes:
            #     labels = torch.clamp(labels, 0, self.num_classes - 1)

            # Forward pass with mixed precision
            with torch.amp.autocast(device.type):
                outputs = self.forward(images)
                loss = loss_function(outputs, labels)

            # Backward pass with gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            # Clean up intermediate tensors
            del images, labels

            return outputs, loss.item(), True
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
            for name, param in self.encoder.named_parameters():
                if name in task_vector.vector:
                    param.data += task_vector.vector[name]

    def evaluate(self):
        """
        Evaluate the model and return metrics on both train and test loaders.
        Memory-optimized version with aggressive memory management for large datasets like MMWHS.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.freeze()

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean"
        )

        results = {}
        self.to(device)

        for split in ["train", "val", "test"]:
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
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Evaluating {split}"):
                    images = batch[0].to(device, non_blocking=True)
                    labels = batch[1] if len(batch) > 1 else None

                    if labels is None:
                        del images
                        continue

                    labels = labels.to(device, non_blocking=True)

                    # Ensure async transfers complete before proceeding
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    # if hasattr(self.dataset, "decode"):
                    #     labels = self.dataset.decode(labels)

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

        self.unfreeze()
        return results
