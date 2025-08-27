import gc
import os
import zipfile
from pathlib import Path
from typing import OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from clip.model import CLIP
from clipseg.clipseg import CLIPDensePredT
from monai.apps import download_url
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
from tqdm import tqdm

from src.CLIPSeg import CLIPSeg


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
            # Extract dataset information for medical template selection
            dataset_info = None
            if dataset is not None:
                dataset_name = getattr(dataset, "name", type(dataset).__name__)
                domain = getattr(dataset, "domain", None)
                if dataset_name and domain:
                    dataset_info = (dataset_name, domain)

            model = CLIPSeg(
                classes=dataset.classnames,
                version="ViT-B/16",
                reduce_dim=64,  # Rimuovi questo parametro
                aggregation_mode="argmax",
                background_class=True,
                dataset_info=dataset_info,
            )

            # Download and load weights
            resource = "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download"
            dst = Path("./data/weights.zip")
            if not Path("./data/clipseg_weights/rd64-uni-refined.pth").exists():
                download_url(resource, dst)
                with zipfile.ZipFile(dst, "r") as zip_ref:
                    zip_ref.extractall("./data/")
                dst.unlink(missing_ok=True)

            print("🔄 Loading CLIPSeg weights...")
            from transformers import CLIPTextModel, CLIPVisionModel

            safe_globals = [
                CLIPDensePredT,
                CLIP,
                CLIPVisionModel,
                CLIPTextModel,
                torch.nn.Module,
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.BatchNorm2d,
                torch.nn.LayerNorm,
                torch.nn.Dropout,
                torch.nn.ReLU,
                torch.nn.GELU,
            ]

            with torch.serialization.safe_globals(safe_globals=safe_globals):
                state_dict = torch.load(
                    "data/clipseg_weights/rd64-uni-refined.pth",
                    map_location=torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    ),
                    weights_only=False,
                )

                # Carica i pesi nel componente clipseg specifico
                model.clipseg.load_state_dict(state_dict, strict=False)

            # Ensure CLIPSeg parameters are trainable (requires_grad=True)
            #try:
            #    for name, param in model.clipseg.named_parameters():
            #        param.requires_grad = True
            #    # Also ensure the head (prediction layers) are trainable
            #    for name, param in model.head.named_parameters():
            #        param.requires_grad = True
            #except Exception:
                # If the model structure differs, fall back to enabling all model params
            #    for name, param in model.named_parameters():
            #        param.requires_grad = True

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
        max_grad_norm: float = 5.0,  # previously 1.0
        visualize_batches: bool = True,
    ):
        if self.dataset is None:
            raise ValueError("Dataset must be provided to finetune the model")

        # Force CPU execution and comment out CUDA selection to disable CUDA optimizations
        device = torch.device("cpu")
        self.to(device)
        # Previous line:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # torch.backends.cudnn.benchmark = True

        print(f"🚀 Starting training for {epochs} epochs")
        print(f"   Device: {device}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Weight Decay: {weight_decay}")

        loss_function, dice_metric, optimizer, scaler = self._setup_training_components(
            learning_rate, weight_decay
        )

        # Debug: parameter counts and trainable modules
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Params: total={total_params:,}, trainable={trainable_params:,}")

        try:
            n_train_batches = len(self.dataset.train_loader)
            n_val_batches = len(self.dataset.val_loader)
            print(f"   Batches: train={n_train_batches}, val={n_val_batches}")
        except Exception:
            pass

        # Choose a few tracked parameters (prefer head/out) to monitor updates
        tracked = []
        for name, p in self.named_parameters():
            if p.requires_grad and ("head" in name or ".out" in name):
                tracked.append((name, p))
            if len(tracked) >= 3:
                break
        if not tracked:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    tracked.append((name, p))
                if len(tracked) >= 3:
                    break
        if tracked:
            print("   Tracking params:")
            for n, _ in tracked:
                print(f"     - {n}")

        history = {"train_loss": [], "val_loss": [], "val_dice": []}

        best_val_dice = 0.0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\n📖 Epoch {epoch + 1}/{epochs}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.train()
            train_losses = []

            # Debug: snapshot tracked param norms before epoch
            pre_norms = {n: p.detach().float().norm().item() for n, p in tracked}
            # Debug: epoch aggregates
            epoch_unique_labels = set()
            grad_nonzero_batches = 0
            batch_count = 0

            # Debug: print current LR(s)
            try:
                lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
                print(
                    f"   LR(s): {', '.join(f'{lr:.6e}' for lr in lrs if lr is not None)}"
                )
            except Exception:
                pass

            train_pbar = tqdm(self.dataset.train_loader, desc="Training")
            for batch_idx, batch in enumerate(train_pbar):
                outputs, loss_value, success, dbg = self._process_training_batch(
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
                    avg_loss = np.mean(train_losses[-3:])
                    train_pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

                    # Update epoch-level debug stats
                    batch_count += 1
                    if dbg is not None:
                        if "unique_labels" in dbg:
                            try:
                                epoch_unique_labels.update(
                                    dbg["unique_labels"]
                                )  # list of ints
                            except Exception:
                                pass
                        if "grad_norm" in dbg and dbg["grad_norm"] is not None:
                            if dbg["grad_norm"] > 0:
                                grad_nonzero_batches += 1

            if train_losses:
                epoch_train_loss = float(np.mean(train_losses))
                history["train_loss"].append(epoch_train_loss)
                print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}")

            # Debug: parameter update magnitudes on tracked params
            post_norms = {n: p.detach().float().norm().item() for n, p in tracked}
            if tracked:
                print("   Param norm deltas (after epoch):")
                for n in pre_norms:
                    delta = post_norms[n] - pre_norms[n]
                    print(
                        f"     {n}: Δnorm={delta:+.6e} (before={pre_norms[n]:.6e}, after={post_norms[n]:.6e})"
                    )

            # Debug: epoch gradient non-zero ratio and label coverage
            if batch_count > 0:
                ratio = grad_nonzero_batches / batch_count
                print(
                    f"   Grad non-zero in batches: {grad_nonzero_batches}/{batch_count} ({ratio:.1%})"
                )
            if epoch_unique_labels:
                try:
                    print(
                        f"   Labels seen this epoch: {sorted(list(epoch_unique_labels))}"
                    )
                except Exception:
                    pass

            # Debug: AMP scaler
            try:
                cur_scale = scaler.get_scale() if scaler is not None else None
                if cur_scale is not None:
                    print(f"   GradScaler scale: {cur_scale}")
            except Exception:
                pass

            self.eval()
            val_losses = []
            epoch_val_dice = 0.0

            with torch.no_grad():
                # Print a compact summary of a few validation batches (images, preds, labels)
                for batch_idx, batch in enumerate(
                    tqdm(self.dataset.val_loader, desc="Validating")
                ):
                    images = batch[0].to(device, non_blocking=True)
                    labels = batch[1].to(device, non_blocking=True)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    if self.encoder_type == "swin_unetr" and labels.dim() == 4:
                        labels = labels.unsqueeze(1)

                    # Remove AMP autocast here as well
                    outputs = self.forward(images)
                    loss_val = loss_function(outputs, labels)
                    val_losses.append(loss_val.item())

                    preds = torch.argmax(outputs, dim=1, keepdim=True)
                    dice_metric(y_pred=preds, y=labels)

                    # Print or visualize debug info (limit verbose output)
                    if visualize_batches:
                        try:
                            self._visualize_batch(images, preds, labels, title=f"Val batch {batch_idx}")
                        except Exception as e:
                            print(f"[DEBUG] Visualization failed for val batch {batch_idx}: {e}")
                    else:
                        try:
                            imgs_np = images.detach().cpu().numpy()
                            labels_np = labels.detach().cpu().numpy()
                            preds_np = preds.detach().cpu().numpy()
                            print(
                                f"[DEBUG] Val batch {batch_idx} - images:{imgs_np.shape}, labels:{labels_np.shape}, preds:{preds_np.shape}"
                            )
                            print(
                                f"[DEBUG] Val batch {batch_idx} - unique labels: {np.unique(labels_np)}, unique preds: {np.unique(preds_np)}"
                            )
                            # print small sample to inspect values without flooding
                            flat_labels = labels_np.flatten()
                            flat_preds = preds_np.flatten()
                            print(f"[DEBUG] sample labels[:20]: {flat_labels[:20]}")
                            print(f"[DEBUG] sample preds[:20]: {flat_preds[:20]}")
                        except Exception as e:
                            print(f"[DEBUG] Failed to print val batch {batch_idx}: {e}")

                epoch_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
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
                f"Epoch {epoch+1} - Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}"
            )

            if save_best and epoch_val_dice > best_val_dice:
                best_val_dice = epoch_val_dice
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.state_dict().items()
                }
                print(f"   ✅ New best Val Dice: {best_val_dice:.4f}")

        if save_best and best_model_state is not None:
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
        #class_weights[0] = 0.1  # Reduce background weight (adjust as needed)

        # CLIPSeg produces sigmoids / probability-like outputs; do not apply softmax again.
        loss_function = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=False,  # Changed: don't apply softmax to already-probabilistic outputs
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

        # NOTE: AMP/GradScaler disabled to remove mixed-precision optimizations
        # scaler = torch.amp.GradScaler(self.device.type)
        scaler = None

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

            optimizer.zero_grad()
            
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            # If dataset provides a decode mapping (e.g. CHAOS/MMWHS), apply it here
            if hasattr(self.dataset, "decode"):
                try:
                    labels = self.dataset.decode(labels)
                except Exception:
                    pass

            # Ensure labels are integer class indices for to_onehot_y=True
            if not torch.is_floating_point(labels) or labels.dtype != torch.long:
                labels = labels.long()
            
            # Ensure labels are in correct format [B, 1, D, H, W]
            if self.encoder_type == "swin_unetr" and labels.dim() == 4:  # [B, D, H, W]
                labels = labels.unsqueeze(1)

            # Apply dataset-specific label decoding if available
            # NOTE: dataset.decode commented out elsewhere; keep as-is

            # Forward pass WITHOUT AMP (mixed precision disabled)
            outputs = self.forward(images)
            loss = loss_function(outputs, labels)

            # Debug ogni 20 batch
            if batch_idx % 20 == 0:
                print("labels dtype/min/max:", labels.dtype, labels.min().item(), labels.max().item())
                uniq = np.unique(labels.detach().cpu().numpy())
                print(f"[DEBUG] Batch {batch_idx} - Loss: {loss.item():.6f}")
                print(f"[DEBUG] Unique labels: {uniq}")
                print(
                    f"[DEBUG] Outputs -> mean: {outputs.mean().item():.6f}, std: {outputs.std().item():.6f}"
                )
                print("Unique labels in batch:", torch.unique(labels))
                # Convert to class indices for compact debug (choose argmax for multi-class)
                try:
                    preds_idx = torch.argmax(outputs, dim=1, keepdim=False)
                    print("Unique prediction classes in batch:", torch.unique(preds_idx))
                except Exception:
                    # Fallback: show summary stats if argmax not applicable
                    print("Unique predictions (summary):", torch.unique(outputs))

            # Backward pass (no GradScaler) and NO gradient norm clipping
            loss.backward()
            # Removed gradient clipping:
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            optimizer.step()

            # Prepare debug info (compute grad norm without clipping if needed)
            total_norm = None
            try:
                total_norm = 0.0
                param_count = 0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    total_norm = total_norm ** (1.0 / 2)
            except Exception:
                total_norm = None

            dbg = {
                "batch_idx": batch_idx,
                "grad_norm": float(total_norm) if total_norm is not None else None,
                "unique_labels": [
                    int(x) for x in np.unique(labels.detach().cpu().numpy()).tolist()
                ],
            }

            # Clean up intermediate tensors
            del images, labels

            return outputs, loss.item(), True, dbg
        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM at batch {batch_idx}, clearing cache and continuing...")
            # CUDA cache calls commented out per request to remove CUDA optimizations
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            return None, 0.0, False, None
        # except Exception as e:
        #     print(f"⚠️ Error at batch {batch_idx}: {e}")
        #     return None, 0.0, False, None

    def load_task_vector(self, task_vector):
        """Load a task vector into the model."""
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if name in task_vector.vector:
                    param.data += task_vector.vector[name]

    def _visualize_batch(self, images, preds, labels, title: str = "batch"):
        """Display images, predictions and labels side-by-side for the first item in the batch.

        Expects images: [B, C, H, W] or [B, 1, H, W]; preds/labels: [B, 1, H, W] (class indices).
        """
        # Move to cpu numpy
        imgs = images.detach().cpu()
        p = preds.detach().cpu()
        l = labels.detach().cpu()

        # Take first element
        img = imgs[0]
        pred = p[0]
        lab = l[0]

        # Squeeze channel dims
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.ndim == 3 and img.shape[0] > 1:
            # If multi-channel, take first channel for display
            img = img[0]

        if pred.ndim > 2:
            pred = pred.squeeze(0)
        if lab.ndim > 2:
            lab = lab.squeeze(0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"{title} - image")
        axes[0].axis("off")

        axes[1].imshow(pred, cmap="viridis")
        axes[1].set_title(f"{title} - pred")
        axes[1].axis("off")

        axes[2].imshow(lab, cmap="viridis")
        axes[2].set_title(f"{title} - label")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    def evaluate(self, visualize: bool = False):
        """
        Evaluate the model and return metrics on both train and test loaders.
        Memory-optimized version with aggressive memory management for large datasets like MMWHS.
        """
        # Force CPU execution and disable CUDA/AMP synchronization
        device = torch.device("cpu")
        self.eval()
        self.freeze()

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean"
        )

        results = {}
        self.to(device)

        for split in ["train", "val", "test"]:
            # Clear cache before each split - CUDA calls commented out
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     torch.cuda.synchronize()
            gc.collect()

            loader = getattr(self.dataset, f"{split}_loader", None)
            results[split] = {}
            if loader is None:
                continue

            has_labels = False  # Track if any labels were processed

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
                    images = batch[0].to(device)
                    labels = batch[1] if len(batch) > 1 else None

                    if labels is None:
                        del images
                        continue

                    labels = labels.to(device)
                    has_labels = True  # At least one batch had labels

                    # Remove AMP autocast here as well
                    outputs = self(images)

                    preds = torch.argmax(outputs, dim=1, keepdim=True)

                    # Compute metrics
                    dice_metric(y_pred=preds, y=labels)
                    hausdorff_metric(y_pred=preds, y=labels)

                    # Visualize or print compact debug info for evaluation batches
                    if visualize:
                        try:
                            self._visualize_batch(images, preds, labels, title=f"Eval {split} batch {idx}")
                        except Exception as e:
                            print(f"[DEBUG] Visualization failed for eval {split} batch {idx}: {e}")
                    else:
                        try:
                            imgs_np = images.detach().cpu().numpy()
                            labels_np = labels.detach().cpu().numpy()
                            preds_np = preds.detach().cpu().numpy()
                            print(
                                f"[DEBUG] Eval {split} batch {idx} - images:{imgs_np.shape}, labels:{labels_np.shape}, preds:{preds_np.shape}"
                            )
                            print(
                                f"[DEBUG] Eval {split} batch {idx} - unique labels: {np.unique(labels_np)}, unique preds: {np.unique(preds_np)}"
                            )
                            flat_labels = labels_np.flatten()
                            flat_preds = preds_np.flatten()
                            print(f"[DEBUG] sample labels[:20]: {flat_labels[:20]}")
                            print(f"[DEBUG] sample preds[:20]: {flat_preds[:20]}")
                        except Exception as e:
                            print(f"[DEBUG] Failed to print eval batch {idx}: {e}")

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
