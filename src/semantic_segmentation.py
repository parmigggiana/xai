import contextlib
import gc
import os
import zipfile
from pathlib import Path
from typing import Optional, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        # AMP dtype preference (set later based on hardware)
        self._amp_dtype = None

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
            if pretrained:
                self._load_swinvit_weights()

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
                reduce_dim=64,
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

            state_dict = torch.load(
                "data/clipseg_weights/rd64-uni-refined.pth",
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
                weights_only=False,
            )

            #### Debug: compare keys before loading
            # model_state = model.clipseg.state_dict()
            # sd_keys = set(state_dict.keys())
            # model_keys = set(model_state.keys())
            # unexpected = sorted(sd_keys - model_keys)
            # missing = sorted(model_keys - sd_keys)

            # # Attempt to load with strict=False to allow partial matches
            # load_result = model.clipseg.load_state_dict(state_dict, strict=False)

            # # Summary of key matching
            # matched = len(sd_keys & model_keys)
            # print("model state keys:", len(model_keys))
            # print("checkpoint state keys:", len(sd_keys))
            # print(
            #     f"CLIPSeg weight load summary: matched={matched}, "
            #     f"missing={len(missing)}, unexpected={len(unexpected)}"
            # )

            # # Print missing and unexpected keys (limit output length)
            # if load_result.missing_keys:
            #     print(f"Missing keys ({len(load_result.missing_keys)}):")
            #     for k in load_result.missing_keys[:25]:
            #         print(f"  - {k}")
            #     if len(load_result.missing_keys) > 25:
            #         print(f"  ... and {len(load_result.missing_keys) - 25} more")

            # if load_result.unexpected_keys:
            #     print(f"Unexpected keys ({len(load_result.unexpected_keys)}):")
            #     for k in load_result.unexpected_keys[:25]:
            #         print(f"  - {k}")
            #     if len(load_result.unexpected_keys) > 25:
            #         print(f"  ... and {len(load_result.unexpected_keys) - 25} more")

            # # Extra: detect any size mismatches among intersecting keys
            # size_mismatches = []
            # for k in sd_keys & model_keys:
            #     try:
            #         sd_shape = tuple(state_dict[k].shape)  # type: ignore[attr-defined]
            #         mdl_shape = tuple(model_state[k].shape)  # type: ignore[attr-defined]
            #         if sd_shape != mdl_shape:
            #             size_mismatches.append((k, sd_shape, mdl_shape))
            #     except Exception:
            #         # If an entry isn't a Tensor-like with shape, skip it
            #         continue
            # if size_mismatches:
            #     print(f"Size-mismatched tensors ({len(size_mismatches)}):")
            #     for k, s1, s2 in size_mismatches[:25]:
            #         print(f"  - {k}: checkpoint {s1} != model {s2}")
            #     if len(size_mismatches) > 25:
            #         print(f"  ... and {len(size_mismatches) - 25} more")
            #### Debug end

            self.encoder = model

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

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    # --- Optional fine-grained freezing for CLIPSeg encoders ---
    def freeze_text_encoder(self):
        """Freeze CLIP text encoder parameters (when using CLIPSeg).

        This avoids tracking/updating the CLIP text tower which isn't on the
        training graph when using cached, detached text embeddings.
        """
        if getattr(self, "encoder_type", None) != "clipseg":
            return
        for name, p in self.encoder.named_parameters():
            # In CLIPSeg, text params live under 'clip_model.' but not under '.visual.'
            if "clip_model." in name and ".visual." not in name:
                if p.requires_grad:
                    p.requires_grad = False

    def unfreeze_text_encoder(self):
        """Unfreeze CLIP text encoder parameters (when using CLIPSeg)."""
        if getattr(self, "encoder_type", None) != "clipseg":
            return
        unfrozen = 0
        for name, p in self.encoder.named_parameters():
            if "clip_model." in name and ".visual." not in name:
                if not p.requires_grad:
                    p.requires_grad = True
                    unfrozen += 1
        if unfrozen:
            print(f"Unfrozen {unfrozen} CLIP text-encoder params.")

    def _unpack_batch(self, batch):
        """Return (images, labels) from either a dict-style MONAI batch or a tuple/list.
        Falls back to common alternative keys when using dict-based datasets.
        """
        images = labels = None
        try:
            if isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
                labels = batch.get("label", batch.get("labels"))
                if labels is None:
                    labels = (
                        batch.get("seg") or batch.get("mask") or batch.get("target")
                    )
            else:
                # tuple/list
                if len(batch) > 0:
                    images = batch[0]
                if len(batch) > 1:
                    labels = batch[1]
        except Exception:
            # Leave as None if extraction fails
            pass
        return images, labels

    def finetune(
        self,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_best: bool = True,
        max_grad_norm: float = 5.0,  # previously 1.0
        visualize_batches: bool = False,
        early_stop_patience: int = 5,
        profile: bool | str = False,  # False | 'cprofile' | 'torch'
        profile_dir: str = "./outputs/profiling",
        debug: Optional[bool] = None,
        compile_model: bool = False,
        # Validation performance knobs
        val_max_batches: Optional[int] = None,
        # Learning rate schedule: linearly decay from lr_start to lr_end over lr_decay_epochs
        lr_start: float = 1e-3,
        lr_end: float = 1e-4,
        lr_decay_epochs: int = 20,
    ):
        # If caller doesn't pass debug (None), use global DEBUG (if available);
        # otherwise, always respect the explicit value (True/False) passed in.
        if debug is None:
            debug = False
        if self.dataset is None:
            raise ValueError("Dataset must be provided to finetune the model")

        # Force CPU execution and comment out CUDA selection to disable CUDA optimizations
        # device = torch.device("cpu")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Enable CuDNN autotuner when using CUDA to speed up convolutions for varying input sizes
        try:
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Configure TF32 and AMP dtype based on hardware support
        try:
            if device.type == "cuda":
                # Allow TF32 matmul/conv on Ampere and newer (no-op on older GPUs)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Encourage TF32 for float32 matmuls where applicable
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

                # Prefer BF16 autocast on GPUs that support it; else fall back to FP16
                if (
                    hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                ):
                    self._amp_dtype = torch.bfloat16
                else:
                    self._amp_dtype = torch.float16
            else:
                self._amp_dtype = None
        except Exception:
            self._amp_dtype = None

        # Optionally compile the encoder for speed (CUDA-only)
        if compile_model and device.type == "cuda" and hasattr(torch, "compile"):
            try:
                # Prefer more aggressive autotuning on CUDA
                self.encoder = torch.compile(self.encoder, mode="max-autotune")
                if debug:
                    print("   torch.compile enabled (encoder)")
            except Exception as e:
                print(f"[compile] disabled (fallback): {e}")

        print(f"🚀 Starting training for {epochs} epochs")
        print(f"   Device: {device}")
        # Fixed 5-epoch warmup for stability
        warmup_epochs = 5
        if debug:
            print(
                f"   LR: 5-epoch warmup -> cosine decay: start={lr_start:.3e} -> end={lr_end:.3e} over {lr_decay_epochs} epochs (warmup={warmup_epochs})"
            )
            print(f"   Weight Decay: {weight_decay}")

        # Initialize optimizer at starting LR
        init_lr = lr_start
        loss_function, optimizer, scaler = self._setup_training_components(
            init_lr, weight_decay, debug=debug
        )

        # Warmup + Cosine LR schedule (epoch-based). Clamp after decay window.
        # Warmup: linear ramp to lr_start over `warmup_epochs`.
        # Cosine: lr(e) = lr_end + 0.5*(lr_start - lr_end)*(1 + cos(pi * t)), t in [0,1]
        def _lr_at_epoch(e: int) -> float:
            # Linear warmup
            w = max(0, int(warmup_epochs))
            if w > 0 and e < w:
                # Start near-0 and ramp to lr_start by end of warmup
                return lr_start * float(e + 1) / float(w)

            # Cosine decay after warmup
            steps = max(1, int(lr_decay_epochs))
            # Progress t from 0 at the first post-warmup epoch to 1 at the end of decay window
            idx = max(0, min(e - w, steps))
            t = idx / float(steps)
            return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + np.cos(np.pi * t))

        # Debug: parameter counts and trainable modules
        tracked = []
        if debug:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(f"   Params: total={total_params:,}, trainable={trainable_params:,}")

            try:
                n_train_batches = len(self.dataset.train_loader)
                n_val_batches = len(self.dataset.val_loader)
                print(f"   Batches: train={n_train_batches}, val={n_val_batches}")
            except Exception:
                pass

            # Choose a few tracked parameters (prefer head/out) to monitor updates (debug only)
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

        history = {"train_loss": [], "val_loss": []}

        best_model_state = None
        # Early stopping state (track best by validation loss)
        best_val_loss = float("inf")
        epochs_no_improve = 0

        # Optional profilers (opt-in, lazy import)
        if profile == "cprofile":
            from src.profiling import cprofile_ctx as _cprofile_ctx

            prof_cm = _cprofile_ctx("train", out_dir=profile_dir)
        elif profile == "torch":
            from src.profiling import torch_profiler_ctx as _torch_profiler_ctx

            prof_cm = _torch_profiler_ctx(
                name="train",
                out_dir=profile_dir,
                wait=1,
                warmup=1,
                active=5,
            )
        else:
            prof_cm = contextlib.nullcontext()

        with prof_cm as prof:
            for epoch in range(epochs):
                print(f"\n📖 Epoch {epoch + 1}/{epochs}")

                # Apply scheduled LR at the start of the epoch
                try:
                    new_base_lr = _lr_at_epoch(epoch)
                    for pg in optimizer.param_groups:
                        factor = pg.get("lr_factor", 1.0)
                        pg["lr"] = new_base_lr * float(factor)
                except Exception:
                    pass

                self.train()
                train_losses = []

                # Debug: snapshot tracked param norms before epoch
                if debug and tracked:
                    pre_norms = {
                        n: p.detach().float().norm().item() for n, p in tracked
                    }
                else:
                    pre_norms = {}
                # Debug: epoch aggregates
                if debug:
                    epoch_unique_labels = set()
                    grad_nonzero_batches = 0
                    batch_count = 0

                # Debug: print current LR(s)
                if debug:
                    try:
                        lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
                        print(
                            f"   LR(s): {', '.join(f'{lr:.6e}' for lr in lrs if lr is not None)}"
                        )
                    except Exception:
                        pass

                # Lazy import timer only when debug is enabled to minimize overhead
                if debug:
                    from src.profiling import timer as _timer
                else:
                    _timer = None

                train_pbar = tqdm(self.dataset.train_loader, desc="Training")
                for batch_idx, batch in enumerate(train_pbar):
                    ctx = (
                        _timer(f"batch {batch_idx}")
                        if _timer
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        outputs, loss_value, success, dbg = (
                            self._process_training_batch(
                                batch,
                                device,
                                loss_function,
                                optimizer,
                                scaler,
                                max_grad_norm,
                                batch_idx,
                                compute_debug=debug,
                            )
                        )

                    if success:
                        train_losses.append(loss_value)
                        avg_loss = np.mean(train_losses[-3:])
                        train_pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

                        # Update epoch-level debug stats
                        if debug:
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

                                try:
                                    layer_grads = dbg.get("layer_grad_sums")
                                    if layer_grads is not None:
                                        compact = ", ".join(
                                            (
                                                f"{name}: {val:.3e}"
                                                if val is not None
                                                else f"{name}: None"
                                            )
                                            for name, val in layer_grads.items()
                                        )
                                        print(
                                            f"      step {batch_idx}: |grad| sums -> {compact}"
                                        )
                                except Exception:
                                    pass

                if train_losses:
                    epoch_train_loss = float(np.mean(train_losses))
                    history["train_loss"].append(epoch_train_loss)
                    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}")

                # Debug: parameter update magnitudes on tracked params
                if debug and tracked:
                    post_norms = {
                        n: p.detach().float().norm().item() for n, p in tracked
                    }
                    if tracked:
                        print("   Param norm deltas (after epoch):")
                        for n in pre_norms:
                            delta = post_norms[n] - pre_norms[n]
                            print(
                                f"     {n}: Δnorm={delta:+.6e} (before={pre_norms[n]:.6e}, after={post_norms[n]:.6e})"
                            )

                # Debug: epoch gradient non-zero ratio and label coverage
                if debug:
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

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                with torch.inference_mode():

                    # Print a compact summary of a few validation batches (images, preds, labels)
                    for batch_idx, batch in enumerate(
                        tqdm(self.dataset.val_loader, desc="Validating")
                    ):
                        if val_max_batches is not None and batch_idx >= int(
                            val_max_batches
                        ):
                            break
                        if profile == "torch" and prof is not None:
                            try:
                                prof.step()
                            except Exception:
                                pass
                        images, labels = self._unpack_batch(batch)
                        if images is None or labels is None:
                            # Skip batches without required components
                            continue
                        # Drop MetaTensor metadata for network forward to avoid inconsistent meta batching
                        if hasattr(images, "as_tensor"):
                            images = images.as_tensor()
                        if hasattr(labels, "as_tensor"):
                            labels = labels.as_tensor()
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        if self.encoder_type == "swin_unetr" and labels.dim() == 4:
                            labels = labels.unsqueeze(1)

                        # Use mixed precision for validation forward for speed when CUDA is available
                        if device.type == "cuda":
                            amp_dtype = getattr(self, "_amp_dtype", torch.float16)
                            with torch.amp.autocast("cuda", dtype=amp_dtype):
                                outputs = self.forward(images)
                                loss_val = loss_function(outputs, labels)
                        else:
                            outputs = self.forward(images)
                            loss_val = loss_function(outputs, labels)
                        val_losses.append(loss_val.item())

                        preds = torch.argmax(outputs, dim=1, keepdim=True)

                        # Cache summary after each batch (validation) - removed verbose prints

                        # Print or visualize debug info (limit verbose output)
                        if visualize_batches or debug:
                            try:
                                self._visualize_batch(
                                    images,
                                    preds,
                                    labels,
                                    title=f"Val batch {batch_idx}",
                                )
                            except Exception as e:
                                print(
                                    f"[DEBUG] Visualization failed for val batch {batch_idx}: {e}"
                                )
                        else:
                            # keep loop lightweight when not visualizing
                            pass

                    epoch_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

                history["val_loss"].append(epoch_val_loss)
                print(f"Epoch {epoch+1} - Val Loss: {epoch_val_loss:.4f}")

                # Save best and early stopping based on validation loss only
                improved = epoch_val_loss < best_val_loss - 1e-6
                if improved:
                    if save_best:
                        best_model_state = {
                            k: v.cpu().clone() for k, v in self.state_dict().items()
                        }
                        print(f"   New best Val Loss: {epoch_val_loss:.4f}")
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if debug:
                        print(
                            f"   No Val Loss improvement: {epochs_no_improve}/{early_stop_patience}"
                        )
                    if epochs_no_improve >= early_stop_patience:
                        print(
                            f"Early stopping triggered: no Val Loss improvement for {early_stop_patience} epochs."
                        )
                        break

        if save_best and best_model_state is not None:
            best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
            self.load_state_dict(best_model_state)

        print("\n✅ Training completed!")
        return history

    def _setup_training_components(self, learning_rate, weight_decay, debug=False):
        """Setup loss function, metrics, optimizer, and scaler for training."""
        # Setup loss function (memory-efficient configuration)
        # Custom class weights: reduce background weight (assume background is excluded, so we add 1)
        class_weights = torch.ones(
            self.num_classes, dtype=torch.float32, device=self.device
        )
        class_weights[0] = 0.2  # Reduce background weight

        # CLIPSeg produces sigmoids / probability-like outputs; do not apply softmax again.
        loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            # lambda_dice=0.7,
            # lambda_ce=0.3,
            # weight=class_weights,
        )

        # Setup optimizer with separate LR for backbone vs decoder (backbone = 0.1x)
        param_groups, grouping_dbg = self._build_param_groups(
            base_lr=learning_rate, weight_decay=weight_decay
        )

        if debug:
            try:
                total = sum(1 for _ in self.parameters())
                trainable = sum(1 for p in self.parameters() if p.requires_grad)
                print(f"   Trainable params: {trainable} of {total}")
                print(
                    "   Optimizer groups: "
                    + ", ".join(
                        f"{k}={v}" for k, v in grouping_dbg.items() if k != "patterns"
                    )
                )
                if "patterns" in grouping_dbg:
                    print(
                        f"   Grouping patterns -> backbone: {grouping_dbg['patterns']['backbone']} | decoder: other trainable"
                    )
            except Exception:
                pass

        # AdamW fused fallback (CPU or older PyTorch may not support 'fused')
        if self.device.type == "cuda":
            try:
                optimizer = optim.AdamW(
                    param_groups,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    fused=True,
                )
            except (TypeError, RuntimeError):
                optimizer = optim.AdamW(
                    param_groups, lr=learning_rate, weight_decay=weight_decay
                )
        else:
            # Do not pass 'fused' on CPU to avoid unsupported-arg errors
            optimizer = optim.AdamW(
                param_groups, lr=learning_rate, weight_decay=weight_decay
            )

        # Enable AMP GradScaler on CUDA only for FP16; BF16 does not need GradScaler
        amp_dtype = getattr(self, "_amp_dtype", None)
        if self.device.type == "cuda" and amp_dtype == torch.float16:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except Exception:
                scaler = None
        else:
            scaler = None

        return loss_function, optimizer, scaler

    def _build_param_groups(self, base_lr: float, weight_decay: float):
        """Return optimizer param groups with LR multiplier on backbone.

        Policy: decoder uses base_lr; backbone uses 0.1 * base_lr.
        Backbone/decoder split is inferred from stable name patterns per encoder type:
          - swin_unetr: backbone params start with 'encoder.swinViT.'
          - clipseg:    backbone params start with 'encoder.clipseg.model.'
        All other trainable params are treated as decoder.
        """
        # Choose patterns
        if self.encoder_type == "swin_unetr":
            backbone_patterns = ("encoder.swinViT.",)
        elif self.encoder_type == "clipseg":
            backbone_patterns = ("encoder.clipseg.clip_model.",)
        else:
            backbone_patterns = ()

        backbone_params = []
        decoder_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(pref) for pref in backbone_patterns):
                backbone_params.append(p)
            else:
                decoder_params.append(p)

        # Fallbacks to avoid empty groups
        if not decoder_params and backbone_params:
            # Unusual: if everything matched backbone, keep small group for backbone
            pass
        elif decoder_params and not backbone_params:
            # If we failed to detect backbone, treat all as decoder (single group)
            param_groups = [
                {
                    "params": decoder_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ]
            dbg = {
                "decoder_params": len(decoder_params),
                "backbone_params": 0,
                "patterns": {"backbone": backbone_patterns},
            }
            return param_groups, dbg

        param_groups = []
        if decoder_params:
            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "lr_factor": 1.0,
                }
            )
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": base_lr * 0.1,
                    "weight_decay": weight_decay * 0.1,
                    "lr_factor": 0.1,
                }
            )
        dbg = {
            "decoder_params": len(decoder_params),
            "backbone_params": len(backbone_params),
            "patterns": {"backbone": backbone_patterns},
        }
        return param_groups, dbg

    def _process_training_batch(
        self,
        batch,
        device,
        loss_function,
        optimizer,
        scaler,
        max_grad_norm,
        batch_idx,
        compute_debug: bool = False,
    ):
        """Process a single training batch with error handling."""

        try:

            optimizer.zero_grad(set_to_none=True)

            images, labels = self._unpack_batch(batch)
            if images is None or labels is None:
                raise RuntimeError("Batch does not contain 'image' and 'label'.")
            # Convert MetaTensor -> Tensor to avoid meta tracking issues in forward
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            if hasattr(labels, "as_tensor"):
                labels = labels.as_tensor()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Ensure labels are in correct format [B, 1, D, H, W]
            if self.encoder_type == "swin_unetr" and labels.dim() == 4:  # [B, D, H, W]
                labels = labels.unsqueeze(1)

            # Apply dataset-specific label decoding if available
            # NOTE: dataset.decode commented out elsewhere; keep as-is

            # Forward pass with AMP (if enabled)
            if device.type == "cuda":
                amp_dtype = getattr(self, "_amp_dtype", torch.float16)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = self.forward(images)
                    loss = loss_function(outputs, labels)
            else:
                outputs = self.forward(images)
                loss = loss_function(outputs, labels)

            # Debug ogni 20 batch
            # if batch_idx % 20 == 0:
            #     print(
            #         "labels dtype/min/max:",
            #         labels.dtype,
            #         labels.min().item(),
            #         labels.max().item(),
            #     )
            #     uniq = np.unique(labels.detach().cpu().numpy())
            #     print(f"[DEBUG] Batch {batch_idx} - Loss: {loss.item():.6f}")
            #     print(f"[DEBUG] Unique labels: {uniq}")
            #     print(
            #         f"[DEBUG] Outputs -> mean: {outputs.mean().item():.6f}, std: {outputs.std().item():.6f}"
            #     )
            #     print("Unique labels in batch:", torch.unique(labels))
            #     # Convert to class indices for compact debug (choose argmax for multi-class)
            #     try:
            #         preds_idx = torch.argmax(outputs, dim=1, keepdim=False)
            #         print(
            #             "Unique prediction classes in batch:", torch.unique(preds_idx)
            #         )
            #     except Exception:
            #         # Fallback: show summary stats if argmax not applicable
            #         print("Unique predictions (summary):", torch.unique(outputs))

            # Backward/step: use GradScaler only for FP16; BF16/FP32 use standard path
            if (
                device.type == "cuda"
                and getattr(self, "_amp_dtype", None) == torch.float16
                and scaler is not None
            ):
                scaler.scale(loss).backward()
                # Unscale before optional gradient clipping
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                if max_grad_norm and max_grad_norm > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm and max_grad_norm > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    except Exception:
                        pass
                optimizer.step()

            # Prepare debug info only when requested to minimize overhead
            if compute_debug:
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

                try:
                    uniq_labels = [
                        int(x)
                        for x in np.unique(labels.detach().cpu().numpy()).tolist()
                    ]
                except Exception:
                    uniq_labels = []

                # Per-step gradient sums for a few representative layers
                layer_grad_sums = None

                # Summary counts: trainable vs. with gradients this step
                grad_counts = None
                try:
                    trainable_cnt = 0
                    grad_present_cnt = 0
                    for p in self.parameters():
                        if p.requires_grad:
                            trainable_cnt += 1
                            if p.grad is not None:
                                grad_present_cnt += 1
                    grad_counts = {
                        "trainable": trainable_cnt,
                        "with_grad": grad_present_cnt,
                    }
                except Exception:
                    grad_counts = None

                # Optimizer coverage check on the first debug batch
                opt_cov = None
                try:
                    if batch_idx == 0:
                        opt_ids = {
                            id(p)
                            for g in optimizer.param_groups
                            for p in g.get("params", [])
                        }
                        req_ids = {id(p) for p in self.parameters() if p.requires_grad}
                        missing = len(req_ids - opt_ids)
                        extra = len(opt_ids - req_ids)
                        opt_cov = {
                            "trainable_not_in_opt": missing,
                            "opt_not_trainable": extra,
                        }
                except Exception:
                    opt_cov = None

                dbg = {
                    "batch_idx": batch_idx,
                    "grad_norm": float(total_norm) if total_norm is not None else None,
                    "unique_labels": uniq_labels,
                    "layer_grad_sums": layer_grad_sums,
                    "grad_counts": grad_counts,
                    "optimizer_coverage": opt_cov,
                }
            else:
                dbg = None

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

    def load_task_vector(self, task_vector, scaling_coef: float = 1.0):
        """Apply a task vector to the current encoder with a scaling coefficient.

        This mirrors the upstream task_vectors API where scaling is applied at application time.
        """
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if name in task_vector.vector:
                    param.data += scaling_coef * task_vector.vector[name]

    def _visualize_batch(self, images, preds, labels, title: str = "batch"):
        """Display images, predictions and labels for the first item in the batch.

        Expects images: [B, C, H, W] or [B, 1, H, W]; preds/labels: [B, 1, H, W] (class indices).
        """
        # Defer heavy imports to avoid overhead when visualization is disabled
        import matplotlib.pyplot as plt

        imgs = images.detach().cpu()
        p = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()

        # First item
        img = imgs[0]
        pred = p[0]
        lab = labels_cpu[0]

        # Squeeze channel dims
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.ndim == 3 and img.shape[0] > 1:
            img = img[0]

        if pred.ndim > 2:
            pred = pred.squeeze(0)
        if lab.ndim > 2:
            lab = lab.squeeze(0)

        _, axes = plt.subplots(1, 3, figsize=(12, 4))
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

    def evaluate(
        self,
        visualize: bool = False,
        profile: bool | str = False,
        profile_dir: str = "./outputs/profiling",
        # Evaluation performance knobs (opt-in)
        max_batches_per_split: Optional[int] = None,
        fast_metrics: bool = True,
        compute_hausdorff: bool = False,
    ):
        """
        Evaluate the model and return metrics on both train and test loaders.
        Memory-optimized version with aggressive memory management for large datasets like MMWHS.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        self.freeze()

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = None
        if compute_hausdorff:
            hausdorff_metric = HausdorffDistanceMetric(
                include_background=False,
                reduction="none",
                percentile=95,
            )

        results = {}

        # Configure TF32 and AMP dtype (same policy as training)
        try:
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
                if (
                    hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                ):
                    amp_dtype = torch.bfloat16
                else:
                    amp_dtype = torch.float16
            else:
                amp_dtype = None
        except Exception:
            amp_dtype = None

        for split in ["train", "val", "test"]:
            gc.collect()
            loader = getattr(self.dataset, f"{split}_loader", None)
            results[split] = {}
            if loader is None:
                continue

            has_labels = False
            viz_images = viz_preds = viz_labels = None

            if profile == "torch":
                from src.profiling import torch_profiler_ctx as _torch_profiler_ctx

                prof_cm = _torch_profiler_ctx(
                    name=f"eval-{split}",
                    out_dir=profile_dir,
                    wait=1,
                    warmup=1,
                    active=5,
                )
            else:
                prof_cm = contextlib.nullcontext()

            with torch.inference_mode(), prof_cm as prof:
                # Optional fast accumulator
                fast_accum = None
                if fast_metrics:
                    device_accum = device if device.type == "cuda" else "cpu"
                    fast_accum = {
                        "tp": torch.zeros(
                            self.num_classes, dtype=torch.float64, device=device_accum
                        ),
                        "fp": torch.zeros(
                            self.num_classes, dtype=torch.float64, device=device_accum
                        ),
                        "fn": torch.zeros(
                            self.num_classes, dtype=torch.float64, device=device_accum
                        ),
                    }
                for idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
                    if max_batches_per_split is not None and idx >= int(
                        max_batches_per_split
                    ):
                        break
                    if profile == "torch" and prof is not None:
                        try:
                            prof.step()
                        except Exception:
                            pass

                    images, labels = self._unpack_batch(batch)
                    if images is None:
                        continue
                    if hasattr(images, "as_tensor"):
                        images = images.as_tensor()
                    images = images.to(device, non_blocking=True)
                    if labels is not None and hasattr(labels, "as_tensor"):
                        labels = labels.as_tensor()
                    if labels is None:
                        del images
                        continue
                    labels = labels.to(device, non_blocking=True)
                    try:
                        labels = labels.long()
                    except Exception:
                        pass

                    has_labels = True

                    if device.type == "cuda" and amp_dtype is not None:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            outputs = self(images)
                    else:
                        outputs = self(images)

                    preds = torch.argmax(outputs, dim=1, keepdim=True)

                    if fast_metrics and fast_accum is not None:
                        try:
                            p = preds.squeeze(1).to(dtype=torch.int64)
                            y = labels.squeeze(1).to(dtype=torch.int64)
                            K = int(self.num_classes)
                            p = p.reshape(-1)
                            y = y.reshape(-1)
                            valid = (y >= 0) & (y < K)
                            if valid.any():
                                idxs = y[valid] * K + p[valid]
                                conf = torch.bincount(idxs, minlength=K * K)
                                conf = conf.view(K, K).to(dtype=torch.float64)
                                tp = torch.diag(conf)
                                fp = conf.sum(dim=0) - tp
                                fn = conf.sum(dim=1) - tp
                                fast_accum["tp"] += tp
                                fast_accum["fp"] += fp
                                fast_accum["fn"] += fn
                        except Exception:
                            pass
                    else:

                        def _to_onehot(
                            x: torch.Tensor, num_classes: int
                        ) -> torch.Tensor:
                            x = x.squeeze(1).long()
                            oh = (
                                F.one_hot(x, num_classes=num_classes)
                                .movedim(-1, 1)
                                .float()
                            )
                            return oh

                        try:
                            preds_oh = _to_onehot(preds, self.num_classes)
                            labels_oh = _to_onehot(labels, self.num_classes)
                            dice_metric(y_pred=preds_oh, y=labels_oh)
                            if compute_hausdorff and hausdorff_metric is not None:
                                hausdorff_metric(y_pred=preds_oh, y=labels_oh)
                        except Exception:
                            pass

                    if visualize and viz_images is None:
                        try:
                            viz_images = images[:1].detach().cpu()
                            viz_preds = preds[:1].detach().cpu()
                            viz_labels = labels[:1].detach().cpu()
                        except Exception:
                            viz_images = viz_preds = viz_labels = None

            if has_labels:
                try:
                    if fast_metrics and fast_accum is not None:
                        eps = 1e-8
                        tp = fast_accum["tp"].detach().cpu()
                        fp = fast_accum["fp"].detach().cpu()
                        fn = fast_accum["fn"].detach().cpu()
                        if tp.numel() > 1:
                            tp1, fp1, fn1 = tp[1:], fp[1:], fn[1:]
                            dice_score = float(
                                ((2.0 * tp1) / (2.0 * tp1 + fp1 + fn1 + eps))
                                .mean()
                                .item()
                            )
                        else:
                            dice_score = 0.0
                        hausdorff_dist = None
                    else:
                        dice_score = dice_metric.aggregate().item()
                        hausdorff_dist = None
                        if compute_hausdorff and hausdorff_metric is not None:
                            hd_vals = hausdorff_metric.aggregate()
                            if hasattr(hd_vals, "numel"):
                                mask = torch.isfinite(hd_vals)
                                if mask.any():
                                    hausdorff_dist = hd_vals[mask].mean().item()
                                else:
                                    hausdorff_dist = float("nan")
                            else:
                                hausdorff_dist = float(hd_vals)

                    results[split] = {"dice": dice_score, "hausdorff": hausdorff_dist}
                    print(
                        f"✅ {split} - Dice: {dice_score:.4f}, Hausdorff: {hausdorff_dist if hausdorff_dist is not None else 'N/A'}"
                    )
                    if visualize and viz_images is not None:
                        try:
                            self._visualize_batch(
                                viz_images,
                                viz_preds,
                                viz_labels,
                                title=f"Eval {split} summary",
                            )
                        except Exception as e:
                            print(
                                f"[DEBUG] Final visualization failed for {split}: {e}"
                            )
                except (ValueError, RuntimeError) as e:
                    print(f"⚠️ Error aggregating metrics for {split}: {e}")
                    results[split] = {"dice": None, "hausdorff": None}
            else:
                results[split] = {"dice": None, "hausdorff": None}

        self.unfreeze()
        return results
