import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from clipseg.clipseg import CLIPDensePredT as _CLIPDensePredT
from clipseg.clipseg import forward_multihead_attention
from torch.nn import functional as nnf


class GradCLIPDensePredT(_CLIPDensePredT):
    """CLIPDensePredT variant that allows gradients through visual_forward.

    This mirrors the library behavior but removes the torch.no_grad() wrapper
    so the visual tower participates in backprop.
    """

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None):
        # Copied structure: call the parent implementation but without no_grad.

        inp_size = x_inp.shape[2:]

        if self.n_tokens is not None:
            stride2 = x_inp.shape[2] // self.n_tokens
            conv_weight2 = nnf.interpolate(
                self.model.conv1.weight,
                (stride2, stride2),
                mode="bilinear",
                align_corners=True,
            )
            x = nnf.conv2d(
                x_inp,
                conv_weight2,
                bias=self.model.conv1.bias,
                stride=stride2,
                dilation=self.model.conv1.dilation,
            )
        else:
            x = self.model.conv1(x_inp)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [
                self.model.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        standard_n_tokens = 50 if self.model.conv1.kernel_size[0] == 32 else 197

        if x.shape[1] != standard_n_tokens:
            new_shape = int(math.sqrt(x.shape[1] - 1))
            x = (
                x
                + self.rescaled_pos_emb((new_shape, new_shape)).to(x.dtype)[None, :, :]
            )
        else:
            x = x + self.model.positional_embedding.to(x.dtype)

        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        activations, affinities = [], []
        for i, res_block in enumerate(self.model.transformer.resblocks):

            if mask is not None:
                mask_layer, mask_type, mask_tensor = mask
                if mask_layer == i or mask_layer == "all":
                    # import ipdb; ipdb.set_trace()
                    size = int(math.sqrt(x.shape[0] - 1))

                    attn_mask = (
                        mask_type,
                        nnf.interpolate(
                            mask_tensor.unsqueeze(1).float(), (size, size)
                        ).view(mask_tensor.shape[0], size * size),
                    )

                else:
                    attn_mask = None
            else:
                attn_mask = None

            x, aff_per_head = forward_multihead_attention(
                x, res_block, with_aff=True, attn_mask=attn_mask
            )

            if i in extract_layers:
                affinities += [aff_per_head]

                # if self.n_tokens is not None:
                #    activations += [nnf.interpolate(x, inp_size, mode='bilinear', align_corners=True)]
                # else:
                activations += [x]

            if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                print("early skip")
                break

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_post(x[:, 0, :])

        if self.model.proj is not None:
            x = x @ self.model.proj

        return x, activations, affinities


class CLIPSeg(nn.Module):
    """
    Wrapper for CLIPDensePredT that provides unified segmentation maps for multiple classes.

    This wrapper allows overriding CLIPSeg's default prompt templates with medical-specific templates.

    Args:
        classes: List of class names to segment
        version: CLIP model version ('ViT-B/16' or 'ViT-B/32')
        aggregation_mode: How to combine multiple class predictions
            - 'argmax': Take class with highest probability per pixel
            - 'max': Take maximum probability across classes
            - 'mean': Average all class probabilities
            - 'weighted': Weighted average (requires class_weights)
        class_weights: Optional weights for each class (for weighted aggregation)
        background_class: Whether to include background class (index 0)
        medical_templates: Optional dict mapping class names to template functions
        dataset_info: Tuple of (dataset_name, domain) for automatic template selection
        **kwargs: Additional arguments for CLIPDensePredT
    """

    def __init__(
        self,
        classes: List[str],
        version: str = "ViT-B/16",
        aggregation_mode: str = "argmax",
        class_weights: Optional[List[float]] = None,
        background_class: bool = True,
        medical_templates: Optional[List] = None,
        dataset_info: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__()

        # Handle explicit background class in classes list
        self.explicit_background = any(
            cls.lower() in ["background", "bg"] for cls in classes
        )

        if self.explicit_background:
            # Remove background from classes list and handle it separately
            self.classes = [
                cls for cls in classes if cls.lower() not in ["background", "bg"]
            ]
            self.background_class = True  # Force background_class to True
            print("Found explicit background class in input. Treating it separately.")
            print(f"Non-background classes: {self.classes}")
        else:
            self.classes = classes
            self.background_class = background_class

        self.aggregation_mode = aggregation_mode

        # Store medical templates for better prompt generation
        self.medical_templates = medical_templates
        self.dataset_info = dataset_info

        # Setup class weights - adjust for removed background if needed
        if class_weights is not None:
            if self.explicit_background and len(class_weights) == len(classes):
                # Remove background weight if it was included
                background_indices = [
                    i
                    for i, cls in enumerate(classes)
                    if cls.lower() in ["background", "bg"]
                ]
                if background_indices:
                    # Remove background weight(s)
                    class_weights = [
                        w
                        for i, w in enumerate(class_weights)
                        if i not in background_indices
                    ]
                    print(
                        f"Removed background class weight. Remaining weights: {class_weights}"
                    )

            assert len(class_weights) == len(
                self.classes
            ), f"Number of weights ({len(class_weights)}) must match number of non-background classes ({len(self.classes)})"
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

        # Initialize CLIPSeg model
        default_kwargs = {
            "extract_layers": (3, 6, 9),
            "reduce_dim": 64,
            # "complex_trans_conv": True,
        }
        default_kwargs.update(kwargs)

        # Use the grad-enabled variant so visual features get gradients during finetuning
        self.clipseg = GradCLIPDensePredT(version=version, **default_kwargs)

        # CPU-side caches to avoid recomputing/retokenizing text each step (always enabled)
        self._avg_prompt_embed_cpu = {}
        self._class_text_embed_cpu = {}

        # Setup class mapping
        self.num_classes = len(self.classes) + (1 if self.background_class else 0)
        self.class_to_idx = {
            cls: i + (1 if self.background_class else 0)
            for i, cls in enumerate(self.classes)
        }
        if self.background_class:
            self.class_to_idx["background"] = 0

    def generate_medical_prompts(self, class_name: str) -> List[str]:
        """
        Generate medical-specific prompts for a given class.

        Uses the medical templates if provided, otherwise falls back to
        automatic template selection based on dataset_info.
        """
        if self.medical_templates:
            # Use provided medical templates
            return [template(class_name) for template in self.medical_templates]

        if self.dataset_info:
            # Import templates from head.py
            from src.datasets.templates import dataset_to_template

            dataset_name, domain = self.dataset_info

            if (dataset_name, domain) in dataset_to_template:
                templates = dataset_to_template[(dataset_name, domain)]
                return [template(class_name) for template in templates]

        # Fallback to basic medical templates if nothing specific is available
        fallback_templates = [
            lambda c: f"medical image showing {c}.",
            lambda c: f"medical scan of {c}.",
            lambda c: f"anatomical structure {c}.",
            lambda c: f"medical imaging of {c}.",
            lambda c: f"radiological image showing {c}.",
        ]
        print("WARNING: Using fallback medical templates for class:", class_name)
        return [template(class_name) for template in fallback_templates]

    def _get_avg_prompt_embedding(
        self, class_name: str, device: torch.device
    ) -> torch.Tensor:
        """Return averaged medical prompt embedding for a class, with CPU caching.

        - Uses `generate_medical_prompts` to build a set of prompts for the class.
        - Computes text embeddings with `self.clipseg.compute_conditional`.
        - Caches the averaged embedding on CPU and moves it to the requested device on use.
        """
        # Return cached CPU embedding moved to target device
        cached = self._avg_prompt_embed_cpu.get(class_name)
        if cached is not None:
            return cached.to(device)

        prompts = self.generate_medical_prompts(class_name)
        embeds: List[torch.Tensor] = []
        for prompt in prompts:
            emb = self.clipseg.compute_conditional(prompt)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            emb = emb.detach().cpu().float()
            # Flatten to 1D if needed (implementation may return [1, D] or [D])
            embeds.append(emb.view(-1))

        # Average and keep as [1, D] for conditioning
        avg = torch.stack(embeds, dim=0).mean(dim=0, keepdim=True).contiguous()
        self._avg_prompt_embed_cpu[class_name] = avg  # cache on CPU
        return avg.to(device)

    def predict_single_class_with_medical_prompts(
        self, image: torch.Tensor, class_name: str
    ) -> torch.Tensor:
        """
        Get segmentation for a single class using medical-specific prompts.
        """
        if class_name not in self.classes:
            raise ValueError(
                f"Class '{class_name}' not in initialized classes: {self.classes}"
            )

        # Get (cached) averaged embedding for this class
        avg_embedding = self._get_avg_prompt_embedding(class_name, device=image.device)
        # Run the segmentation forward pass and convert logits to probabilities
        pred = self.clipseg(image, conditional=avg_embedding)[0]

        # Resize if needed
        if pred.shape[2:] != image.shape[2:]:
            pred = F.interpolate(
                pred, size=image.shape[2:], mode="bilinear", align_corners=False
            )

        return pred

    def forward(
        self,
        image: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            image: Input image tensor (B, C, H, W)

        Returns:
            Segmentation map (B, num_classes, H, W) where first channel is background
        """
        height, width = image.shape[2], image.shape[3]
        foreground_predictions = []

        for cls in self.classes:
            if self.medical_templates or self.dataset_info:
                # Use cached medical prompt embedding
                pred = self.predict_single_class_with_medical_prompts(image, cls)
            else:
                # Fallback to original CLIPSeg behavior
                pred = self.clipseg(image, cls)[0]  # (B, 1, H, W)

            pred = torch.sigmoid(pred)  # Convert to probabilities

            # Resize to original image size if needed
            if pred.shape[2:] != (height, width):
                pred = F.interpolate(
                    pred, size=(height, width), mode="bilinear", align_corners=False
                )

            foreground_predictions.append(pred)

        # Stack foreground predictions (B, num_foreground_classes, H, W)
        foreground_stack = torch.cat(foreground_predictions, dim=1)

        # Compute background as 1 - max(foreground_classes)
        # Background probability is high where no foreground class is confident
        max_foreground, _ = torch.max(
            foreground_stack, dim=1, keepdim=True
        )  # (B, 1, H, W)
        background_pred = 1.0 - max_foreground  # (B, 1, H, W)

        # Combine background (first) + foreground predictions
        all_predictions = torch.cat(
            [background_pred, foreground_stack], dim=1
        )  # (B, num_classes, H, W)

        return all_predictions

    def predict_single_class(
        self, image: torch.Tensor, class_name: str
    ) -> torch.Tensor:
        """Get segmentation for a single class."""
        if class_name not in self.classes:
            raise ValueError(
                f"Class '{class_name}' not in initialized classes: {self.classes}"
            )

        pred = self.clipseg(image, class_name)[0]
        pred = torch.sigmoid(pred)

        # Resize if needed
        if pred.shape[2:] != image.shape[2:]:
            pred = F.interpolate(
                pred, size=image.shape[2:], mode="bilinear", align_corners=False
            )

        return pred

    def get_class_index(self, class_name: str) -> int:
        """Get the index for a given class name."""
        return self.class_to_idx.get(class_name, -1)

    def get_class_names(self) -> List[str]:
        """Get list of all class names including background if enabled."""
        if self.background_class:
            return ["background"] + self.classes
        return self.classes

    def parameters(self, recurse=True):
        # include both the internal CLIPDensePredT params and the custom head params
        return self.clipseg.parameters(recurse=recurse)
