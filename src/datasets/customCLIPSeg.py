import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Dict, Optional
from clipseg.clipseg import CLIPDensePredT


class CustomCLIPSeg(nn.Module):
    """
    Wrapper for CLIPDensePredT that provides unified segmentation maps for multiple classes.

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
        threshold: Probability threshold for binary segmentation
        **kwargs: Additional arguments for CLIPDensePredT
    """

    def __init__(
        self,
        classes: List[str],
        version: str = "ViT-B/16",
        aggregation_mode: str = "argmax",
        class_weights: Optional[List[float]] = None,
        background_class: bool = True,
        threshold: float = 0.5,
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
        self.threshold = threshold

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
            "reduce_dim": 128,
            "prompt": "shuffle+",
            "complex_trans_conv": True,
        }
        default_kwargs.update(kwargs)

        self.clipseg = CLIPDensePredT(version=version, **default_kwargs)

        # Setup class mapping
        self.num_classes = len(self.classes) + (1 if self.background_class else 0)
        self.class_to_idx = {
            cls: i + (1 if self.background_class else 0)
            for i, cls in enumerate(self.classes)
        }
        if self.background_class:
            self.class_to_idx["background"] = 0

        # Create segmentation head - contains all layers responsible for final prediction
        self.head = nn.ModuleDict(
            {
                # Core CLIPSeg segmentation components
                "clip_visual": self.clipseg.model,  # CLIP visual encoder
                "reduces": self.clipseg.reduces,  # Feature reduction layers
                "blocks": self.clipseg.blocks,  # Transformer encoder blocks
                "extra_blocks": self.clipseg.extra_blocks,  # Additional transformer blocks
                "trans_conv": self.clipseg.trans_conv,  # Main segmentation head (transposed conv)
                # Text conditioning components
                "film_mul": self.clipseg.film_mul,  # FiLM multiplicative conditioning
                "film_add": self.clipseg.film_add,  # FiLM additive conditioning
            }
        )

        # Add optional components if they exist
        if (
            hasattr(self.clipseg, "reduce_cond")
            and self.clipseg.reduce_cond is not None
        ):
            self.head["reduce_cond"] = self.clipseg.reduce_cond

        if (
            hasattr(self.clipseg, "upsample_proj")
            and self.clipseg.upsample_proj is not None
        ):
            self.head["upsample_proj"] = self.clipseg.upsample_proj

    def forward(
        self,
        image: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            image: Input image tensor (B, C, H, W)
            return_individual: Whether to return individual class predictions
            return_probabilities: Whether to return probabilities instead of class indices

        Returns:
            Segmentation map (B, len(classes), H, W)
        """
        height, width = image.shape[2], image.shape[3]

        # Get predictions for each class
        class_predictions = {}
        all_predictions = []

        for cls in self.classes:
            with torch.no_grad():
                pred = self.clipseg(image, cls)[0]  # (B, 1, H, W)
                pred = torch.sigmoid(pred)  # Convert to probabilities

                # Resize to original image size if needed
                if pred.shape[2:] != (height, width):
                    pred = F.interpolate(
                        pred, size=(height, width), mode="bilinear", align_corners=False
                    )

                class_predictions[cls] = pred
                all_predictions.append(pred)

        # Stack all predictions (B, num_classes, H, W)
        all_predictions = torch.cat(all_predictions, dim=1)

        return all_predictions

    def predict_single_class(
        self, image: torch.Tensor, class_name: str
    ) -> torch.Tensor:
        """Get segmentation for a single class."""
        if class_name not in self.classes:
            raise ValueError(
                f"Class '{class_name}' not in initialized classes: {self.classes}"
            )

        with torch.no_grad():
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

    def get_head_parameters(self):
        """Get all parameters from the segmentation head."""
        return self.head.parameters()

    def get_head_named_parameters(self):
        """Get all named parameters from the segmentation head."""
        return self.head.named_parameters()

    def freeze_backbone(self):
        """Freeze the CLIP visual backbone, keeping only head trainable."""
        for param in self.head["clip_visual"].parameters():
            param.requires_grad = False
        print("Frozen CLIP visual backbone. Only segmentation head is trainable.")

    def unfreeze_backbone(self):
        """Unfreeze the CLIP visual backbone."""
        for param in self.head["clip_visual"].parameters():
            param.requires_grad = True
        print("Unfrozen CLIP visual backbone. All parameters are trainable.")

    def get_head_layer_names(self) -> List[str]:
        """Get names of all layers in the segmentation head."""
        return list(self.head.keys())

    def print_head_summary(self):
        """Print a summary of the segmentation head architecture."""
        print("\n" + "=" * 50)
        print("SEGMENTATION HEAD ARCHITECTURE")
        print("=" * 50)

        total_params = 0
        trainable_params = 0

        for name, module in self.head.items():
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

            print(
                f"{name:15} | {str(type(module).__name__):20} | "
                f"Params: {module_params:8,} | Trainable: {module_trainable:8,}"
            )

            total_params += module_params
            trainable_params += module_trainable

        print("-" * 50)
        print(
            f"{'TOTAL':15} | {'':20} | "
            f"Params: {total_params:8,} | Trainable: {trainable_params:8,}"
        )
        print("=" * 50)


def create_chaos_ct_clipseg(**kwargs) -> CustomCLIPSeg:
    """Create CLIPSeg wrapper for CHAOS CT dataset classes."""
    chaos_ct_classes = ["liver", "kidney_right", "kidney_left", "spleen"]

    # Default weights favoring organ detection (background handled separately)
    default_kwargs = {
        "aggregation_mode": "argmax",
        "background_class": True,
        "threshold": 0.3,
        "version": "ViT-B/16",
    }
    default_kwargs.update(kwargs)

    return CustomCLIPSeg(classes=chaos_ct_classes, **default_kwargs)
