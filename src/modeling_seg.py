"""
Generated by GPT-4.1 converting modeling.py to segmentation
"""

import open_clip
import torch
import torch.nn.functional as F
# from sklearn.metrics import jaccard_score
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from src import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, model, keep_lang=False, cache_dir=None, openclip_cachedir=None):
        super().__init__()
        print(f"Loading {model} pre-trained weights.")
        if "__pretrained__" in model:
            name, pretrained = model.split("__pretrained__")
        else:
            name = model
            pretrained = "openai"
        self.model, self.train_preprocess, self.val_preprocess = (
            open_clip.create_model_and_transforms(
                name, pretrained=pretrained, cache_dir=openclip_cachedir
            )
        )
        self.cache_dir = cache_dir
        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        # For segmentation, we want feature maps, not pooled features
        if hasattr(self.model.visual, "forward_features"):
            features = self.model.visual.forward_features(images)
        else:
            features = self.model.encode_image(images)
        return features

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)


# class SegmentationHead(torch.nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         # Simple 1x1 conv for per-pixel classification
#         assert (
#             num_classes > 1
#         ), "num_classes must be set to the number of classes in your dataset"
#         self.conv = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

#     def forward(self, features):
#         # features: (B, C, H, W)
#         return self.conv(features)

#     def __call__(self, features):
#         return self.forward(features)

#     def save(self, filename):
#         print(f"Saving segmentation head to {filename}")
#         utils.torch_save(self, filename)

#     @classmethod
#     def load(cls, filename):
#         print(f"Loading segmentation head from {filename}")
#         return utils.torch_load(filename)


class ImageSegmenter(torch.nn.Module):
    def __init__(self, image_encoder, segmentation_head, dataset=None):
        super().__init__()
        self.dataset = dataset
        self.image_encoder = image_encoder
        self.segmentation_head = segmentation_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess
        # Check that segmentation head output matches dataset classes if possible
        if dataset is not None and hasattr(dataset, "num_classes"):
            # Try to get out_channels from the last module if segmentation_head is Sequential
            if hasattr(self.segmentation_head, "out_channels"):
                out_channels = self.segmentation_head.out_channels
            elif isinstance(self.segmentation_head, torch.nn.Sequential):
                last_module = list(self.segmentation_head.children())[-1]
                out_channels = getattr(last_module, "out_channels", None)
                if out_channels is None:
                    raise AttributeError(
                        "The last module in segmentation_head does not have an 'out_channels' attribute."
                    )
            else:
                raise AttributeError(
                    "segmentation_head does not have an 'out_channels' attribute."
                )
            assert (
                out_channels == dataset.num_classes
            ), f"Segmentation head output channels ({out_channels}) do not match dataset.num_classes ({dataset.num_classes})"
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        # If we have an image encoder, use it; otherwise pass inputs directly to segmentation head
        if self.image_encoder is not None:
            features = self._encode_features(inputs)
        else:
            # No encoder, use inputs directly
            features = inputs

        outputs = self.segmentation_head(features)
        outputs = self._resize_outputs(outputs, inputs)
        return outputs

    def _encode_features(self, inputs):
        """Encode input through image encoder and handle dimension mismatches."""
        features = self.image_encoder(inputs)

        if features.dim() == 2:
            features = self._expand_2d_features(features, inputs)
        elif features.dim() == 3 and inputs.dim() == 5:
            features = self._handle_sequence_features(features)

        return features

    def _expand_2d_features(self, features, inputs):
        """Expand 2D features to match input dimensions."""
        if inputs.dim() == 5:  # 3D input (B, C, D, H, W)
            return features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:  # 2D input (B, C, H, W)
            return features.unsqueeze(-1).unsqueeze(-1)

    def _handle_sequence_features(self, features):
        """Handle sequence outputs from vision transformers for 3D."""
        # (B, N, C) -> global average pooling
        features = features.mean(dim=1, keepdim=True)  # (B, 1, C)
        return features.transpose(1, 2)  # (B, C, 1)

    def _resize_outputs(self, outputs, inputs):
        """Resize outputs to match input spatial dimensions."""
        if inputs.dim() == 5 and outputs.shape[-3:] != inputs.shape[-3:]:
            # 3D case
            return F.interpolate(
                outputs, size=inputs.shape[-3:], mode="trilinear", align_corners=False
            )
        elif inputs.dim() == 4 and outputs.shape[-2:] != inputs.shape[-2:]:
            # 2D case
            return F.interpolate(
                outputs, size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )
        return outputs

    def freeze_head(self):
        for param in self.segmentation_head.parameters():
            param.requires_grad = False

    def freeze(self):
        """
        Freeze the image encoder and segmentation head.
        """
        self.freeze_head()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image segmenter to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image segmenter from {filename}")
        return utils.torch_load(filename)

    def evaluate(self):
        """
        Evaluate the segmenter on the dataset.
        """
        if self.dataset is None:
            raise ValueError("Dataset not provided for evaluation.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        self.freeze()

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95
        )

        # Aggregate metrics
        # dice_score = dice_metric.aggregate().item()
        # hausdorff_dist = hausdorff_metric.aggregate().item()

        # Reset metrics for potential use
        dice_metric.reset()
        hausdorff_metric.reset()

        with torch.no_grad():
            for batch in self.dataset.train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                # Ensure images have batch and channel dimensions
                if images.dim() == 2:
                    images = images.unsqueeze(0).unsqueeze(0)
                elif images.dim() == 3:
                    images = images.unsqueeze(1)
                # Ensure images have 3 channels (e.g., RGB)
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                outputs = self(images)
                # Apply argmax to get discrete segmentation
                preds = torch.argmax(outputs, dim=1, keepdim=True)

                # Compute metrics
                dice_metric(y_pred=preds, y=labels)
                hausdorff_metric(y_pred=preds, y=labels)

        train_dice_score = dice_metric.aggregate().item()
        train_hausdorff_dist = hausdorff_metric.aggregate().item()

        dice_metric.reset()
        hausdorff_metric.reset()

        with torch.no_grad():
            for batch in self.dataset.test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=self,
                )

                # Apply argmax to get discrete segmentation
                preds = torch.argmax(outputs, dim=1, keepdim=True)

                # Compute metrics
                dice_metric(y_pred=preds, y=labels)
                hausdorff_metric(y_pred=preds, y=labels)

        test_dice_score = dice_metric.aggregate().item()
        test_hausdorff_dist = hausdorff_metric.aggregate().item()

        # Reset metrics for potential use
        dice_metric.reset()
        hausdorff_metric.reset()

        return {
            "train_dice_score": train_dice_score,
            "train_hausdorff_dist": train_hausdorff_dist,
            "test_dice_score": test_dice_score,
            "test_hausdorff_dist": test_hausdorff_dist,
        }

    def finetune(self, epochs=100, save_path=None):
        """
        Finetune the segmentation head while keeping the encoder frozen.
        """
        if self.dataset is None:
            raise ValueError("Dataset not provided for finetuning.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        # Only train the segmentation head
        if self.image_encoder is not None:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Optimizer for segmentation head only
        optimizer = torch.optim.Adam(self.segmentation_head.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Finetuning for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.dataset.train_loader:
                try:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)

                    # Handle different input dimensions
                    if images.dim() == 3:
                        images = images.unsqueeze(0).unsqueeze(0)
                    elif images.dim() == 4:
                        images = images.unsqueeze(0)

                    optimizer.zero_grad()
                    outputs = self(images)

                    # Ensure labels have the right shape
                    if labels.dim() == outputs.dim():
                        labels = labels.squeeze(
                            1
                        )  # Remove channel dimension for labels

                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"Skipping batch due to error: {e}")
                    continue

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        if save_path:
            self.save(save_path)

        return self

    def load_task_vector(self, task_vector):
        """
        Apply a task vector to the model.
        """
        with torch.no_grad():
            current_state_dict = self.state_dict()
            new_state_dict = {}

            for key in current_state_dict:
                if key in task_vector.vector:
                    new_state_dict[key] = (
                        current_state_dict[key] + task_vector.vector[key]
                    )
                else:
                    new_state_dict[key] = current_state_dict[key]

            self.load_state_dict(new_state_dict, strict=False)
