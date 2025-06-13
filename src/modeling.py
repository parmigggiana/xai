"""
Adapted from https://github.com/mlfoundations/task_vectors/ by Ilalrco et al.
"""

import open_clip
import torch

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
        return self.model.encode_image(images)

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

    # @classmethod
    # def load_from_state_dict(cls, model_name, state_dict):
    #     self.model, self.train_preprocess, self.val_preprocess = (
    #         open_clip.create_model_and_transforms(
    #             name, pretrained=pretrained, cache_dir=args.openclip_cachedir
    #         )
    #     )
    #     self.model.load_from_state_dict(state_dict)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


# def get_baseline_encoder(base_path: str = "", depth: int = 10) -> torch.nn.Module:
#     """
#     Get the baseline encoder for the model.

#     Args:
#         base_path (str): Base path to load the encoder from.

#     Returns:
#         torch.nn.Module: Baseline encoder.
#     """

#     encoder_path = Path(base_path) / f"baseline_{depth}.pth"
#     encoder = ResNet(depth)

#     if encoder_path.exists():
#         state_dict = torch.load(encoder_path, map_location="cpu")
#         encoder.load_state_dict(state_dict)
#     else:
#         state_dict = get_pretrained_resnet_medicalnet(depth)
#         encoder.load_state_dict(state_dict)
#         Path(base_path).mkdir(parents=True, exist_ok=True)
#         torch.save(encoder.state_dict(), encoder_path)
#     return encoder
