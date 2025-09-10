"""
Adapted from https://github.com/mlfoundations/task_vectors/ by Ilalrco et al.
"""

import torch
from numbers import Number


class TaskVector:
    def __init__(
        self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                pretrained_state_dict = torch.load(
                    pretrained_checkpoint, map_location="cpu", weights_only=False
                ).state_dict()
                finetuned_state_dict = torch.load(
                    finetuned_checkpoint, map_location="cpu", weights_only=False
                ).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                try:
                    new_vector[key] = self.vector[key] + other.vector[key]
                except Exception as e:
                    print(f"Error adding key {key}: {e}")
                    continue
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __mul__(self, scalar):
        """Scale a task vector by a scalar (float or int)."""
        if not isinstance(scalar, Number):
            return NotImplemented
        with torch.no_grad():
            new_vector = {}
            for k, v in self.vector.items():
                try:
                    # Match dtype/device of the parameter delta to avoid promotion/move issues later
                    s = torch.as_tensor(scalar, dtype=v.dtype, device=v.device)
                    new_vector[k] = v * s
                except Exception:
                    # Fallback to plain Python scaling if tensor casting fails
                    new_vector[k] = v * scalar
        return TaskVector(vector=new_vector)

    def __rmul__(self, scalar):
        """Right-hand scalar multiplication to support scalar * TaskVector."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Divide a task vector by a scalar (float or int)."""
        if not isinstance(scalar, Number):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        with torch.no_grad():
            new_vector = {}
            for k, v in self.vector.items():
                try:
                    s = torch.as_tensor(scalar, dtype=v.dtype, device=v.device)
                    new_vector[k] = v / s
                except Exception:
                    new_vector[k] = v / scalar
        return TaskVector(vector=new_vector)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def __sub__(self, other):
        """Subtract one task vector from another."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                try:
                    new_vector[key] = self.vector[key] - other.vector[key]
                except Exception as e:
                    print(f"Error subtracting key {key}: {e}")
                    continue
        return TaskVector(vector=new_vector)

    def __getitem__(self, key):
        """Get an item from the task vector."""
        return self.vector[key]

    def keys(self):
        """Get the keys of the task vector."""
        return self.vector.keys()

    def items(self):
        """Get the items of the task vector."""
        return self.vector.items()

    def __len__(self):
        """Get the length of the task vector."""
        return len(self.vector)

    def __contains__(self, key):
        """Check if a key is in the task vector."""
        return key in self.vector

    def __delattr__(self, name):
        """Delete an attribute from the task vector."""
        if name in self.vector:
            del self.vector[name]
        else:
            raise AttributeError(f"{name} not found in task vector.")

    def __delitem__(self, key):
        """Delete an item from the task vector using del operator."""
        if key in self.vector:
            del self.vector[key]
        else:
            raise KeyError(f"{key} not found in task vector.")

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
