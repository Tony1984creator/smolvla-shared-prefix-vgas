"""Inference-only weight storage. Arithmetic follows the source critic dtype."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class PackedWeight(nn.Module):
    def __init__(self, storage, dtype):
        super().__init__()
        self.storage = storage
        self.dtype = dtype

    def right_inverse(self, value):
        if self.storage != "int8":
            return (value.detach().to(getattr(torch, self.storage)),)
        dims = tuple(range(1, value.ndim))
        scale = value.abs().amax(dim=dims, keepdim=True) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return (value / scale).round().clamp(-127, 127).to(torch.int8), scale

    def forward(self, packed, scale=None):
        value = packed.to(self.dtype)
        return value if scale is None else value * scale


def pack_weights(model, storage):
    if storage not in {"int8", "float16", "bfloat16"}:
        raise ValueError(storage)
    model.requires_grad_(False)
    for module in list(model.modules()):
        for name, weight in list(module.named_parameters(recurse=False)):
            if weight.ndim >= 2 and weight.is_floating_point():
                parametrize.register_parametrization(
                    module, name, PackedWeight(storage, weight.dtype), unsafe=True
                )
    return model


def storage_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())
