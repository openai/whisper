"""
Low-Rank Adaptation (LoRA) for Whisper, implemented in plain PyTorch.

LoRA freezes the pretrained weights and learns a low-rank update for selected
linear layers: ``W x + (alpha / r) * B A x``, where ``A`` has shape ``(r, in)``
and ``B`` has shape ``(out, r)``. ``B`` is initialized to zero, so a freshly
wrapped model produces exactly the same outputs as the original model.

See https://arxiv.org/abs/2106.09685
"""

import math
import warnings
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import Linear, MultiHeadAttention, ResidualAttentionBlock, Whisper

LORA_FORMAT = "whisper-lora"
LORA_FORMAT_VERSION = 1

ATTENTION_TARGETS = ("query", "key", "value", "out")
ALL_TARGETS = ATTENTION_TARGETS + ("mlp",)
ALL_PARTS = ("encoder", "decoder")


class LoRALinear(nn.Module):
    """Wraps a linear layer with a trainable low-rank update; the base layer is frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout=0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = base.in_features
        self.out_features = base.out_features

        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA weights are always kept in fp32 (master weights), regardless of the
        # dtype of the base layer; they are cast to the input dtype in forward().
        device = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(r, self.in_features, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, r, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.base(x)
        update = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
        update = F.linear(update, self.lora_B.to(x.dtype))
        return result + update * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return (self.base.weight.float() + delta).to(self.base.weight.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}"
        )


def _normalize(
    values: Union[str, Iterable[str]], allowed: Iterable[str], name: str
) -> List[str]:
    if isinstance(values, str):
        values = values.split(",")
    values = [v.strip() for v in values if v.strip()]
    if values == ["all"]:
        return list(allowed)
    unknown = [v for v in values if v not in allowed]
    if unknown or not values:
        raise ValueError(
            f"Invalid {name} {unknown or values}; choose from {list(allowed)} or 'all'"
        )
    return list(dict.fromkeys(values))


def _target_linears(block: ResidualAttentionBlock, targets: List[str]):
    """Yields (parent module, attribute name) pairs for the targeted linear layers"""
    for attn in (block.attn, block.cross_attn):
        if isinstance(attn, MultiHeadAttention):
            for name in ATTENTION_TARGETS:
                if name in targets:
                    yield attn, name
    if "mlp" in targets:
        for index, layer in enumerate(block.mlp):
            if isinstance(layer, (nn.Linear, LoRALinear)):
                yield block.mlp, str(index)


def apply_lora(
    model: Whisper,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
    target_modules: Union[str, Iterable[str]] = ("query", "value"),
    apply_to: Union[str, Iterable[str]] = ("encoder", "decoder"),
) -> Whisper:
    """
    Wrap the selected linear layers of a Whisper model with LoRALinear, in place.

    Parameters
    ----------
    model: Whisper
        The model to adapt
    r: int
        The rank of the low-rank update
    alpha: float
        The scaling numerator; the update is scaled by ``alpha / r``
    dropout: float
        Dropout applied to the input of the LoRA branch
    target_modules: Iterable[str]
        Any of "query", "key", "value", "out" (applied to self- and cross-attention)
        and "mlp" (both MLP layers), or "all"
    apply_to: Iterable[str]
        "encoder", "decoder", or both

    Returns
    -------
    model : Whisper
        The same model, with LoRA layers added and only the LoRA parameters trainable
    """
    targets = _normalize(target_modules, ALL_TARGETS, "target_modules")
    parts = _normalize(apply_to, ALL_PARTS, "apply_to")

    for part in parts:
        for block in getattr(model, part).blocks:
            for parent, name in _target_linears(block, targets):
                layer = getattr(parent, name)
                if isinstance(layer, LoRALinear):
                    raise RuntimeError("LoRA has already been applied to this model")
                lora_layer = LoRALinear(layer, r, alpha, dropout)
                setattr(parent, name, lora_layer.train(model.training))

    model.lora_config = dict(
        r=r, alpha=alpha, dropout=dropout, target_modules=targets, apply_to=parts
    )
    mark_only_lora_as_trainable(model)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def lora_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    """Returns only the LoRA weights, as fp32 tensors on the CPU"""
    return {
        name: param.detach().float().cpu()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def save_lora(model: Whisper, path: str, base_model: Optional[str] = None) -> None:
    """Save the LoRA adapter (weights and config) of a model to a file"""
    if not has_lora(model):
        raise ValueError("The model has no LoRA layers; call apply_lora() first")
    config = dict(getattr(model, "lora_config", {}))
    if base_model is not None:
        config["base_model"] = base_model
    torch.save(
        {
            "format": LORA_FORMAT,
            "version": LORA_FORMAT_VERSION,
            "config": config,
            "dims": asdict(model.dims),
            "state_dict": lora_state_dict(model),
        },
        path,
    )


def load_lora(
    model: Whisper, path: Union[str, dict], base_model: Optional[str] = None
) -> Whisper:
    """
    Apply a saved LoRA adapter to a model, in place.

    If the model does not have LoRA layers yet, they are created using the
    configuration stored in the adapter file.

    Parameters
    ----------
    model: Whisper
        The base model the adapter was trained on
    path: Union[str, dict]
        Path to an adapter file written by `save_lora()`, or its loaded contents
    base_model: Optional[str]
        The name of the base model, used to warn about mismatches
    """
    if isinstance(path, dict):
        adapter = path
    else:
        kwargs = {"weights_only": True} if torch.__version__ >= "1.13" else {}
        adapter = torch.load(path, map_location="cpu", **kwargs)

    if not isinstance(adapter, dict) or adapter.get("format") != LORA_FORMAT:
        raise ValueError(f"{path} is not a Whisper LoRA adapter file")

    config = adapter["config"]
    trained_on = config.get("base_model")
    if adapter["dims"] != asdict(model.dims):
        raise ValueError(
            f"LoRA adapter was trained on a different base model "
            f"({trained_on or 'unknown'}); its dimensions {adapter['dims']} do not "
            f"match the loaded model's {asdict(model.dims)}"
        )
    if trained_on and base_model and trained_on != base_model:
        warnings.warn(
            f"LoRA adapter was trained on '{trained_on}' but is being loaded into "
            f"'{base_model}'; the dimensions match but the results may be poor"
        )

    if not has_lora(model):
        apply_lora(
            model,
            r=config["r"],
            alpha=config["alpha"],
            dropout=config.get("dropout", 0.0),
            target_modules=config["target_modules"],
            apply_to=config["apply_to"],
        )

    state_dict = adapter["state_dict"]
    expected = {name for name, _ in model.named_parameters() if "lora_" in name}
    missing = sorted(expected - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"LoRA adapter does not match the model's LoRA layers; "
            f"missing keys: {missing[:5]}, unexpected keys: {unexpected[:5]}"
        )
    model.load_state_dict(state_dict, strict=False)
    model.lora_config = dict(config)
    return model


@torch.no_grad()
def merge_lora(model: Whisper) -> Whisper:
    """
    Fold the LoRA updates into the base weights and replace every LoRALinear with
    a plain Linear layer, in place. The result is a standard Whisper model that
    can be saved as a regular checkpoint.
    """
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                merged = Linear(
                    base.in_features,
                    base.out_features,
                    bias=base.bias is not None,
                    device=base.weight.device,
                    dtype=base.weight.dtype,
                )
                merged.weight.copy_(child.merged_weight())
                if base.bias is not None:
                    merged.bias.copy_(base.bias)
                setattr(module, name, merged)

    if hasattr(model, "lora_config"):
        del model.lora_config
    for param in model.parameters():
        param.requires_grad = True
    return model
