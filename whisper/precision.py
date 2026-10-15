"""
Mixed-precision settings for training and evaluation.

Model weights and optimizer state are always kept in fp32; reduced precision is
only used for the forward and backward passes, via torch.autocast.
"""

import contextlib
from dataclasses import dataclass
from typing import Optional, Union

import torch

PRECISIONS = ("auto", "fp32", "fp16", "bf16")


@dataclass(frozen=True)
class Precision:
    name: str  # one of "fp32", "fp16", "bf16"
    device_type: str  # "cuda" or "cpu"

    @property
    def dtype(self) -> torch.dtype:
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.name]

    @property
    def use_grad_scaler(self) -> bool:
        # fp16 has a narrow exponent range, so small gradients underflow to zero
        # unless the loss is scaled up; bf16 has the same range as fp32.
        return self.name == "fp16"

    def autocast(self):
        if self.name == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def grad_scaler(self):
        enabled = self.use_grad_scaler
        try:
            return torch.amp.GradScaler(self.device_type, enabled=enabled)
        except (AttributeError, TypeError):  # torch < 2.3
            return torch.cuda.amp.GradScaler(enabled=enabled)


def bf16_supported(device_type: str) -> bool:
    if device_type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        # recent PyTorch versions report emulated (slow) bf16 support on older GPUs
        # such as the T4 unless including_emulation=False is passed
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.is_bf16_supported()


def resolve_precision(
    precision: str = "auto", device: Optional[Union[str, torch.device]] = None
) -> Precision:
    """
    Resolve a precision name to the settings to use on the given device.

    "auto" selects bf16 on GPUs that support it, fp16 on other GPUs, and fp32 on
    the CPU. Explicitly requesting a precision that the device cannot use raises
    a ValueError instead of silently falling back to another one.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {PRECISIONS}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = torch.device(device).type

    if precision == "auto":
        if device_type != "cuda":
            return Precision("fp32", device_type)
        return Precision("bf16" if bf16_supported(device_type) else "fp16", device_type)

    if device_type != "cuda" and precision != "fp32":
        raise ValueError(
            f"--precision {precision} requires a CUDA GPU; use --precision fp32 "
            f"on {device_type}"
        )
    if precision == "bf16" and not bf16_supported(device_type):
        name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else ""
        raise ValueError(
            f"bf16 is not supported on this GPU ({name}); bf16 requires compute "
            f"capability 8.0 or newer (e.g. A100, RTX 30xx/40xx, H100). "
            f"Use --precision fp16 instead."
        )
    return Precision(precision, device_type)
