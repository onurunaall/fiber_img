from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from torch import nn

from model_UNET_EDSR import UNET
from utils import load_checkpoint


type ModelBuilder = Callable[[int, int], nn.Module]

COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def build_unet(in_channels: int, out_channels: int) -> nn.Module:
    return UNET(in_channels=in_channels, out_channels=out_channels, inplace=False)


# To add an architecture: write a builder function and register it here under a new name.
MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "unet": build_unet,
}


def available_architectures() -> list[str]:
    return sorted(MODEL_BUILDERS)


def build_model(architecture: str, in_channels: int, out_channels: int) -> nn.Module:
    if architecture not in MODEL_BUILDERS:
        known = ", ".join(available_architectures())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {known}")

    builder = MODEL_BUILDERS[architecture]
    return builder(in_channels, out_channels)


def load_trained_model(checkpoint_path: Path, architecture: str, n_colors: int) -> nn.Module:
    model = build_model(architecture, in_channels=n_colors, out_channels=n_colors)

    # weights_only=False: the file also stores 'args'. Only load checkpoint files you trust.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_checkpoint(checkpoint, model)
    return model.eval()


def compile_model(model: nn.Module, mode: str) -> nn.Module:
    # torch.compile returns an OptimizedModule (an nn.Module), but its type hint only says "callable".
    return cast(nn.Module, torch.compile(model, mode=mode))
