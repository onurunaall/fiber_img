import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from model_SGARNet import SGARNet
from model_UNET_EDSR import UNET
from utils import load_checkpoint


type ModelBuilder = Callable[..., nn.Module]

COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def build_unet(in_channels: int, out_channels: int) -> nn.Module:
    return UNET(in_channels=in_channels, out_channels=out_channels, inplace=False)


def build_sgarnet(in_channels: int, out_channels: int, lattice_period_px: float, lattice_angle_deg: float) -> nn.Module:
    # SGARNet adds its input to its output, so it needs as many output channels as input channels.
    if in_channels != out_channels:
        raise ValueError(f"SGARNet needs in_channels == out_channels, got {in_channels} and {out_channels}")

    # Network and gate settings of options/train/MCFArtifactFree.yml in https://github.com/THUHoloLab/SGARNet
    return SGARNet(
        img_channel=in_channels,
        width=32,
        enc_blk_nums=[1, 1, 1, 14],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        gate_lattice_period_px=lattice_period_px,
        gate_lattice_angle_deg=lattice_angle_deg,
        gate_bandwidth=0.06,
        gate_harmonics=(1, 2),
        gate_alpha_max=0.7,
        gate_init_alpha=0.0,
        gate_per_channel=True,
    )


# To add an architecture: write a builder function and register it here under a new name.
MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "unet": build_unet,
    "sgarnet": build_sgarnet,
}


def available_architectures() -> list[str]:
    return sorted(MODEL_BUILDERS)


def architecture_options(architecture: str, args: argparse.Namespace) -> dict[str, Any]:
    """Extra builder arguments of an architecture, taken from the train.py arguments.

    Checkpoints store these arguments, so a loaded model gets the same settings it was trained with.
    """
    if architecture != "sgarnet":
        return {}

    period = getattr(args, "sgarnet_lattice_period", None)
    angle = getattr(args, "sgarnet_lattice_angle", None)
    if period is None or angle is None:
        raise ValueError(
            "SGARNet needs --sgarnet_lattice_period and --sgarnet_lattice_angle. "
            "Run estimate_lattice_period.py on the training inputs to measure them."
        )
    return {"lattice_period_px": period, "lattice_angle_deg": angle}


def build_model(architecture: str, in_channels: int, out_channels: int, **options: Any) -> nn.Module:
    if architecture not in MODEL_BUILDERS:
        known = ", ".join(available_architectures())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {known}")

    builder = MODEL_BUILDERS[architecture]
    return builder(in_channels, out_channels, **options)


def load_trained_model(checkpoint_path: Path, architecture: str, n_colors: int) -> nn.Module:
    # weights_only=False: the file also stores 'args'. Only load checkpoint files you trust.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    options = architecture_options(architecture, checkpoint["args"])
    model = build_model(architecture, in_channels=n_colors, out_channels=n_colors, **options)
    load_checkpoint(checkpoint, model)
    return model.eval()


def compile_model(model: nn.Module, mode: str) -> nn.Module:
    # torch.compile returns an OptimizedModule (an nn.Module), but its type hint only says "callable".
    return cast(nn.Module, torch.compile(model, mode=mode))
