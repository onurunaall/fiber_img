import argparse
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from model_UNET_EDSR import UNET, UNET_EDSR
from utils import load_checkpoint


type ModelBuilder = Callable[..., nn.Module]
type ModelDescription = dict[str, Any]

# EDSR size of the commented-out UNET_EDSR line in the original train.py (n_resblocks=32, n_feats=256)
EDSR_DEFAULT_RESBLOCKS = 32
EDSR_DEFAULT_FEATS = 256

COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def build_unet(in_channels: int, out_channels: int, full_size_skip: bool = False) -> nn.Module:
    # full_size_skip (unet_v2): 8 channels instead of out_channels at full size, plus a skip connection at full size.
    return UNET(in_channels=in_channels, out_channels=out_channels, inplace=False, full_size_skip=full_size_skip)


def build_unet_edsr(
    in_channels: int,
    out_channels: int,
    n_resblocks: int,
    n_feats: int,
    res_scale: float = 1.0,
    full_size_skip: bool = False,
) -> nn.Module:
    # UNet (input -> 8 channels) followed by an EDSR with scale 1, i.e. no upsampling: the output has the input size.
    # full_size_skip (unet_v2_edsr): the UNet of unet_v2, which adds ConvIn's output back at full size, before the EDSR.
    return UNET_EDSR(
        in_channels=in_channels,
        out_channels=out_channels,
        features=[32, 64, 128, 256],
        n_resblocks=n_resblocks,
        n_feats=n_feats,
        scale=1,
        inplace=False,
        full_size_skip=full_size_skip,
        res_scale=res_scale,
    )


# To add an architecture: write a builder function and register it here under a new name.
MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "unet": build_unet,
    "unet_v2": partial(build_unet, full_size_skip=True),
    "unet_edsr": build_unet_edsr,
    "unet_v2_edsr": partial(build_unet_edsr, full_size_skip=True),
}


def available_architectures() -> list[str]:
    return sorted(MODEL_BUILDERS)


def architecture_options(architecture: str, args: argparse.Namespace) -> dict[str, Any]:
    """Extra builder arguments of an architecture, taken from the train.py arguments.

    Checkpoints store these arguments, so a loaded model gets the same settings it was trained with.
    """
    if architecture in ("unet_edsr", "unet_v2_edsr"):
        return {
            "n_resblocks": getattr(args, "edsr_n_resblocks", EDSR_DEFAULT_RESBLOCKS),
            "n_feats": getattr(args, "edsr_n_feats", EDSR_DEFAULT_FEATS),
            # Checkpoints from before this option were trained with 1.0.
            "res_scale": getattr(args, "edsr_res_scale", 1.0),
        }
    return {}


def model_description(args: argparse.Namespace) -> ModelDescription:
    """What the model is, e.g. {"arch": "unet_edsr", "n_colors": 1, "options": {"n_resblocks": 32, ...}}.

    Saved in every checkpoint as checkpoint["model"], so the checkpoint can build its own model again.
    """
    architecture = getattr(args, "arch", "unet")  # the first train.py (before --arch existed) only trained the unet
    # in_channels: the image channels, plus 1 when the fiber's reference image is an extra input channel.
    in_channels = args.n_colors + (1 if getattr(args, "reference_image", None) else 0)
    return {"arch": architecture, "n_colors": args.n_colors, "in_channels": in_channels,
            "options": architecture_options(architecture, args)}


def saved_model_description(checkpoint: Mapping[str, Any]) -> ModelDescription:
    """The model description of a checkpoint. Checkpoints from before 2026-10-22 have none: their 'args' hold it."""
    description = dict(checkpoint["model"]) if "model" in checkpoint else model_description(checkpoint["args"])
    # Checkpoints from before the reference image (2026-11-13) have no "in_channels": their input is the image alone.
    description.setdefault("in_channels", description["n_colors"])
    return description


def build_model(architecture: str, in_channels: int, out_channels: int, **options: Any) -> nn.Module:
    if architecture not in MODEL_BUILDERS:
        known = ", ".join(available_architectures())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {known}")

    builder = MODEL_BUILDERS[architecture]
    return builder(in_channels, out_channels, **options)


def build_described_model(description: ModelDescription) -> nn.Module:
    return build_model(description["arch"], in_channels=description["in_channels"],
                       out_channels=description["n_colors"], **description["options"])


def check_reference(description: ModelDescription, reference_image: str | None) -> None:
    """Stops if a model trained with a reference image gets none, or a model trained without one gets one."""
    needs_reference = description["in_channels"] > description["n_colors"]
    if needs_reference and reference_image is None:
        raise ValueError("This checkpoint was trained with a reference image of the fiber: "
                         "give that image with --reference_image FILE.")
    if not needs_reference and reference_image is not None:
        raise ValueError("This checkpoint was trained without a reference image: "
                         "leave out --reference_image (also in the config file).")


def load_trained_model(checkpoint_path: Path) -> tuple[nn.Module, ModelDescription]:
    """The model of a train.py checkpoint, with its weights, in eval mode, and its description.

    The checkpoint says what model it is.
    """
    # weights_only=False: old checkpoints store 'args' as an argparse.Namespace, and --save_model_arch stores the whole
    # model object. Only load checkpoint files you trust.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    description = saved_model_description(checkpoint)
    model = build_described_model(description)
    load_checkpoint(checkpoint, model)
    return model.eval(), description


def compile_model(model: nn.Module, mode: str) -> nn.Module:
    # torch.compile returns an OptimizedModule (an nn.Module), but its type hint only says "callable".
    return cast(nn.Module, torch.compile(model, mode=mode))
