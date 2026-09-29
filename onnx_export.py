import argparse
import copy
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.export import Dim

from model_factory import available_architectures, load_trained_model


type Precision = Literal["fp32", "fp16"]

INPUT_NAME = "input"
OUTPUT_NAME = "output"


class HalfPrecisionModel(nn.Module):
    """Runs the model in FP16 but keeps FP32 inputs and outputs.

    TensorRT 11 runs every layer in the data type the ONNX graph uses, so an FP16 engine
    needs an FP16 graph. The casts at both ends keep the engine interface identical to FP32.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).half()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs: torch.Tensor = self.model(inputs.half())
        return outputs.float()


def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    channels: int,
    height: int,
    width: int,
    precision: Precision,
) -> None:
    exportable_model = HalfPrecisionModel(model) if precision == "fp16" else model
    exportable_model.eval()

    # The batch dimension stays dynamic. torch.export treats a dimension of size 1
    # as a constant, so the example input needs at least 2 images.
    example_input = torch.rand(2, channels, height, width)
    dynamic_batch = {0: Dim("batch")}

    torch.onnx.export(
        exportable_model,
        (example_input,),
        onnx_path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=(dynamic_batch,),
        external_data=False,
        dynamo=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="path of the .onnx file to write")
    parser.add_argument("--arch", default="unet", choices=available_architectures())
    parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")
    parser.add_argument("--height", required=True, type=int, help="input image height")
    parser.add_argument("--width", required=True, type=int, help="input image width")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_trained_model(args.checkpoint, args.arch, args.n_colors)
    export_to_onnx(model, args.output, args.n_colors, args.height, args.width, args.precision)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
