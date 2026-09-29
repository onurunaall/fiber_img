import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from imageDatastore import imageDatastore
from model_factory import COMPILE_MODES, available_architectures, compile_model, load_trained_model
from validation import Predictor, accumulate_validation_totals, compute_validation_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint (PyTorch) or a TensorRT engine on the validation set."
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint", type=Path, help="run a PyTorch checkpoint")
    model_source.add_argument("--engine", type=Path, help="run a TensorRT engine")

    parser.add_argument("--arch", default="unet", choices=available_architectures(), help="architecture of --checkpoint")
    parser.add_argument("--compile", action="store_true", help="optimize --checkpoint with torch.compile")
    parser.add_argument("--compile_mode", default="default", choices=COMPILE_MODES)

    parser.add_argument("--dir_ZValid", required=True, type=str)
    parser.add_argument("--dir_XValid", required=True, type=str)
    parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")
    parser.add_argument("--rgb_range", default=1, type=int, help="maxium value of RGB")
    parser.add_argument("--ext", default=".png", type=str)
    parser.add_argument("--val_batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--timed_iterations", default=50, type=int)
    return parser.parse_args()


def load_predictor(args: argparse.Namespace, device: torch.device) -> Predictor:
    if args.engine is not None:
        # Imported here so that evaluating a PyTorch checkpoint does not require TensorRT.
        from tensorrt_engine import TensorRTModel

        return TensorRTModel(args.engine, device)

    model = load_trained_model(args.checkpoint, args.arch, args.n_colors).to(device)
    if args.compile:
        model = compile_model(model, args.compile_mode)
    return model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency_ms(
    predict: Predictor,
    inputs: torch.Tensor,
    warmup_iterations: int,
    timed_iterations: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup_iterations):
            predict(inputs)
        synchronize(inputs.device)

        start = time.perf_counter()
        for _ in range(timed_iterations):
            predict(inputs)
        synchronize(inputs.device)
        elapsed_seconds = time.perf_counter() - start

    return elapsed_seconds * 1000 / timed_iterations


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = imageDatastore(
        args.dir_ZValid,
        args.dir_XValid,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        ext=args.ext,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    predict = load_predictor(args, device)
    criterion = nn.L1Loss().to(device)

    totals = accumulate_validation_totals(predict, loader, criterion, device)
    result = compute_validation_result(totals)
    print(f"Valid loss: {result.loss:.4f} \t\t PSNR: {result.psnr:.4f} \t\t SSIM: {result.ssim:.4f}")

    first_batch_inputs = next(iter(loader))[0].to(device)
    latency = measure_latency_ms(predict, first_batch_inputs, args.warmup_iterations, args.timed_iterations)
    print(f"Latency: {latency:.3f} ms per batch of {first_batch_inputs.shape[0]} images")


if __name__ == "__main__":
    main()
