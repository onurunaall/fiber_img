"""Validation metrics of the raw sim_MCF input against the HR target: the score of a model that changes nothing.

Usage:  python baseline.py DIR_Z DIR_X [--n_colors N]
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from imageDatastore import imageDatastore
from validation import accumulate_validation_totals, compute_validation_result


def identity(inputs):
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics of the raw input against the target (no model).")
    parser.add_argument("dir_z", help="folder of the HR targets")
    parser.add_argument("dir_x", help="folder of the sim_MCF inputs")
    parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")
    args = parser.parse_args()

    dataset = imageDatastore(args.dir_z, args.dir_x, n_colors=args.n_colors)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    totals = accumulate_validation_totals(identity, loader, nn.L1Loss(), torch.device("cpu"))
    result = compute_validation_result(totals)
    print(f"Baseline (raw sim input vs HR): L1 {result.loss:.4f}  PSNR {result.psnr:.4f}  SSIM {result.ssim:.4f}")


if __name__ == "__main__":
    main()
