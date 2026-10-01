import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from imageDatastore import imageDatastore
from validation import accumulate_validation_totals, compute_validation_result


def identity(inputs):
    return inputs


dataset = imageDatastore(sys.argv[1], sys.argv[2])
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
totals = accumulate_validation_totals(identity, loader, nn.L1Loss(), torch.device("cpu"))
result = compute_validation_result(totals)
print(f"Baseline (raw sim input vs HR): L1 {result.loss:.4f}  PSNR {result.psnr:.4f}  SSIM {result.ssim:.4f}")
