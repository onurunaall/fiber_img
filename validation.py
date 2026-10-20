from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


type Predictor = Callable[[torch.Tensor], torch.Tensor]

# PSNR and SSIM are measured on 8-bit images, so their data range is 255.
DATA_RANGE = 255.0
# SSIM settings of utils.compute_ssim (skimage, like MATLAB's ssim): a Gaussian window with sigma 1.5,
# cut off at 3.5 sigma, i.e. 5 pixels to each side (11 taps); K1 = 0.01, K2 = 0.03.
SSIM_SIGMA = 1.5
SSIM_RADIUS = 5


@dataclass
class ValidationTotals:
    """Per-image sums, so that totals from several GPUs can simply be added together."""

    loss: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    count: int = 0

    def as_list(self) -> list[float]:
        return [self.loss, self.psnr, self.ssim, float(self.count)]

    @classmethod
    def from_list(cls, values: list[float]) -> Self:
        loss, psnr, ssim, count = values
        return cls(loss=loss, psnr=psnr, ssim=ssim, count=round(count))


@dataclass(frozen=True)
class ValidationResult:
    """Mean over the images. loss: L1 of the raw output. psnr, ssim: of the output rounded to 8 bit."""

    loss: float
    psnr: float
    ssim: float

    def text(self) -> str:
        return f"Valid L1 (raw output): {self.loss:.4f} \t\t PSNR / SSIM (8-bit output): {self.psnr:.4f} / {self.ssim:.4f}"


def to_8bit(images: torch.Tensor) -> torch.Tensor:
    """The pixel values of the saved 8-bit PNGs, as float64: clipped to 0..1, times 255, rounded (like utils.tensor2np)."""
    return (images.float().clamp(0, 1) * 255).round().double()


def gaussian_blur_valid(images: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Gaussian blur without padding: an (N, 1, H, W) batch becomes (N, 1, H - 2r, W - 2r) for r = SSIM_RADIUS."""
    images = F.conv2d(images, weights.view(1, 1, -1, 1))
    return F.conv2d(images, weights.view(1, 1, 1, -1))


def psnr_ssim_8bit(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image PSNR and SSIM of (N, C, H, W) batches rounded to 8 bit, computed on their device.

    Gives the same numbers as utils.compute_psnr and utils.compute_ssim (skimage) on tensor2np images, but for
    a whole batch at once and without copying every image to the CPU. float64 keeps the numbers exact
    (TF32 on Ampere GPUs only affects float32).
    """
    prediction = to_8bit(predictions)
    target = to_8bit(targets)
    psnr = 10 * torch.log10(DATA_RANGE**2 / ((prediction - target) ** 2).mean(dim=(1, 2, 3)))

    # SSIM of each channel, then the mean over the channels, like skimage with channel_axis.
    batch_size, channels, height, width = prediction.shape
    x = prediction.reshape(batch_size * channels, 1, height, width)
    y = target.reshape(batch_size * channels, 1, height, width)
    offsets = torch.arange(-SSIM_RADIUS, SSIM_RADIUS + 1, dtype=torch.float64, device=x.device)
    weights = torch.exp(-0.5 * (offsets / SSIM_SIGMA) ** 2)
    weights /= weights.sum()

    # skimage blurs the whole image and then leaves out a border of SSIM_RADIUS pixels. Blurring without
    # padding gives exactly the inner pixels, so the border handling does not matter.
    mean_x = gaussian_blur_valid(x, weights)
    mean_y = gaussian_blur_valid(y, weights)
    variance_x = gaussian_blur_valid(x * x, weights) - mean_x * mean_x
    variance_y = gaussian_blur_valid(y * y, weights) - mean_y * mean_y
    covariance = gaussian_blur_valid(x * y, weights) - mean_x * mean_y
    c1 = (0.01 * DATA_RANGE) ** 2
    c2 = (0.03 * DATA_RANGE) ** 2
    ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)
    )
    ssim = ssim_map.mean(dim=(1, 2, 3)).view(batch_size, channels).mean(dim=1)
    return psnr, ssim


def accumulate_validation_totals(
    predict: Predictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> ValidationTotals:
    totals = ValidationTotals()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            predictions = predict(inputs)
            batch_size = targets.shape[0]
            # L1 of the raw output: the number training lowers, and the one that picks net_best.
            totals.loss += criterion(predictions, targets).item() * batch_size
            # PSNR and SSIM of the output rounded to 8 bit: what the PNGs of save_test_preds.py contain.
            psnr, ssim = psnr_ssim_8bit(predictions, targets)

        totals.psnr += psnr.sum().item()
        totals.ssim += ssim.sum().item()
        totals.count += batch_size

    return totals


def compute_validation_result(totals: ValidationTotals) -> ValidationResult:
    if totals.count == 0:
        raise ValueError("The validation set is empty.")

    return ValidationResult(
        loss=totals.loss / totals.count,
        psnr=totals.psnr / totals.count,
        ssim=totals.ssim / totals.count,
    )
