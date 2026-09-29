from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import compute_psnr, compute_ssim, tensor2np


type Predictor = Callable[[torch.Tensor], torch.Tensor]


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
    loss: float
    psnr: float
    ssim: float


def accumulate_validation_totals(
    predict: Predictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> ValidationTotals:
    totals = ValidationTotals()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            predictions = predict(inputs)
            batch_size = targets.shape[0]
            totals.loss += criterion(predictions, targets).item() * batch_size

        totals.count += batch_size
        for index in range(batch_size):
            prediction_image = tensor2np(predictions.detach()[index])
            target_image = tensor2np(targets.detach()[index])

            totals.psnr += compute_psnr(target_image, prediction_image)
            totals.ssim += compute_ssim(prediction_image, target_image)

    return totals


def compute_validation_result(totals: ValidationTotals) -> ValidationResult:
    if totals.count == 0:
        raise ValueError("The validation set is empty.")

    return ValidationResult(
        loss=totals.loss / totals.count,
        psnr=totals.psnr / totals.count,
        ssim=totals.ssim / totals.count,
    )
