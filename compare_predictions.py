"""Puts the U-Net and SGARNet predictions side by side, from the pred/ folders written by save_test_preds.py.

Usage:
    python compare_predictions.py DIR_Z DIR_X UNET_PRED_DIR SGARNET_PRED_DIR OUT_DIR
Each OUT_DIR/<name>_compare.png shows, from left to right: sim_MCF input | U-Net | SGARNet | HR target,
under a header row with these labels.
"""
import os
import sys

import cv2
import numpy as np

from imageDatastore import imageDatastore
from utils import tensor2np


LABELS = ("input", "U-Net", "SGARNet", "target")


def read_prediction(pred_dir: str, name: str) -> np.ndarray:
    path = os.path.join(pred_dir, name + "_pred.png")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Missing prediction {path}. Run save_test_preds.py first.")
    return image


def label_row(panel_width: int, dtype: np.dtype) -> np.ndarray:
    scale = max(0.4, panel_width / 400)
    thickness = max(1, round(scale * 1.5))
    (_, text_height), baseline = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    padding = max(3, text_height // 2)
    height = text_height + baseline + 2 * padding
    white = np.iinfo(dtype).max
    row = np.full((height, panel_width * len(LABELS)), white, dtype=dtype)
    for index, label in enumerate(LABELS):
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = index * panel_width + (panel_width - text_width) // 2
        cv2.putText(row, label, (x, padding + text_height), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness, cv2.LINE_AA)
    return row


def main() -> None:
    dir_z, dir_x, unet_pred_dir, sgarnet_pred_dir, out_dir = sys.argv[1:6]
    os.makedirs(out_dir, exist_ok=True)
    dataset = imageDatastore(dir_z, dir_x)

    for index in range(len(dataset)):
        input_tensor, target_tensor = dataset[index]
        input_img = tensor2np(input_tensor)[:, :, 0]
        target_img = tensor2np(target_tensor)[:, :, 0]

        name = os.path.splitext(os.path.basename(dataset.list_X[index]))[0]
        unet_img = read_prediction(unet_pred_dir, name)
        sgarnet_img = read_prediction(sgarnet_pred_dir, name)

        panels = np.concatenate([input_img, unet_img, sgarnet_img, target_img], axis=1)
        image = np.concatenate([label_row(input_img.shape[1], panels.dtype), panels], axis=0)
        cv2.imwrite(os.path.join(out_dir, name + "_compare.png"), image)

    print(f"Saved {len(dataset)} comparisons to {out_dir}")


if __name__ == "__main__":
    main()
