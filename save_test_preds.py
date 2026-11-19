"""Saves the prediction of every image pair, plus a side-by-side comparison.

Usage:  python save_test_preds.py CHECKPOINT DIR_Z DIR_X OUT_DIR [--n_colors N] [--reference_image FILE]
The checkpoint says which model it holds. Writes OUT_DIR/pred/<name>_pred.png and
OUT_DIR/compare/<name>_compare.png (left: sim_MCF input, middle: prediction, right: HR target).
"""
import argparse
import os

import cv2
import numpy as np
import numpy.typing as npt
import torch

from imageDatastore import imageDatastore
from model_factory import check_reference, load_trained_model
from utils import tensor2np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save the predictions of a checkpoint as 8-bit PNG files.")
    parser.add_argument("checkpoint", help="checkpoint file written by train.py")
    parser.add_argument("dir_z", help="folder of the HR targets")
    parser.add_argument("dir_x", help="folder of the sim_MCF inputs")
    parser.add_argument("out_dir", help="the pred/ and compare/ folders are made here")
    parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")
    parser.add_argument("--reference_image", default=None, type=str,
                        help="white image of the fiber: needed if the model was trained with --reference_image")
    return parser.parse_args()


def to_cv2_image(tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
    """A (C, H, W) tensor as an 8-bit image for cv2.imwrite: gray as (H, W), color from RGB to cv2's BGR order."""
    image = tensor2np(tensor)
    return image[:, :, 0] if image.shape[2] == 1 else np.ascontiguousarray(image[:, :, ::-1])


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "pred"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "compare"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, description = load_trained_model(args.checkpoint)
    check_reference(description, args.reference_image)
    model = model.to(device)
    dataset = imageDatastore(args.dir_z, args.dir_x, n_colors=args.n_colors, reference_image=args.reference_image)

    with torch.no_grad():
        for index in range(len(dataset)):
            input_tensor, target_tensor = dataset[index]
            prediction = model(input_tensor.unsqueeze(0).to(device))[0]

            # Only the image channels: with a reference image, the input has it as an extra channel.
            input_img = to_cv2_image(input_tensor[:args.n_colors])
            pred_img = to_cv2_image(prediction)
            target_img = to_cv2_image(target_tensor)

            name = os.path.splitext(os.path.basename(dataset.list_X[index]))[0]
            cv2.imwrite(os.path.join(args.out_dir, "pred", name + "_pred.png"), pred_img)
            # left: sim_MCF input, middle: prediction, right: HR target
            side_by_side = np.concatenate([input_img, pred_img, target_img], axis=1)
            cv2.imwrite(os.path.join(args.out_dir, "compare", name + "_compare.png"), side_by_side)

    print(f"Saved {len(dataset)} predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
