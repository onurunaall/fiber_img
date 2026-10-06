import os
import sys

import cv2
import numpy as np
import torch

from imageDatastore import imageDatastore
from model_factory import load_trained_model
from utils import tensor2np

# Usage: python save_test_preds.py CHECKPOINT DIR_Z DIR_X OUT_DIR [ARCH]   (ARCH: unet (default) or sgarnet)
checkpoint_path, dir_z, dir_x, out_dir = sys.argv[1:5]
architecture = sys.argv[5] if len(sys.argv) > 5 else "unet"
os.makedirs(os.path.join(out_dir, "pred"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "compare"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model(checkpoint_path, architecture, n_colors=1).to(device)
dataset = imageDatastore(dir_z, dir_x)

with torch.no_grad():
    for index in range(len(dataset)):
        input_tensor, target_tensor = dataset[index]
        prediction = model(input_tensor.unsqueeze(0).to(device))[0]

        input_img = tensor2np(input_tensor)[:, :, 0]
        pred_img = tensor2np(prediction)[:, :, 0]
        target_img = tensor2np(target_tensor)[:, :, 0]

        name = os.path.splitext(os.path.basename(dataset.list_X[index]))[0]
        cv2.imwrite(os.path.join(out_dir, "pred", name + "_pred.png"), pred_img)
        # left: sim_MCF input, middle: prediction, right: HR target
        side_by_side = np.concatenate([input_img, pred_img, target_img], axis=1)
        cv2.imwrite(os.path.join(out_dir, "compare", name + "_compare.png"), side_by_side)

print(f"Saved {len(dataset)} predictions to {out_dir}")
