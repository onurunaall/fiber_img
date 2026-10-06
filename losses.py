# ------------------------------------------------------------------------
# Training losses of SGARNet, ported from https://github.com/THUHoloLab/SGARNet
# (basicsr/models/losses/losses.py and options/train/MCFArtifactFree.yml).
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Modified from BasicSR (https://github.com/xinntao/BasicSR), Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


# Index of each ReLU layer in torchvision's vgg19().features
VGG19_LAYERS = {"relu1_1": 1, "relu2_1": 6, "relu3_1": 11, "relu4_1": 20, "relu5_1": 29}


def load_vgg19_weights() -> None:
    """Downloads the ImageNet VGG19 weights into the torch cache, if they are not there yet."""
    VGG19_Weights.IMAGENET1K_V1.get_state_dict(progress=True)


class PSNRLoss(nn.Module):
    """-PSNR for images in [0, 1]: 10 * log10(MSE), averaged over the batch."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss on the features of an ImageNet-trained VGG19.

    Grayscale images are repeated to 3 channels. (The original subtracts the 3-channel ImageNet mean
    from the 1-channel image, which broadcasts to the same result.)
    """

    def __init__(self, loss_weight=1.0, layer_weights=None, criterion="mse", norm_img=True):
        super().__init__()
        if layer_weights is None:
            layer_weights = {name: 1.0 for name in VGG19_LAYERS}
        if criterion not in ("mse", "l1"):
            raise ValueError(f"Unsupported criterion: {criterion}")

        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.criterion = criterion
        self.norm_img = norm_img
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        # Only the layers up to the deepest one used are needed.
        self.layer_index = {VGG19_LAYERS[name]: name for name in layer_weights}
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(self.layer_index) + 1].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def train(self, mode=True):
        # VGG stays in eval mode, like the frozen feature extractor of the original.
        super().train(mode)
        self.vgg.eval()
        return self

    def _features(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.norm_img:
            x = (x - self.mean) / self.std

        features = {}
        for index, layer in enumerate(self.vgg):
            x = layer(x)
            if index in self.layer_index:
                features[self.layer_index[index]] = x
        return features

    def forward(self, pred, target):
        pred_features = self._features(pred)
        with torch.no_grad():
            target_features = self._features(target)

        loss = 0.0
        distance = F.mse_loss if self.criterion == "mse" else F.l1_loss
        for name, weight in self.layer_weights.items():
            loss += weight * distance(pred_features[name], target_features[name])
        return self.loss_weight * loss


class SGARNetLoss(nn.Module):
    """pixel_opt + perceptual_opt of options/train/MCFArtifactFree.yml: PSNRLoss + 0.01 * VGG19 perceptual loss."""

    def __init__(self):
        super().__init__()
        self.pixel = PSNRLoss(loss_weight=1.0)
        self.perceptual = PerceptualLoss(
            loss_weight=0.01,
            layer_weights={name: 0.1 for name in VGG19_LAYERS},
            criterion="mse",
            norm_img=True,
        )

    def forward(self, pred, target):
        return self.pixel(pred, target) + self.perceptual(pred, target)
