import random
import torch
import cv2
import numpy as np
import skimage.color as sc
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, Compose
)

def get_patch(*args, patch_size, scale):
    ih, iw = args[0].shape[:2]

    tp = patch_size   # target patch (HR)
    ip = tp // scale  # input patch (LR)

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [args[0][iy:iy + ip, ix:ix + ip, :], *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]] # results

    # ret = [args[0][iy:iy + ip, ix:ix + ip], *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]]  # results
    return ret




def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img
    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        # np_transpose = np.ascontiguousarray(img)
        tensor = torch.from_numpy(np_transpose.copy()).float()
        tensor.mul_(rgb_range / 255)

        return tensor
    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=False, vflip=False, rot=False):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        # if hflip: img = img[:, ::-1]
        # if vflip: img = img[::-1, :]
        # if rot90: img = img.transpose(1, 0)

        return img

    return [_augment(a) for a in args]




def augment2(*args, hflip=False, vflip=False, rot=False, translate=False, max_translate=0.0):
    
    transforms = []
    
    if hflip:
        transforms.append(HorizontalFlip(p=0.5))
    if vflip:
        transforms.append(VerticalFlip(p=0.5))
    if rot:
        transforms.append(Rotate(limit=90, p=0.5))
    if translate:
        transforms.append(ShiftScaleRotate(
            shift_limit=max_translate, 
            scale_limit=0, 
            rotate_limit=0, 
            p=0.5, 
            border_mode=0,  # BORDER_CONSTANT, default value = 0
            interpolation=cv2.INTER_CUBIC,
        ))
    
    # One random draw, applied to ALL images: args[0] is 'image', the others 'image1', 'image2', ...
    extra_targets = {}
    for i in range(1, len(args)):
        extra_targets['image%d' % i] = 'image'
    augmentation = Compose(transforms, additional_targets=extra_targets)

    inputs = {'image': args[0]}
    for i in range(1, len(args)):
        inputs['image%d' % i] = args[i]
    out = augmentation(**inputs)

    results = [out['image']]
    for i in range(1, len(args)):
        results.append(out['image%d' % i])
    return results