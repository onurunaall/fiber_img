import torch.utils.data as data
import os
import re
import cv2
import numpy as np
from lib import common


IMG_EXTENSIONS = ['.png', '.npy', ]


def default_loader(path, n_colors):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, 'cv2 could not read the image: %s' % path
    if n_colors == 3:
        # RGB image
        return img[:, :, [2, 1, 0]]   # BGR to RGB
    else:
        # grayscale image
        return img

def npy_loader(path):
    return np.load(path)

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class imageDatastore(data.Dataset):
    def __init__(self, 
                 dir_Z,
                 dir_X, 
                 patches=False,
                 patch_size=256, 
                 n_colors=1, 
                 rgb_range=1, 
                 augment=False, 
                 hflip=False, 
                 vflip=False, 
                 rotate=False,
                 ext=".png"):
        
        self.dir_Z = dir_Z
        self.dir_X = dir_X
        self.patches = patches
        
        self.patch_size = patch_size
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.ext = ext  # ".png" ".npy"
        
        # Data augmentation
        self.augment = True if hflip or vflip or rotate else False
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        
        self.scale = 1  # ratio of output size to input size, self.opt.scale
        self.repeat = 1
        
        self.list_Z, self.list_X = self._scan()  # return sorted list
        self.num_train = len(self.list_Z)
        self._check_pairs()

    def __getitem__(self, idx):
        X, Z = self._load_file(idx)
        
        X, Z = common.set_channel(X, Z, n_channels=self.n_colors)
        X, Z = self._get_patch(X, Z)
        X, Z = self._augment_data(X, Z)
        
        X_tensor, Z_tensor = common.np2Tensor(X, Z, rgb_range=self.rgb_range)
        return X_tensor, Z_tensor

    def __len__(self):
        return self.num_train * self.repeat

    def _get_index(self, idx):
        return idx % self.num_train
    
    def _scan(self):
        list_Z = sorted(make_dataset(self.dir_Z))
        list_X = sorted(make_dataset(self.dir_X))
        return list_Z, list_X

    # Check that X file number i and Z file number i show the same image
    def _check_pairs(self):
        assert len(self.list_X) == len(self.list_Z), \
            'X has %d files but Z has %d files' % (len(self.list_X), len(self.list_Z))
        for path_X, path_Z in zip(self.list_X, self.list_Z):
            num_X = re.findall(r'\d+', os.path.basename(path_X))
            num_Z = re.findall(r'\d+', os.path.basename(path_Z))
            assert num_X == num_Z, 'Pair mismatch: %s <-> %s' % (path_X, path_Z)
    
    # Split patches for training
    def _get_patch(self, imgX, imgZ):
        scale = self.scale
        if self.patches:
            imgX, imgZ = common.get_patch(imgX, imgZ, patch_size=self.patch_size, scale=scale)
        elif not self.patches and scale == 1: 
            imgZ = imgZ
        else:
            ih, iw = imgX.shape[:2]
            imgZ = imgZ[0:ih * scale, 0:iw * scale, :]
        return imgX, imgZ

    def _augment_data(self, imgX, imgZ):
        if self.augment:
            imgX, imgZ = common.augment(imgX, imgZ, hflip=self.hflip, vflip=self.vflip, rot=self.rotate)
        return imgX, imgZ

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            X = npy_loader(self.list_X[idx])
            Z = npy_loader(self.list_Z[idx])
        else:
            X = default_loader(self.list_X[idx], self.n_colors)
            Z = default_loader(self.list_Z[idx], self.n_colors)
        return X, Z