from __future__ import annotations
import math
import os
import time
import numpy as np
import random
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision
from torch import nn
from torchvision import models
# from torch.utils.data import DataLoader
# from dataset import CarvanaDataset

# High-frequency analysis metrics
import lpips
import piq
from pytorch_msssim import ms_ssim


def setup_seed(seed=42, ConvOptim=False):
    if seed is None:
        seed = random.randint(1, 10000)
        print("Ramdom Seed: ", seed)
    
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if ConvOptim:
         torch.backends.cudnn.benchmark = False
         torch.backends.cudnn.deterministic = True


def get_list(path, ext=".png"):
    # Sorted list
    # fileList.sort(key=lambda x: int(x[5:9]))
    fileList = sorted(os.listdir(path))
    return [os.path.join(path, f) for f in fileList if f.endswith(ext)]


def save_checkpoint(model, model_arch, optimizer, epoch, loss_stats, args, text=''):
    print("=> Saving checkpoint")
    # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
    checkpoint = {
        "state_dict": unwrap_module(model).state_dict(), 
        "optimizer": optimizer.state_dict(),
        "loss_stats": loss_stats, 
        "epoch": epoch, 
        "args": args, 
        }
    
    if args.save_model_arch:
        checkpoint.update({"model_arch": model_arch})

    torch.save(checkpoint, os.path.join(args.save_path, "net_epoch_{}".format(epoch) + text + ".pth.tar"))
    # np.savez(args.save_path + "net_epoch_{}_info.npz".format(epoch), loss_stats=loss_stats, num_epochs=epoch, args=args)


def load_checkpoint(checkpoint, model):
    if isinstance(model, nn.DataParallel):
        # DP
        print("=> Loading checkpoint with data parallel")
        state_dict = fix_state_dict_for_loading(checkpoint["state_dict"], to_dataparallel=1)
        model.load_state_dict(state_dict, strict=True)
    else:   
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])    
    

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for idx, (x, z) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x).clamp(0, 1)
        torchvision.utils.save_image(preds, os.path.join(folder, f"pred_{idx:03d}.png"))
        torchvision.utils.save_image(z, os.path.join(folder, f"target_{idx:03d}.png"))
        

def adjust_learning_rate(optimizer, epoch, lr_init, gamma, step_size):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_cosine_learning_rate(optimizer, iteration, total_iterations, lr_init, eta_min):
    # Closed form of torch.optim.lr_scheduler.CosineAnnealingLR(T_max=total_iterations); iteration starts at 0
    lr = eta_min + (lr_init - eta_min) * (1 + math.cos(math.pi * iteration / total_iterations)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().detach().cpu().clamp_(*min_max)
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8 and min_max==(0, 1):
        img_np = (img_np * 255.0).round()
    else:
        img_np = img_np.round()
        
    return img_np.astype(out_type)


def print_network(net):
    num_params = []
    for param in net.parameters():
        num_params.append(param.numel())
    # print(net)
    print('Total number of parameters: %d' % np.sum(num_params))


class Timer():
    def __init__(self):
        self.v = time.time()
    def s(self):
        self.v = time.time()
    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


# old version
# def compute_psnr(im1, im2):
#     data_range = 1.0 if (im1.dtype == 'float16' or im1.dtype == 'float32') else None
#     p = psnr(im1, im2, data_range=data_range)
#     return p

# def compute_ssim(im1, im2):
#     channel_axis = 2 if len(im1.shape) == 3 else 0
#     data_range = 1.0 if (im1.dtype == 'float16' or im1.dtype == 'float32') else None
#     s = ssim(im1, im2, channel_axis=channel_axis, data_range=data_range)
#     return s



def _get_matlab_like_data_range(im1, im2):
    """
    Approximate MATLAB's DynamicRange / PeakVal behavior for psnr/ssim:
    - uint8  -> 255
    - uint16 -> 65535
    - float images in [0, 1] -> 1.0
    - float images that are actually in [0, 255] -> 255.0
    """
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if np.issubdtype(im1.dtype, np.integer):
        info = np.iinfo(im1.dtype)
        return float(info.max) - float(info.min)  # ? MATLAB: diff(getrangefromclass)
    
    if np.issubdtype(im2.dtype, np.integer):
        info = np.iinfo(im2.dtype)
        return float(info.max) - float(info.min)  # ? MATLAB: diff(getrangefromclass)
    

    max_val = max(float(np.max(im1)), float(np.max(im2)))
    min_val = min(float(np.min(im1)), float(np.min(im2)))

    # MATLAB usually treats double/single images as images in [0, 1].
    if min_val >= 0.0 and max_val <= 1.0:
        return 1.0

    # If the float images are actually in [0, 255], use 255 as the data range.
    if min_val >= 0.0 and max_val <= 255.0:
        return 255.0

    # Fallback: use the actual dynamic range.
    return max_val - min_val

def compute_psnr(ref, im):
    data_range = _get_matlab_like_data_range(ref, ref)  # based on ref
    return psnr(ref, im, data_range=data_range)

def compute_ssim(im1, im2):
    im1 = np.asarray(im1); im2 = np.asarray(im2)
    assert im1.shape == im2.shape
    channel_axis = None if im1.ndim == 2 else -1
    data_range = _get_matlab_like_data_range(im1, im2)

    return ssim(
        im1, im2,
        channel_axis=channel_axis,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,                 # ??? ? ??????
        use_sample_covariance=False,
        K1=0.01,
        K2=0.03,                   # ??????
    )




def _to_tensor_NCHW(im):
    """
    ?????? (1, C, H, W) ? float tensor?
    ??:
      - (H, W)          ??
      - (H, W, 1)       ??(???)
      - (H, W, C)       ???
    """
    im = np.asarray(im)
    if im.ndim == 2:
        im = im[np.newaxis, np.newaxis]            # (1,1,H,W)
    elif im.ndim == 3:
        im = np.transpose(im, (2, 0, 1))[np.newaxis]   # (1,C,H,W)
    else:
        raise ValueError(f"Unsupported image shape: {im.shape}")
    return torch.from_numpy(np.ascontiguousarray(im)).float()


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_gmsd(im1, im2):
    """
    GMSD: Gradient Magnitude Similarity Deviation.
    ???/??????; ????????.
    ??????(? data_range ????); ????????.
    """
    im1 = np.asarray(im1); im2 = np.asarray(im2)
    assert im1.shape == im2.shape, f"Shape mismatch: {im1.shape} vs {im2.shape}"

    data_range = _get_matlab_like_data_range(im1, im2)

    t1 = _to_tensor_NCHW(im1).to(_DEVICE)
    t2 = _to_tensor_NCHW(im2).to(_DEVICE)

    # piq ????? [0, data_range] ?,???? clamp
    t1 = t1.clamp_(0.0, data_range)
    t2 = t2.clamp_(0.0, data_range)

    with torch.no_grad():
        val = piq.gmsd(t1, t2, data_range=data_range, reduction="mean")
    return float(val.item())

_LPIPS_NETS = {}
def _get_lpips_net(net="alex", device=_DEVICE):
    """net ?? 'alex' (????????) / 'vgg' / 'squeeze' [[3]]"""
    key = (net, str(device))
    if key not in _LPIPS_NETS:
        _LPIPS_NETS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LPIPS_NETS[key]


def _to_lpips_tensor(im, data_range):
    """
    ??? LPIPS ??? (1, 3, H, W), ?? [-1, 1] ? tensor.
    ???????? 3 ??.
    """
    im = np.asarray(im).astype(np.float32)

    # ???? [0, 1]
    im = np.clip(im / data_range, 0.0, 1.0)

    if im.ndim == 2:                               # (H, W)
        im = np.stack([im]*3, axis=0)              # (3, H, W)
    elif im.ndim == 3:
        if im.shape[-1] == 1:                      # (H, W, 1)
            im = np.repeat(im, 3, axis=-1)
        im = np.transpose(im, (2, 0, 1))           # (C, H, W)
        if im.shape[0] == 1:
            im = np.repeat(im, 3, axis=0)
    else:
        raise ValueError(f"Unsupported image shape: {im.shape}")

    im = im * 2.0 - 1.0                            # [0,1] -> [-1,1]
    return torch.from_numpy(np.ascontiguousarray(im[np.newaxis])).float()


def compute_lpips(im1, im2, net="alex"):
    """
    LPIPS: Learned Perceptual Image Patch Similarity.
    ???/??/??????; ???????? [[8]].
    """
    im1 = np.asarray(im1); im2 = np.asarray(im2)
    assert im1.shape == im2.shape, f"Shape mismatch: {im1.shape} vs {im2.shape}"

    data_range = _get_matlab_like_data_range(im1, im2)
    t1 = _to_lpips_tensor(im1, data_range).to(_DEVICE)
    t2 = _to_lpips_tensor(im2, data_range).to(_DEVICE)

    net_fn = _get_lpips_net(net)
    with torch.no_grad():
        val = net_fn(t1, t2)
    return float(val.item())



def get_pixels_around_core_4(loss_mode, n_lossPix, corePos):
    corePosX, corePosY = corePos[:,0], corePos[:,1]    
    corePosX, corePosY = [int(x) for x in corePosX], [int(x) for x in corePosY]   # to int list
    # corePosY, corePosX = int(corePos[0]), int(corePos[1])
    
    if loss_mode == 'pixel':
        if n_lossPix < 11**2:
            # the pixel at core center
            lossPixY = corePosY
            lossPixX = corePosX
            
            # 5 pixels around core center
            if n_lossPix >= 5:
                lossPixY = lossPixY + [v-1 for v in corePosY] + [v+1 for v in corePosY] + [v for v in corePosY] + [v for v in corePosY]
                lossPixX = lossPixX + [v for v in corePosX] + [v for v in corePosX] + [v-1 for v in corePosX] + [v+1 for v in corePosX]
                
                # 9 pixels around core center
                if n_lossPix >= 9:
                    lossPixY = lossPixY + [v-1 for v in corePosY] + [v+1 for v in corePosY] + [v-1 for v in corePosY] + [v+1 for v in corePosY]
                    lossPixX = lossPixX + [v-1 for v in corePosX] + [v+1 for v in corePosX] + [v+1 for v in corePosX] + [v-1 for v in corePosX]
                    
                    # 13 pixels around core center
                    if n_lossPix >= 13:
                        lossPixY = lossPixY + [v-2 for v in corePosY] + [v+2 for v in corePosY] + [v for v in corePosY] + [v for v in corePosY]
                        lossPixX = lossPixX + [v for v in corePosX] + [v for v in corePosX] + [v-2 for v in corePosX] + [v+2 for v in corePosX]
        
        elif n_lossPix == 11**2:
            # n_lossPix = 11^2

            r = 5

            # corePos = np.array([[63,63],[190,63],[63,190],[190,190]])
            # corePosY, corePosX = [int(x) for x in corePosY], [int(x) for x in corePosX]

            lossPixY = []
            lossPixX = []
            n_cores = corePos.shape[0]

            for iCore in range(n_cores):
                # 11 x 11 window around THIS core (same order as MATLAB: x outer, y inner)
                cy, cx = corePosY[iCore], corePosX[iCore]
                for px in range(cx - r, cx + r + 1):
                    for py in range(cy - r, cy + r + 1):
                        lossPixY.append(py)
                        lossPixX.append(px)
        else:
            lossPixX, lossPixY = None, None
    
    return lossPixX, lossPixY


def unwrap_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def fix_state_dict_for_loading(sd, to_dataparallel: bool):
    """to_dataparallel=True ?? key ? 'module.';?????"""
    new_sd = OrderedDict()
    for k, v in sd.items():
        if to_dataparallel and not k.startswith("module."):
            new_sd["module."+k] = v
        elif (not to_dataparallel) and k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


_VGG_FEATS = {}
def _get_vgg_feats(device):
    key = str(device)
    if key not in _VGG_FEATS:
        _VGG_FEATS[key] = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:4].eval().to(device)
    return _VGG_FEATS[key]

# VGG perceptual loss
def vgg_freq_diff(img1: torch.Tensor, img2: torch.Tensor, device="cpu") -> torch.Tensor:
    
    def imgnet_norm_gray_to_rgb(x, device="cpu"):
        # gray to RGB
        x = x.repeat(1, 3, 1, 1)   # [N,3,H,W]

        # normalization of ImageNet 
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        return (x - mean) / std
    
    # load first two layers of VGG
    vgg = _get_vgg_feats(device)

    with torch.no_grad():
        x1 = imgnet_norm_gray_to_rgb(img1, device)
        x2 = imgnet_norm_gray_to_rgb(img2, device)
        
        f1 = vgg(x1)
        f2 = vgg(x2)
    
    # f: [N, C, H, W]
    F1 = torch.fft.rfftn(f1, dim=(2,3))
    F2 = torch.fft.rfftn(f2, dim=(2,3))

    # Calculate amplitude spectrum difference
    diff = torch.abs(torch.abs(F1) - torch.abs(F2))  # [N, C, H, W_fft]
    
    # Average across channel and spatial dimensions, retaining only the batch dimension -> [N]
    return diff.mean(dim=(1, 2, 3))


def calc_high_f_metrics(Y, Z, device='cpu'):
    
    def to_minus1_1(x):
        return x * 2 - 1
    
    # LPIPS 
    lpips_model = _get_lpips_net('vgg', device)
    
    val_y = lpips_model(to_minus1_1(Y), to_minus1_1(Z))
    lpips_y = val_y.view(-1).detach().cpu().numpy()
    
    # MS-SSIM
    msssim_y = ms_ssim(Y.half(), Z.half(), data_range=1.0, size_average=False).cpu().numpy()
    
    # FFT spectral differences
    FreqDiff_y = vgg_freq_diff(Y, Z, device=device).cpu().numpy()
    
    return lpips_y, msssim_y, FreqDiff_y


try:
    from models.spatialTransformer import perspective2d
except ImportError:
    perspective2d = None

class HomographyStageScheduler:
    """
    Manages three-stage training of the homography layer.

    Stage 1  [affine_only]:   Train affine 6-DoF only.
    Stage 2  [persp_only]:    Train perspective 2-DoF only.
    Stage 3  [joint]:         Fine-tune both simultaneously.

    Args:
        homography_layer: The LearnableHomography module.
        total_epochs:     Total number of training epochs.
        stage_ratios:     Length-3 list/tuple that sums to 1.0,
                          e.g. [0.4, 0.4, 0.2] means 40% affine,
                          40% perspective, 20% joint fine-tuning.

    Usage:
        scheduler = HomographyStageScheduler(layer, 100, [0.4, 0.4, 0.2])
        for epoch in range(1, 101):
            scheduler.step(epoch)   # call once at the top of each epoch
    """

    STAGE_AFFINE = 1
    STAGE_PERSP  = 2
    STAGE_JOINT  = 3

    _STAGE_NAMES = {
        STAGE_AFFINE: "Stage 1 - Affine only (6 DoF)",
        STAGE_PERSP:  "Stage 2 - Perspective only (2 DoF)",
        STAGE_JOINT:  "Stage 3 - Joint fine-tune (8 DoF)",
    }

    _STAGE_ACTIONS = {
        STAGE_AFFINE: "freeze_persp",
        STAGE_PERSP:  "freeze_affine",
        STAGE_JOINT:  "unfreeze_all",
    }

    def __init__(
        self,
        homography_layer: perspective2d,
        total_epochs: int,
        stage_ratios: tuple | list = (0.4, 0.4, 0.2),
    ):
        assert len(stage_ratios) == 3, "stage_ratios must have exactly 3 elements"
        assert abs(sum(stage_ratios) - 1.0) < 1e-6, "stage_ratios must sum to 1"

        self.layer = homography_layer
        self.current_stage = None

        # Convert ratios to cumulative epoch boundaries (1-indexed)
        cum = 0
        self._boundaries = []
        for r in stage_ratios:
            cum += r
            self._boundaries.append(int(round(cum * total_epochs)))
        # e.g. total=100, ratios=[0.4,0.4,0.2] -> boundaries=[40, 80, 100]

    def _epoch_to_stage(self, epoch: int) -> int:
        """Map a 1-indexed epoch to its stage id."""
        if epoch <= self._boundaries[0]:
            return self.STAGE_AFFINE
        elif epoch <= self._boundaries[1]:
            return self.STAGE_PERSP
        else:
            return self.STAGE_JOINT

    def step(self, epoch: int) -> bool:
        """
        Call at the start of each epoch (1-indexed).
        Returns True when a stage transition occurs.
        """
        target = self._epoch_to_stage(epoch)

        if target == self.current_stage:
            return False                       # no transition

        self.current_stage = target
        getattr(self.layer, self._STAGE_ACTIONS[target])()
        return True

    @property
    def stage_name(self) -> str:
        """Human-readable name of the current stage."""
        return self._STAGE_NAMES.get(self.current_stage, "Unknown")

