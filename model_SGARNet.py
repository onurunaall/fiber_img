# ------------------------------------------------------------------------
# SGARNet (Spectral-Guided Artifact Removal Network), ported from
# https://github.com/THUHoloLab/SGARNet (basicsr/models/archs/SGARNet_arch.py and arch_util.py).
#
# NAFBlock, NAFNet, SimpleGate and LayerNorm2d:
# Copyright (c) 2022 megvii-model. All Rights Reserved. (https://github.com/megvii-research/NAFNet)
# Modified from BasicSR (https://github.com/xinntao/BasicSR), Copyright 2018-2020 BasicSR Authors
#
# Differences from the original code:
#   - hex_peak_mask puts the mask peaks at the lattice frequencies of the torch.fft.fft2 output.
#     The original builds its frequency grid in fft2 order and then applies ifftshift and a
#     transpose, so its peaks land half a spectrum away (and transposed) from the lattice
#     frequencies, and it fails for non-square inputs.
#   - The distance to a peak is measured on the periodic spectrum (wrapped), so a peak near
#     the Nyquist frequency keeps its full Gaussian shape.
#   - The lattice is described by its period (1 / frequency of the first-order spectral peaks)
#     and the angle of one first-order peak, both measured by estimate_lattice_period.py.
#     The original fixes the angles at 0, 60, ..., 300 degrees (= lattice_angle_deg 0).
#   - LayerNormFunction.backward uses ctx.saved_tensors instead of the deprecated ctx.saved_variables.
# ------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def wrap_nyquist(frequency):
    """Wraps a frequency (cycles per pixel, float or tensor) into [-0.5, 0.5)."""
    return ((frequency + 0.5) % 1.0) - 0.5


def hex_peak_mask(height, width, lattice_period_px, lattice_angle_deg=0.0, bandwidth=0.06, harmonics=(1, 2),
                  device=None):
    """Mask of shape [height, width] in torch.fft.fft2 order (DC at [0, 0], no fftshift).

    It is 1 at the peaks of a hexagonal lattice whose first-order spectral peaks have the frequency
    1 / lattice_period_px and lie at the angles lattice_angle_deg + 0, 60, ..., 300 degrees, and
    falls off as a Gaussian around them. Angles are measured from the column-frequency axis
    (fftfreq(width)) towards the row-frequency axis (fftfreq(height)). Frequencies above the
    Nyquist frequency are aliased (wrapped) to where the sampled feature map has them.
    """
    device = device or torch.device('cpu')
    fv = torch.fft.fftfreq(height, device=device)   # row frequency, cycles per pixel
    fu = torch.fft.fftfreq(width, device=device)    # column frequency, cycles per pixel
    V, U = torch.meshgrid(fv, fu, indexing='ij')     # [height, width]
    f0 = 1.0 / max(float(lattice_period_px), 1e-6)
    angles = [0, 60, 120, 180, 240, 300]
    M = torch.zeros((height, width), device=device)

    # The original also adds the mirrored peak (-u, -v) of every peak. The angles already contain
    # theta + 180 degrees, so that only doubles every term, which the normalisation below removes.
    s = bandwidth * 0.5
    for n in harmonics:
        fn = n * f0
        for ang in angles:
            rad = math.radians(ang + lattice_angle_deg)
            uc = wrap_nyquist(fn * math.cos(rad))
            vc = wrap_nyquist(fn * math.sin(rad))
            du = wrap_nyquist(U - uc)
            dv = wrap_nyquist(V - vc)
            M += torch.exp(-(du ** 2 + dv ** 2) / (2 * s * s))

    M = M / (M.max() + 1e-8)
    return M


class SpectralGate2D(nn.Module):

    def __init__(self, channels, lattice_period_px, lattice_angle_deg=0.0, bandwidth=0.06, harmonics=(1, 2),
                 alpha_max=0.7, init_alpha=0.0, per_channel=True):
        super().__init__()
        self.lattice_period_px = float(lattice_period_px)
        self.lattice_angle_deg = float(lattice_angle_deg)
        self.bandwidth = float(bandwidth)
        self.harmonics = tuple(harmonics)
        self.alpha_max = float(alpha_max)
        self.per_channel = per_channel

        if per_channel:
            self.alpha_raw = nn.Parameter(torch.full((channels, 1, 1), float(init_alpha)))
        else:
            self.alpha_raw = nn.Parameter(torch.tensor([[[float(init_alpha)]]]))  # [1,1,1]

        # The mask is recomputed from the settings above, so it is not part of the state_dict.
        self._mask = None
        self._mask_shape = None
        self._mask_device = None

    def _get_mask(self, H, W, device):
        if (self._mask is None) or (self._mask_shape != (H, W)) or (self._mask_device != device):
            M = hex_peak_mask(H, W, lattice_period_px=self.lattice_period_px,
                              lattice_angle_deg=self.lattice_angle_deg, bandwidth=self.bandwidth,
                              harmonics=self.harmonics, device=device)
            self._mask = M
            self._mask_shape = (H, W)
            self._mask_device = device
        return self._mask

    def forward(self, x):
        """
        x: [B,C,H,W]
        """
        B, C, H, W = x.shape
        device = x.device
        M = self._get_mask(H, W, device)                  # [H,W]
        alpha = torch.sigmoid(self.alpha_raw) * self.alpha_max
        if alpha.shape[0] != C:
            alpha = alpha.expand(C, 1, 1)

        dtype_in = x.dtype
        X = torch.fft.fft2(x.to(torch.float32), norm='ortho')     # complex64
        gate = (1.0 - alpha * M[None, None, ...])                 # [C,H,W]
        Y = X * gate                                              # [B,C,H,W]
        Z = torch.fft.ifft2(Y, norm='ortho')  # complex
        y = Z.real.contiguous()  # real part
        return y.to(dtype_in)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class SGARNet(NAFNet):
    """
    gate_lattice_period_px: period of the core lattice in input image pixels, i.e. 1 / frequency of the
        first-order spectral peaks (see estimate_lattice_period.py).
    gate_lattice_angle_deg: angle of one first-order spectral peak, in degrees.
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[],
                 gate_lattice_period_px=4.0, gate_lattice_angle_deg=0.0, gate_bandwidth=0.06,
                 gate_harmonics=(1,2), gate_alpha_max=0.7, gate_init_alpha=0.0,
                 gate_per_channel=True):
        super().__init__(img_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums)

        # Each encoder halves the resolution, so the lattice period is 2^len(encoders) times smaller at the bottleneck.
        self._down_scale = 2 ** len(self.encoders)
        period_bottleneck = gate_lattice_period_px / self._down_scale
        mid_channels = width * (2 ** len(enc_blk_nums))

        self.spectral_gate_mid = SpectralGate2D(
            channels=mid_channels,
            lattice_period_px=period_bottleneck,
            lattice_angle_deg=gate_lattice_angle_deg,
            bandwidth=gate_bandwidth,
            harmonics=gate_harmonics,
            alpha_max=gate_alpha_max,
            init_alpha=gate_init_alpha,
            per_channel=gate_per_channel
        )

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x = self.spectral_gate_mid(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]
