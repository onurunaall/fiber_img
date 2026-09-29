import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# from functools import reduce


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



# Calculate padding "same" for stride = 2
def Conv2d_stride2(kernel_size, dilation):
    padding = math.ceil(0.5*(kernel_size*dilation-dilation-2+1))
    return padding


# Downsampling Block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, inplace=False):
        super(DownBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # padding = ceil(0.5*((out-1)*stride+1+dilation*(kernel-1)-in))
        padding = Conv2d_stride2(self.kernel_size, self.dilation)
                
        self.convBlock1 = nn.Sequential(
                # nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, dilation=dilation, bias=False),
                # nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),

                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)
        
        self.convBlock2 = nn.Sequential(
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),
                
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=True)        

    def forward(self, x):
        
        x1 = self.convBlock1(x)
        skip = self.skip(x)
        x1 += skip
    
        return torch.add(x1, self.convBlock2(x1))
        # return self.layers(x)


# Upsampling Block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, inplace=False):
        super(UpBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
                        
        self.convBlock1 = nn.Sequential(
                # nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=inplace),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
                
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)
                
        self.convBlock2 = nn.Sequential(
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),
                
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)
        
        
        self.skip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)        

    def forward(self, x):
        
        x1 = self.convBlock1(x)
        skip = self.skip(x)
        x1 += skip
    
        # return self.convBlock2(x1)
        return torch.add(x1, self.convBlock2(x1))


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], outScale=1, inplace=False):
        super(UNET, self).__init__()

        if outScale not in (1, 2):
            raise ValueError('outScale must be 1 or 2, got %s' % outScale)

        self.unetEncoder = nn.ModuleList()
        self.unetDecoder = nn.ModuleList()
            
        Feats = features
        
        # Conv in
        self.ConvIn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding='same', bias=True),
            nn.ReLU(inplace=inplace),
            )
        
        # Encoder
        self.unetEncoder.append(DownBlock(8, Feats[0], kernel_size=9, dilation=3, inplace=inplace))
        self.unetEncoder.append(DownBlock(Feats[0], Feats[1], kernel_size=3, dilation=1, inplace=inplace))
        self.unetEncoder.append(DownBlock(Feats[1], Feats[2], kernel_size=3, dilation=1, inplace=inplace))
        self.unetEncoder.append(DownBlock(Feats[2], Feats[3], kernel_size=3, dilation=1, inplace=inplace))
        
        # Decoder
        revFeats = features[::-1]
        self.unetDecoder.append(UpBlock(revFeats[0], revFeats[1], kernel_size=3, inplace=inplace))
        self.unetDecoder.append(UpBlock(revFeats[1], revFeats[2], kernel_size=3, inplace=inplace))
        if outScale <= 2:
            self.unetDecoder.append(UpBlock(revFeats[2], revFeats[3], kernel_size=3, inplace=inplace))
            ConvOut_in = revFeats[3]
            if outScale == 1:
                self.unetDecoder.append(UpBlock(revFeats[3], out_channels=out_channels, kernel_size=3, inplace=inplace))
                ConvOut_in = out_channels
        
        # Conv out
        self.ConvOut = nn.Conv2d(ConvOut_in, out_channels, kernel_size=3, padding='same', bias=True)
        
    
    def forward(self, x):
        factor = 2 ** len(self.unetEncoder)   # 4 down blocks -> 16

        assert x.shape[-2] % factor == 0 and x.shape[-1] % factor == 0, \
            'Height and width must be divisible by %d, got %s' % (factor, tuple(x.shape[-2:]))

        skip_connects = []
        
        x = self.ConvIn(x)
        
        for downblock in self.unetEncoder:
            x = downblock(x)
            skip_connects.append(x)
    
        skip_connects.pop(-1)
        skip_connects = skip_connects[::-1]
        
        for ii in range(0, len(self.unetDecoder)-1):
            x = self.unetDecoder[ii](x)
            # x = torch.add(x, skip_connects[ii], alpha=0.5)
            x = torch.add(x, skip_connects[ii])
            
        x = self.unetDecoder[-1](x)
        return self.ConvOut(x)
            

class EDSR(nn.Module):
    def __init__(self, in_channels=1, n_colors=1, n_resblocks=32, n_feats=256, scale=2, conv=default_conv, rgb_range=1):
        super(EDSR, self).__init__()
        
        in_channels = in_channels
        n_resblocks = n_resblocks
        n_feats = n_feats
        scale = scale
        n_colors = n_colors
        rgb_range = rgb_range
        
        kernel_size = 3 
        act = nn.ReLU(True)
        res_scale = 1
         

        # head module
        m_head = [conv(in_channels, n_feats, kernel_size)]

        # body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # tail module
        # m_tail = [Upsampler(conv, scale, n_feats, act=False),
        #           conv(n_feats, n_colors, kernel_size)]
        m_tail = []
        if scale >= 2:
            m_tail.append(Upsampler(conv, scale, n_feats, act=False))  
        m_tail.append(conv(n_feats, n_colors, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    
    # @autocast()
    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 


class UNET_EDSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], n_resblocks=32, n_feats=256, scale=1, 
                 n_colors=1, rgb_range=1, inplace=True):
        super(UNET_EDSR, self).__init__()

        in_channels = in_channels
        out_channels = out_channels
        n_colors = n_colors
        rgb_range = rgb_range
        # unet
        features = features
        inplace = inplace
        # EDSR
        n_resblocks = n_resblocks
        n_feats = n_feats
        scale = scale
        
        # self.unet = UNET(in_channels=in_channels, out_channels=8, outScale=1, inplace=inplace)
        
        # unet new version
        from models.model_unet import UNET as unet_256
        self.unet = unet_256(in_channels=in_channels, out_channels=8, features=features, inplace=False)
        
        self.edsr = EDSR(in_channels=8, n_colors=1, n_resblocks=n_resblocks, n_feats=n_feats, scale=scale, rgb_range=1)
        
    # @autocast()
    def forward(self, x):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        
        # start.record()
        x = self.unet(x)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'{start.elapsed_time(end)} ms')    # ms
        
        # start.record()
        x = self.edsr(x)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'{start.elapsed_time(end)} ms')    # ms
        
        return x

def test():
    x = torch.randn((1, 1, 960, 960))    # b, c, h, w
    # model = UNET(in_channels=x.shape[1], out_channels=x.shape[1])
    model = UNET_EDSR(in_channels=1, out_channels=1, features=[32, 64, 128, 256], inplace=False, 
                      n_resblocks=16, n_feats=128, scale=1, n_colors=1, rgb_range=1)
    preds = model(x)
    print(preds.shape)
    # assert preds.shape == x.shape
    
    # Visualize NN structure
    from torchview import draw_graph
    model_graph = draw_graph(model, input_size=(8, 1, 960, 960), device='meta')
    model_graph.visual_graph.render(format='png')
    
    
if __name__ == "__main__":
    test()