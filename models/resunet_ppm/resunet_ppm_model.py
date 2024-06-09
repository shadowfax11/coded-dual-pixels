"""
Author: Vivek Boominathan
Website: https://vivekboominathan.com/

Stripped down from https://github.com/lucidrains/denoising-diffusion-pytorch
Modified with added Pyramid Pooling Module (PPM from PSPNet)

model = ResUnet_VB(channels=3, dim=16, out_dim=1, dim_mults=(1,2,4,8), resnet_block_groups=8)
"""

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from einops import reduce
from einops.layers.torch import Rearrange

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        x = self.act(x)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, groups = 8):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)
    

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = nn.ModuleList([])
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=reduction_dim), # original paper used nn.BatchNorm2d(reduction_dim),
                nn.SiLU(inplace=True), # original paper used nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[-2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
    

class ResUnet_VB(nn.Module):
    def __init__(
        self,
        channels,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups = 8,
        use_ppm = True,
        bins_ppm = (1, 2, 3, 6),
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.use_ppm = use_ppm
        self.bins_ppm = bins_ppm
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(self.channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block = block_klass(mid_dim, mid_dim)

        if use_ppm:
            self.ppm = nn.Sequential(
                PPM(mid_dim, int(mid_dim/len(self.bins_ppm)), self.bins_ppm),
                block_klass(2*mid_dim, mid_dim)
            )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim)

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.output_layer = nn.Conv2d(dim, self.out_dim, 1, bias=True)

    def forward(self, captimgs, *args, **kwargs):

        x = self.init_conv(captimgs)
        r = x.clone()

        h = []

        for block, downsample in self.downs:
            x = block(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block(x)

        if self.use_ppm:
            x = self.ppm(x)

        for block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)

        est = self.output_layer(x)

        return est