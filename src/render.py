import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torchvision
import numpy as np

def defocus_to_depth(defocus_map, L, f, g):
    m = f/g
    return (L*f)/(L*m - defocus_map*(1-m))

def normalized_depth_to_normalized_defocus(depth_map, L=12.5, f=50, g=400, p=10.72, def_max=40):
    m = f/g
    z_min = (L*f)/(L*m + def_max*(1-m))
    z_max = (L*f)/(L*m - def_max*(1-m))
    depth_map = (z_min) + (depth_map*(z_max-z_min))
    defocus_map = (L*f/(1-m))*((1/g) - (1/depth_map))
    defocus_map = defocus_map*1000/(p*def_max)
    return defocus_map

def depth_to_defocus(depth_map, L, f, g):
    return (L*f/(1-(f/g)))*((1/g) - (1/depth_map))

def over_op(alpha):
    bs, ps, cs, ds, hs, ws = alpha.shape
    out = torch.cumprod(1. - alpha, dim=-3)
    return torch.cat([torch.ones((bs, ps, cs, 1, hs, ws), 
                dtype=out.dtype, device=out.device), out[:, :, :, :-1]], dim=-3)

def depthmap_to_layereddepth(depthmap, num_depths, depth_vals, binary=False):
    # depthmap = depthmap[:, None, None, ...]  # add polarization and wavelength channel dim
    depth_bins = torch.argmin(torch.abs(depthmap[:,:,:,None,:,:] - depth_vals[None, None, None, :, None, None]), 3)
    layered_depth = torch.cat([(torch.eq(depth_bins,i)).float()[:,:,:,None,:,:] for i in range(num_depths)], 3)
    return layered_depth

class Convolve3DFFT(nn.Module):
    def __init__(self, args):
        super(Convolve3DFFT, self).__init__()
        self.register_buffer('z_N', torch.tensor(args.num_depth_planes))
        z_vals = args.z_vals_mm
        self.register_buffer('z_vals', torch.tensor(z_vals))
        self.rendering_algorithm = args.rendering_algorithm     # choice of occlusion-aware rendering algorithm

    def render_3dscene(self, x, d, h, eps=1e-3):
        """
        Renders the image of a 3D scene, using occlusion-aware matting (Hayato Ikoma et al. 2021)
        Terms: 
            Batch size [B]
            Polarization channels [P]
            Color channels [C]
            Depth channels [D]
            Image height [H]
            Image width [W]
        x [torch.Tensor]: (all-in-focus) scene. Shape B x 1 x C[or 1] x H x W
        d [torch.Tensor]: depth map of the scene. Shape B x 1 x 1 x H x W
        h [torch.Tensor]: Depth-dependent PSFs. Shape 1 x 1 x C x D x H x W
        """
        layered_depth = depthmap_to_layereddepth(d, self.z_N, self.z_vals, binary=True)     # B x 1 x 1 x D x H x W
        if self.rendering_algorithm=='ikoma-modified':
            # 3x3 MaxPool2D
            layered_depth_interpolated = F.max_pool2d(layered_depth[:,0,0,...], 3, padding=1, stride=1)
            # 2x1x1 AvgPool3D
            layered_depth_interpolated = F.avg_pool3d(layered_depth_interpolated, kernel_size=(2,1,1), padding=0, stride=1)
            # only layered depths having consecutive non-zeros depth layers are kept 
            layered_depth_interpolated = (layered_depth_interpolated>0).unsqueeze(1).unsqueeze(1) 
            # last depth layer after avg pooling does not exist so append the last original layered depth
            layered_depth_interpolated = torch.cat([layered_depth_interpolated, layered_depth[:,:,:,-1,...].unsqueeze(-3)], dim=-3)
            # sum across depth dimensions and normalize alpha maps
            layered_depth_interpolated = layered_depth_interpolated / torch.sum(layered_depth_interpolated, dim=-3, keepdim=True)
            # Shape B x 1 x C x D x H x W
            s = (layered_depth_interpolated) * x[:,:,:,None,...]
            layered_depth = layered_depth_interpolated
        else:
            s = layered_depth * x[:,:,:,None,...]   # Shape B x 1 x 1 x D x H x W
        
        M = s.max()
        s = s/M
        S = fft.rfft2(s)
        h = fft.ifftshift(h,(-2,-1))
        H = fft.rfft2(h)

        if self.rendering_algorithm=='naive':
            Y = (S*H)
            y = fft.irfft2(Y).sum(dim=-3)
        else:
            Flayered_depth = fft.rfft2(layered_depth)
            blurred_alpha_rgb = fft.irfft2(Flayered_depth*H)
            blurred_volume = fft.irfft2(S*H)
            
            # normalize blurred intensity 
            if self.rendering_algorithm=='ikoma2021' or self.rendering_algorithm=='ikoma-modified':
                cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
                Fcumsum_alpha = fft.rfft2(cumsum_alpha)
                blurred_cumsum_alpha = fft.irfft2(Fcumsum_alpha*H)
                blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
                blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)
            elif self.rendering_algorithm=='xin2021':
                blurred_volume = blurred_volume / 1
                blurred_alpha_rgb = blurred_alpha_rgb / 1

            over_alpha = over_op(blurred_alpha_rgb)
            y = torch.sum(over_alpha * blurred_volume, dim=-3)
        y = M*y

        return y

    def forward(self, x_scene, x_depth, psf):
        _, h, w = x_depth.shape

        # pad to make the scene and psf have the same dimensions
        pad_x = [torch.div(h - psf.shape[-2], 2, rounding_mode='trunc'), 
                torch.div(w - psf.shape[-1], 2, rounding_mode='trunc')]
        if psf.shape[-1]%2==1:
            psf = F.pad(psf, (pad_x[1]+1, pad_x[1], pad_x[0]+1, pad_x[0]), 'constant', 0)        
        else:
            psf = F.pad(psf, (pad_x[1], pad_x[1], pad_x[0], pad_x[0]), 'constant', 0)
        
        x_scene = x_scene.unsqueeze(1)                      # B x 1 x C x H x W
        if len(x_scene.shape)==4:                           # means that C=1, hence need to add another dimension
            x_scene = x_scene.unsqueeze(1)                  # B x 1 x 1 x H x W
        x_depth = x_depth.unsqueeze(1).unsqueeze(1)         # B x 1 x 1 x H x W
        psf = psf.unsqueeze(1)                              # 1 x 1 x C x D x H x W
        x = self.render_3dscene(x_scene, x_depth, psf)

        return torch.squeeze(x, dim=1)

def add_noise_poisson(x, num_photons):
    if num_photons:
        x = x*num_photons
        y = torch.poisson(x)/num_photons
        return y 
    else:
        return x

def add_noise_readout(x, readout_noise_level):
    x = x + (readout_noise_level*torch.randn(x.shape).to(x.device))
    return x

def add_noise_poisson_gauss(x, map, a, b): 
    x = x + ((a*map + b)*torch.randn(x.shape).to(x.device))
    return x

def clip_signal(x):
    return torch.clamp(x, min=0, max=1)
