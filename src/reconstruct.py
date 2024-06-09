import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.fft as fft
from utils.pytorch_utils import var_tensor

class Deconvolve3DFFTFast(nn.Module):
    def __init__(self, h_psf, w_psf):
        """
        psf(torch.Tensor) has size D x H x W
        """
        super(Deconvolve3DFFTFast, self).__init__()

        self.register_buffer('h_psf', var_tensor(h_psf))
        self.register_buffer('w_psf', var_tensor(w_psf))
        pad_h = [ torch.div(self.h_psf,2,rounding_mode='trunc'),
                 torch.div(self.w_psf,2,rounding_mode='trunc') ]
        self.register_buffer('pad_h', var_tensor(pad_h))

    def forward(self, x, h_adj, Txy):
        _, _, h, w = x.shape
        pad_x = [ int(h//2), int(w//2)]
        H = F.pad(h_adj, (pad_x[1],pad_x[1],pad_x[0],pad_x[0]), 'constant', 0)
        H = fft.fft2(fft.ifftshift(H,(2,3)))
        x = F.pad(x, (self.pad_h[1],self.pad_h[1],self.pad_h[0],self.pad_h[0]), 'constant', 0)
        x = fft.fft2(x)
        x = H*x
        x = torch.real(fft.ifft2(x))
        # x = torch.sum(x, 1).unsqueeze(0)
        x = x[:, :, self.pad_h[0]:self.pad_h[0]+h, self.pad_h[1]:self.pad_h[1]+w]
        x = torchvision.transforms.functional.affine(x, 0, Txy, 1.0, 0.0)
        return x