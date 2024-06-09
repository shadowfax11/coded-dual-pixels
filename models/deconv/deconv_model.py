import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class DeconvModel(nn.Module):
    def __init__(self, args):
        super(DeconvModel, self).__init__()

    def forward(self, x, h):
        """
        Perform deconvolution by convolving with conjugate of the PSF
        Args:
            x (torch.Tensor): B x C x H x W 
            h (torch.Tensor): 1 x C x D x H x W
        Returns:
            y (torch.Tensor): B x D x H x W
        """
        _, _, h, w = x.shape
        pad_x = [torch.div(h - psf.shape[-2], 2, rounding_mode='trunc'), 
                    torch.div(w - psf.shape[-1], 2, rounding_mode='trunc')]
        if psf.shape[-1]%2==1:
            psf = F.pad(psf, (pad_x[1]+1, pad_x[1], pad_x[0]+1, pad_x[0]), 'constant', 0)        
        else:
            psf = F.pad(psf, (pad_x[1], pad_x[1], pad_x[0], pad_x[0]), 'constant', 0)
        h = torch.flip(h, [-2, -1])
        
        x = x.unsqueeze(2)
        M = x.max()
        X = fft,rfft2(x/M)
        H = fft.rfft2(fft.ifftshift(h, (-2,-1)))
        Y = (X*H)
        y = fft.irfft2(Y)           # B x C x D x H x W
        y = torch.mean(y, dim=1)    # B x D x H x W
        return y

class WienerNet(torch.nn.Module):
    def __init__(self, args, psf):
        super(WienerNet, self).__init__()

        self.psfs = psf
        self.stack_size = psf.shape[-3]
        impulse = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        # self.batch_size = batch_size
        regf_2 = impulse.unsqueeze(0).repeat(self.stack_size, 1, 1)
        regf_2 = regf_2.unsqueeze(0).unsqueeze(0).float()
        
        self.reg_filter = torch.nn.Parameter(regf_2)
        self.lambd = torch.nn.Parameter(torch.ones(size = [1, 1, self.stack_size, 1, 1])) # 3 -> channels
            
    def forward(self, meas):

        # Reshaping psfs to match scene
        h, w = meas.shape[-2:]
        pad_x = [torch.div(h - self.psfs.shape[-2], 2, rounding_mode='trunc'), 
                    torch.div(w - self.psfs.shape[-1], 2, rounding_mode='trunc')]
        if self.psfs.shape[-1]%2==1:
            psf_padded = F.pad(self.psfs, (pad_x[1]+1, pad_x[1], pad_x[0]+1, pad_x[0]), 'constant', 0)        
        else:
            psf_padded = F.pad(self.psfs, (pad_x[1], pad_x[1], pad_x[0], pad_x[0]), 'constant', 0)

        left = int((psf_padded.shape[-2]-3)//2)
        top = int((psf_padded.shape[-1]-3)//2)
        pad = (top+1, top, left+1, left)
        reg_padded = torch.nn.functional.pad(self.reg_filter, pad, "constant", 0)
        regfft = torch.fft.fft2((torch.fft.fftshift(reg_padded, dim=(-2,-1))))#;print(regfft.shape) ##assuming 1x25xHxWx2

        Hr = torch.fft.fft2((psf_padded[0,:,:,:,:].unsqueeze(0))) ##assuming 1xDxHxW
        invFiltr = torch.conj_physical(Hr)/(torch.abs(Hr)**2 + (self.lambd**2)*torch.abs(regfft)**2+(1e-6))
        measrfft = torch.fft.fft2(meas)
        DR = measrfft*invFiltr
    
        deconvolvedr =  torch.real(torch.fft.ifft2(DR)) #Bx25xHxWx1 #should be real
        deconvolved = torch.fft.ifftshift(deconvolvedr, dim=(-2,-1))

        deconvolved_max = deconvolved.reshape(deconvolved.size(0), deconvolved.size(1), -1).max(2)[0]
        deconvolved_max = deconvolved_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = deconvolved/deconvolved_max

        return deconvolved/deconvolved_max  # (1,1,D,H,W)