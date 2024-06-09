import torch 
import numpy as np
from torchvision import transforms as tfm


class to_tensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample)

class RescaleDepthMapRng(object):
    """Re-scale depth map to suggested depth ranges"""
    def __init__(self, depth_min, depth_max):
        self.depth_min = depth_min 
        self.depth_max = depth_max

    def __call__(self, sample):
        rng1 = np.minimum(1+0.05*np.abs(np.random.randn(1)), 1.20)
        rng2 = np.maximum(1-0.05*np.abs(np.random.randn(1)), 0.80)
        sample[0,:,:] = (rng1*self.depth_min) + sample[0,:,:]*((rng2*self.depth_max) - (rng1*self.depth_min))
        return sample

class RescaleDepthMap(object):
    """Re-scale depth map to suggested depth ranges"""
    def __init__(self, depth_min, depth_max):
        self.depth_min = depth_min 
        self.depth_max = depth_max

    def __call__(self, sample):
        sample[0,:,:] = self.depth_min + sample[0,:,:]*(self.depth_max - self.depth_min)
        return sample

class RandomAugment(object):
    """Randomly change the brightness and sharpness"""
    def __init__(self):
        self.jitter_fn = tfm.ColorJitter(brightness=[0.5, 1.1], contrast=0.2)

    def __call__(self, sample):
        sample[1:,:,:] = self.jitter_fn(sample[1:,:,:])
        return sample

class RandomAugment_dualpix_rgb(object):
    """Randomly change the brightness and sharpness"""
    def __init__(self): 
        pass 

    def __call__(self, sample): 
        brightness_factor = float(torch.empty(1).uniform_(0.5, 1.1))
        contrast_factor = float(torch.empty(1).uniform_(0.2,1.2))
        gamma_factor = float(torch.empty(1).uniform_(0.8,1.2))
        hue_factor = float(torch.empty(1).uniform_(-0.25,0.25))
        fn_idx = torch.randperm(4)
        img = sample[1::2,:,:]
        for fn_id in fn_idx: 
            if fn_id == 0:
                img = tfm.functional.adjust_brightness(img, brightness_factor)
            elif fn_id == 1:
                img = tfm.functional.adjust_contrast(img, contrast_factor)
            elif fn_id == 2:
                img = tfm.functional.adjust_saturation(img, gamma_factor)
            elif fn_id == 3:
                img = tfm.functional.adjust_hue(img, hue_factor)
        sample[1::2,:,:] = img 
        sample[2::2,:,:] = img
        return sample

class RandomAugment_stdpix_rgb(object):
    """Randomly change the brightness and sharpness"""
    def __init__(self): 
        pass 

    def __call__(self, sample): 
        brightness_factor = float(torch.empty(1).uniform_(0.5, 1.1))
        contrast_factor = float(torch.empty(1).uniform_(0.2,1.2))
        gamma_factor = float(torch.empty(1).uniform_(0.8,1.2))
        hue_factor = float(torch.empty(1).uniform_(-0.25,0.25))
        fn_idx = torch.randperm(4)
        img = sample[1:,:,:]
        for fn_id in fn_idx: 
            if fn_id == 0:
                img = tfm.functional.adjust_brightness(img, brightness_factor)
            elif fn_id == 1:
                img = tfm.functional.adjust_contrast(img, contrast_factor)
            elif fn_id == 2:
                img = tfm.functional.adjust_saturation(img, gamma_factor)
            elif fn_id == 3:
                img = tfm.functional.adjust_hue(img, hue_factor)
        sample[1:,:,:] = img 
        return sample