import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio 
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt


class LearnedCode(nn.Module):
	def __init__(self,blur_sizes,alpha_init=2.5,init_mask_fpath=None,keep_mask_fixed=False):
		super(LearnedCode, self).__init__()
		self.blur_sizes = np.load(blur_sizes)
		# blur_sizes = np.load(blur_sizes)
		# self.register_buffer('blur_sizes', torch.from_numpy(blur_sizes))
		if init_mask_fpath is None: 
			aperture = np.random.randn(16,16).astype(np.float32)
			[X,Y] = np.mgrid[-7.5:7.6, -7.5:7.6]
			circ_roi = np.ones((16,16)).astype(np.float32)
			aperture[np.where((X**2+Y**2)>=8**2)] = -10 	# mask pixels outside the circle are zero
			circ_roi[np.where((X**2+Y**2)>=8**2)] = 0
			self.aperture_ = nn.Parameter(torch.from_numpy(aperture).unsqueeze(0).unsqueeze(0))
			self.init_blur_size = 16
			# self.aperture_ = nn.Parameter(torch.randn(1,1,max(self.blur_sizes),max(self.blur_sizes)))
		else:
			self.aperture_ = cv2.imread(init_mask_fpath, 0).astype(np.float32)
			self.aperture_ = self.aperture_/np.max(self.aperture_)
			N = self.aperture_.shape[-1]
			self.init_blur_size = N
			self.aperture_ = 2*(self.aperture_ - 0.5) 	# convert to [-1, 1] range
			self.aperture_ = torch.from_numpy(self.aperture_).unsqueeze(0).unsqueeze(0)
			self.aperture_ = nn.Parameter(self.aperture_, requires_grad=(not keep_mask_fixed))
			[X,Y] = np.mgrid[-1*((N-1)/2):((N-1)/2)+0.1, -1*((N-1)/2):((N-1)/2)+0.1]
			circ_roi = np.ones((N,N)).astype(np.float32)
			circ_roi[np.where((X**2+Y**2)>=(N/2)**2)] = 0
		self.register_buffer('circ_roi', torch.from_numpy(circ_roi).unsqueeze(0).unsqueeze(0))
		self.alpha_init = alpha_init
		self.alpha_t = alpha_init

	def forward(self,psf, training=False, psf_provided=False):
		psf = psf[0]
		test_mode = not training #this becomes True if model is in eval() mode
		psf_size = psf.shape[-1] ##assuming square PSF of shape ...HxW
		mask_pattern = torch.sigmoid(self.alpha_t*self.aperture_)*self.circ_roi
		# mask_pattern = F.hardsigmoid(self.alpha_t*self.aperture_)*self.circ_roi
		if test_mode:
			mask_pattern = (mask_pattern > 0.5).float()
		light_efficiency_factor = mask_pattern.sum()/((np.pi/4)*(self.init_blur_size)**2)

		coded_psfs = []

		transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])

		for i in range(len(self.blur_sizes)):
			blur_size = self.blur_sizes[i]
			pad_size = psf_size-blur_size
			temp_psf = F.interpolate(mask_pattern,size=(blur_size,blur_size),mode='bilinear')
			temp_psf_pad = F.pad(temp_psf,(pad_size//2,pad_size//2,pad_size//2,pad_size//2))
			if i<=len(self.blur_sizes)//2:
				coded_psfs.append(temp_psf_pad)
			else:
				coded_psfs.append(transform(temp_psf_pad))
		
		coded_psfs = torch.cat(coded_psfs,1)
		coded_dp_psfs = psf * coded_psfs 
		if psf_provided:
			coded_dp_psfs = psf
		coded_dp_psfs = coded_dp_psfs / torch.sum(coded_dp_psfs, dim=(-2,-1), keepdim=True)
		# coded_dp_psfs = coded_dp_psfs * light_efficiency_factor
		return coded_dp_psfs.unsqueeze(0), light_efficiency_factor

	def update_alpha(self,num_iters):
		#accepts the total num of iters and updates alpha accordingly
		#call this function after every train iteration
		self.alpha_t = self.alpha_init + (num_iters/8000)

