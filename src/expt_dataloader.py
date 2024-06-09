import numpy as np
import time
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import PIL.Image
import os 
from os.path import isfile, join
from scipy.io import loadmat 
from torchvision import transforms as tfm
import struct
import cv2
import skimage.io
import skimage.measure
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

class TestCanonRGBDataset(Dataset):
    def __init__(self, dataroot_dir, N_scenes, transform=None, scene_channels='dualpix_rgb', assets_dir='./assets/', calib_file_str='calib_whitesheet', roi=[800,1360,800,800]):
        self.dataroot_dir = dataroot_dir
        self.len = N_scenes
        self.transform = transform
        self.scene_channels = scene_channels
        self.roi = roi
        img_c = cv2.imread(os.path.join(assets_dir, calib_file_str+'_combined.png'), cv2.IMREAD_UNCHANGED)
        img_c = img_c[:,:,::-1]
        img_l = cv2.imread(os.path.join(assets_dir, calib_file_str+'_left.png'), cv2.IMREAD_UNCHANGED)
        img_l = img_l[:,:,::-1]
        # with PIL.Image.open(os.path.join(assets_dir, calib_file_str+'_combined.png')) as f:
        #     img_c = np.array(f)
        # with PIL.Image.open(os.path.join(assets_dir, calib_file_str+'_left.png')) as f: 
        #     img_l = np.array(f)
        img_c = (img_c.astype(np.float32)-511)/(2**14 -1)
        img_c[np.where(img_c<0)] = 0
        img_l = (img_l.astype(np.float32)-511)/(2**14 -1)
        img_l[np.where(img_l<0)] = 0
        img_l[np.where(img_l>0.5)] = 0.5
        # apply gaussian filtering
        img_c = gaussian_filter(img_c, sigma=(7,7,0))
        img_l = gaussian_filter(img_l, sigma=(7,7,0))
        img_r = img_c - img_l  
        img_l = img_l[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2],:]
        img_r = img_r[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2],:]

        M = np.maximum(np.amax(img_l),np.amax(img_r))
        self.calib_left = img_l/M
        self.calib_right = img_r/M
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        dp_combined_str = "{:03}_combined.png".format(idx)
        dp_left_str = "{:03}_left.png".format(idx)
        img_c = cv2.imread(os.path.join(self.dataroot_dir, dp_combined_str), cv2.IMREAD_UNCHANGED)
        img_c = img_c[:,:,::-1]
        img_l = cv2.imread(os.path.join(self.dataroot_dir, dp_left_str), cv2.IMREAD_UNCHANGED)
        img_l = img_l[:,:,::-1]
        # with PIL.Image.open(os.path.join(self.dataroot_dir, dp_combined_str)) as f:
        #     img_c = np.array(f)
        img_c = (img_c.astype(np.float32)-511)/(2**14 - 1)
        img_c[np.where(img_c<0)] = 0
        # with PIL.Image.open(os.path.join(self.dataroot_dir, dp_left_str)) as f:
        #     img_l = np.array(f)
        img_l = (img_l.astype(np.float32)-511)/(2**14 - 1)
        img_l[np.where(img_l<0)] = 0
        img_l[np.where(img_l>0.5)] = 0.5
        img_r = img_c - img_l

        img_l = img_l[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2],:]
        img_r = img_r[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2],:]

        img_l = img_l / (self.calib_left+1e-3)
        img_r = img_r / (self.calib_right+1e-3)
        if self.scene_channels=='dualpix_rgb':
            sample = np.stack((2*img_l[:,:,0], 2*img_r[:,:,0], 2*img_l[:,:,1], 2*img_r[:,:,1], 2*img_l[:,:,2], 2*img_r[:,:,2]), axis=0)
        elif self.scene_channels=='normalpix_rgb':
            sample = np.stack((img_l[:,:,0]+img_r[:,:,0], img_l[:,:,1]+img_r[:,:,1], img_l[:,:,2]+img_r[:,:,2]), axis=0)
        else: 
            raise NotImplementedError
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class TestCanonDataset(Dataset):
    def __init__(self, dataroot_dir, N_scenes, transform=None, scene_channels='dualpix_only', assets_dir='./assets/', calib_file_str='calib_whitesheet', roi=[800,1360,800,800]):
        self.dataroot_dir = dataroot_dir
        self.len = N_scenes
        self.transform = transform
        self.scene_channels = scene_channels
        self.roi = roi

        with PIL.Image.open(os.path.join(assets_dir, calib_file_str+'_combined.png')) as f:
            img_c = np.array(f)
        with PIL.Image.open(os.path.join(assets_dir, calib_file_str+'_left.png')) as f: 
            img_l = np.array(f)
        img_c = (img_c.astype(np.float32)-511)/(2**14 -1)
        img_c[np.where(img_c<0)] = 0
        img_l = (img_l.astype(np.float32)-511)/(2**14 -1)
        img_l[np.where(img_l<0)] = 0
        img_l[np.where(img_l>0.5)] = 0.5
        img_r = img_c - img_l  
        img_l = img_l[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]
        img_r = img_r[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]
        M = np.maximum(np.amax(img_l),np.amax(img_r))
        self.calib_left = img_l/M
        self.calib_right = img_r/M
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        dp_combined_str = "{:03}_combined.png".format(idx)
        dp_left_str = "{:03}_left.png".format(idx)
        with PIL.Image.open(os.path.join(self.dataroot_dir, dp_combined_str)) as f:
            img_c = np.array(f)
            img_c = (img_c.astype(np.float32)-511)/(2**14 - 1)
            img_c[np.where(img_c<0)] = 0
        with PIL.Image.open(os.path.join(self.dataroot_dir, dp_left_str)) as f:
            img_l = np.array(f)
            img_l = (img_l.astype(np.float32)-511)/(2**14 - 1)
            img_l[np.where(img_l<0)] = 0
            img_l[np.where(img_l>0.5)] = 0.5
        img_r = img_c - img_l

        img_l = img_l[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]]
        img_r = img_r[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]]
        img_l = img_l / (self.calib_left+1e-3)
        img_r = img_r / (self.calib_right+1e-3)
        if self.scene_channels=='dualpix+red+blue' or self.scene_channels=='dualpix_rgb' or self.scene_channels=='normalpix_rgb':
            raise NotImplementedError
        else: 
            sample = np.stack((2*img_l, 2*img_r), axis=0)
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class TestPixelXinDataset(Dataset):
    def __init__(self, dataroot_dir, N_scenes, transform=None):
        self.dataroot_dir = dataroot_dir 
        self.len = N_scenes 
        self.transform = transform 

        with PIL.Image.open(os.path.join(self.dataroot_dir, 'calibration/white_sheet_left.png')) as f:
            img_l = (np.array(f).astype(np.float32)-1024)/16383
            img_r = (np.array(f).astype(np.float32)-1024)/16383
        M = np.maximum(np.amax(img_l),np.amax(img_r))
        self.calib_left = img_l/M
        self.calib_right = img_r/M
    
    def __len__(self): 
        return self.len
    
    def __getitem__(self, idx):
        dp_left_str = "{:03}_left.png".format(idx+1)
        dp_right_str = "{:03}_right.png".format(idx+1)
        
        with PIL.Image.open(os.path.join(self.dataroot_dir, dp_left_str)) as f:
            img_l = np.array(f)
            img_l = (img_l.astype(np.float32)-1023)/(2**14 - 1)
            img_l[np.where(img_l<0)] = 0
            img_l = img_l / self.calib_left
        with PIL.Image.open(os.path.join(self.dataroot_dir, dp_right_str)) as f:
            img_r = np.array(f)
            img_r = (img_r.astype(np.float32)-1023)/(2**14 - 1)    # 1512 x 2016
            img_r[np.where(img_r<0)] = 0
            img_r = img_r / self.calib_right
        sample = np.stack((img_l, img_r), axis=0)
        if self.transform:
            sample = self.transform(sample)
        return sample

class TestPixelDataset(Dataset):
    def __init__(self, dataroot_dir, N_scenes, transform=None, scene_channels='dualpix_only', assets_dir='./assets/', calib_file_str='calib_whitesheet', roi=[800,1360,800,800]):
        self.dataroot_dir = dataroot_dir
        self.len = N_scenes
        self.transform = transform
        self.scene_channels = scene_channels
        self.roi = roi

        calib_data = loadmat(os.path.join(assets_dir, calib_file_str))
        self.calib_left = (calib_data['vig_l'].astype(np.float32)-4095)/(2**16 -1)
        self.calib_left[np.where(self.calib_left<0)] = 0
        self.calib_left = self.calib_left[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]]
        self.calib_left = np.repeat(self.calib_left, repeats=2, axis=0)
        self.calib_right = (calib_data['vig_r'].astype(np.float32)-4095)/(2**16 -1)
        self.calib_right[np.where(self.calib_right<0)] = 0
        self.calib_right = self.calib_right[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]]
        self.calib_right = np.repeat(self.calib_right, repeats=2, axis=0)
        M = np.maximum(np.amax(self.calib_left),np.amax(self.calib_right))
        self.calib_left = self.calib_left/M
        self.calib_right = self.calib_right/M

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        dp_left_str = "{:03}_left.pgm".format(idx+1)
        dp_right_str = "{:03}_right.pgm".format(idx+1)
        
        with PIL.Image.open(os.path.join(self.dataroot_dir, dp_left_str)) as f:
            img_l = np.array(f)
            img_l = (img_l.astype(np.float32)-4095)/(2**16 - 1)
            img_l[np.where(img_l<0)] = 0
            img_l = img_l[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]]
            img_l = np.repeat(img_l, repeats=2, axis=0)
            img_l = img_l / self.calib_left
        with PIL.Image.open(os.path.join(self.dataroot_dir, dp_right_str)) as f:
            img_r = np.array(f)
            img_r = (img_r.astype(np.float32)-4095)/(2**16 - 1)
            img_r[np.where(img_r<0)] = 0
            img_r = img_r[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]]
            img_r = np.repeat(img_r, repeats=2, axis=0)
            img_r = img_r / self.calib_right
        if self.scene_channels=='dualpix+red+blue':
            raw_img_str = "{:03}_raw.pgm".format(idx+1)
            with PIL.Image.open(os.path.join(self.dataroot_dir, raw_img_str)) as f: 
                raw_img = np.array(f).astype(np.float32)/(2**16 - 1)    # 3024 x 4032
                img_red = raw_img[0::2,0::2]    # 1512 x 2016
                img_blue = raw_img[1::2,1::2]   # 1512 x 2016
                img_red = 2*skimage.measure.block_reduce(img_red, (2,2), np.mean)     # 756 x 1008
                img_blue = 2*skimage.measure.block_reduce(img_blue, (2,2), np.mean)   # 756 x 1008
                img_red = np.repeat(np.repeat(img_red, 2, axis=0), 2, axis=1)       # 1512 x 2016
                img_blue = np.repeat(np.repeat(img_blue, 2, axis=0), 2, axis=1)     # 1512 x 2016
            sample = np.stack((img_l, img_r, img_red, img_blue), axis=0)
        if self.scene_channels=='dualpix_rgb' or self.scene_channels=='normalpix_rgb':
            raise NotImplementedError
        else:
            sample = np.stack((img_l, img_r), axis=0)
        if self.transform:
            sample = self.transform(sample)
        return sample

class TestCanonDPDBlurDataset(Dataset):
    def __init__(self, dataroot_dir, N_scenes, transform=None): 
        self.dataroot_dir = dataroot_dir
        self.len = N_scenes
        self.transform = transform
        self.all_dp_left_imgfiles = [join(dataroot_dir, 'canon/test_l/source/', f) \
                            for f in os.listdir(join(dataroot_dir, 'canon/test_l/source/')) \
                            if isfile(join(dataroot_dir, 'canon/test_l/source/', f))]
        self.all_dp_left_imgfiles.sort()
        self.all_dp_right_imgfiles = [f.replace('/test_l/','/test_r/').replace('_L.png','_R.png') for f in self.all_dp_left_imgfiles]
        self.all_gt_imgfiles = [join(dataroot_dir, 'canon/test_c/target/', f) \
                            for f in os.listdir(join(dataroot_dir, 'canon/test_c/target/')) \
                            if isfile(join(dataroot_dir, 'canon/test_c/target/', f))]
        self.all_gt_imgfiles.sort()
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx): 
        gt_str = self.all_gt_imgfiles[idx]
        dp_left_str = self.all_dp_left_imgfiles[idx]
        dp_right_str = self.all_dp_right_imgfiles[idx]
        img_l = cv2.imread(os.path.join(self.dataroot_dir, dp_left_str), cv2.IMREAD_UNCHANGED)
        img_l = (img_l[:,:,::-1].astype(np.float32)-0)/(2**16 -1)
        img_r = cv2.imread(os.path.join(self.dataroot_dir, dp_right_str), cv2.IMREAD_UNCHANGED)
        img_r = (img_r[:,:,::-1].astype(np.float32)-0)/(2**16 -1)

        img_gt = cv2.imread(os.path.join(self.dataroot_dir, gt_str), cv2.IMREAD_UNCHANGED)
        img_gt = (img_gt[:,:,::-1].astype(np.float32)-0)/(2**16 -1)
        img_gt = np.transpose(img_gt, axes=(2,0,1))

        sample = np.stack((img_l[:,:,0], img_r[:,:,0], img_l[:,:,1], img_r[:,:,1], img_l[:,:,2], img_r[:,:,2]), axis=0)
        
        if self.transform:
            sample = self.transform(sample)
            img_gt = self.transform(img_gt)
        return sample, img_gt



def prepare_test_dataset(args):
    if args.sensor=='pixel-xin': 
        dataset = TestPixelXinDataset(args.dataroot, N_scenes=args.N_renders_override, 
                                      transform=tfm.Compose([to_tensor(), tfm.CenterCrop((1008, 1344))]))
    if args.sensor=='pixel':
        dataset = TestPixelDataset(args.dataroot, N_scenes=args.N_renders_override, \
                        transform=tfm.Compose([to_tensor()]), \
                        scene_channels=args.scene_channels, assets_dir=args.assets_dir, calib_file_str=args.calib_file, 
                        roi=args.roi)
    if args.sensor=='canon':
        if args.scene_channels=='dualpix_rgb' or args.scene_channels=='normalpix_rgb':
            dataset = TestCanonRGBDataset(args.dataroot, N_scenes=args.N_renders_override, \
                        transform=tfm.Compose([to_tensor()]),  #transform=tfm.Compose([to_tensor(), tfm.CenterCrop(args.image_size)]), \
                        scene_channels=args.scene_channels, assets_dir=args.assets_dir, calib_file_str=args.calib_file, 
                        roi=args.roi)
        if args.scene_channels=='dualpix_only' or args.scene_channels=='normalpix_only': 
            dataset = TestCanonDataset(args.dataroot, N_scenes=args.N_renders_override, \
                        transform=tfm.Compose([to_tensor(), tfm.CenterCrop(args.image_size)]), \
                        scene_channels=args.scene_channels, assets_dir=args.assets_dir, calib_file_str=args.calib_file, 
                        roi=args.roi)
    if args.sensor=='canon-dpdblur':
        if args.scene_channels=='dualpix_rgb':
            dataset = TestCanonDPDBlurDataset(args.dataroot, N_scenes=args.N_renders_override, \
                                              transform=tfm.Compose([to_tensor()]))
        else: 
            raise NotImplementedError
    return dataset
    
class to_tensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample)
