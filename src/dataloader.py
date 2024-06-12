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
from glob import glob

from utils.io import read_pfm
from utils.transforms import *

class nyuD2TestDataset(Dataset): 
    def __init__(self, dataroot_dir, train=False, N_scenes=0, transform=None, scene_channels='dualpix_mono'):
        if train:
            raise NotImplementedError
        else: 
            all_gt_scene_files = [join(dataroot_dir, 'nyu2_test/', f) \
                                for f in os.listdir(join(dataroot_dir, 'nyu2_test/')) \
                                if isfile(join(dataroot_dir, 'nyu2_test/', f)) and 'colors' in f]
            all_gt_scene_files.sort()
            all_gt_depth_files = [join(dataroot_dir, 'nyu2_test/', f) \
                                for f in os.listdir(join(dataroot_dir, 'nyu2_test/')) \
                                if isfile(join(dataroot_dir, 'nyu2_test/', f)) and 'depth' in f]
            all_gt_depth_files.sort()
        if N_scenes: 
            indices_list = np.arange(N_scenes)
        else:
            N_scenes = len(all_gt_scene_files)
            indices_list = np.arange(N_scenes)
        old_seed = np.random.get_state()
        np.random.seed(0)
        if train:
            indices_list_perm = np.random.permutation(indices_list)
        else: 
            indices_list_perm = indices_list
        np.random.set_state(old_seed)
        self.all_gt_scene_files = [all_gt_scene_files[x] for x in indices_list_perm]
        self.all_gt_depth_files = [all_gt_depth_files[x] for x in indices_list_perm]
        self.transform = transform
        self.scene_channels = scene_channels
    
    def __len__(self):
        return len(self.all_gt_scene_files)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.all_gt_scene_files[idx], cv2.IMREAD_UNCHANGED)
        image = image[:,:,::-1].astype(np.float32)/255.
        dmap = cv2.imread(self.all_gt_depth_files[idx], cv2.IMREAD_UNCHANGED)
        dmap = dmap.astype(np.float32)/65535.
        dmap = (dmap - dmap.min())/(dmap.max()-dmap.min())
        if self.scene_channels=='dualpix_mono': 
            image = image[:,:,1]
            sample = np.stack((dmap, image), axis=0)
        if self.scene_channels=='stdpix_green':
            image = image[:,:,1]
            sample = np.stack((dmap, image), axis=0)
        if self.scene_channels=='dualpix_rgb':
            sample = np.stack((dmap, image[:,:,0], image[:,:,0], image[:,:,1], image[:,:,1], image[:,:,2], image[:,:,2]))
        if self.scene_channels=='stdpix_rgb':
            sample = np.stack((dmap, image[:,:,0], image[:,:,1], image[:,:,2]))
        if self.transform:
            sample = self.transform(sample)
        return sample


class flyingthings3dDataset(Dataset):
    def __init__(self, dataroot_dir, train=False, train_val_split=False, N_scenes=0, transform=None, split_seed=0, scene_channels='dualpix_mono'):
        self.dataroot_dir = dataroot_dir
        if train:   # load training dataset files
            all_gt_scene_files = [join(dataroot_dir, 'train/image_clean/left', f) \
                                    for f in os.listdir(join(dataroot_dir, 'train/image_clean/left')) \
                                    if isfile(join(dataroot_dir, 'train/image_clean/left', f))]
            all_gt_depth_files = [join(dataroot_dir, 'train/disparity/left', f) \
                                    for f in os.listdir(join(dataroot_dir, 'train/disparity/left')) \
                                    if isfile(join(dataroot_dir, 'train/disparity/left', f))]
        else:       # load validation dataset files
            all_gt_scene_files = sorted([join(dataroot_dir, 'val/image_clean/left', f) \
                                    for f in os.listdir(join(dataroot_dir, 'val/image_clean/left')) \
                                    if isfile(join(dataroot_dir, 'val/image_clean/left', f))])
            all_gt_depth_files = sorted([join(dataroot_dir, 'val/disparity/left', f) \
                                    for f in os.listdir(join(dataroot_dir, 'val/disparity/left')) \
                                    if isfile(join(dataroot_dir, 'val/disparity/left', f))])
        
        if N_scenes: 
            indices_list = np.arange(N_scenes)
        else:
            N_scenes = len(all_gt_scene_files)
            indices_list = np.arange(N_scenes)
        old_seed = np.random.get_state()
        np.random.seed(split_seed)
        if train:
            indices_list_perm = np.random.permutation(indices_list)
        else: 
            indices_list_perm = indices_list
        np.random.set_state(old_seed)
        if train_val_split:
            raise NotImplementedError
        else:
            self.all_gt_scene_files = [all_gt_scene_files[x] for x in indices_list_perm]
            self.all_gt_depth_files = [all_gt_depth_files[x] for x in indices_list_perm]
        
        self.transform = transform
        self.scene_channels = scene_channels

    def __len__(self):
        return len(self.all_gt_scene_files)
    
    def __getitem__(self, idx):
        with PIL.Image.open(self.all_gt_scene_files[idx]) as f:
            image = np.array(f)
            if self.scene_channels=='dualpix_mono' or self.scene_channels=='stdpix_mono':
                image = image[:,:,1]
            image[image < 0] = 0
            image = image.astype(np.float32)/np.max(image)
        dmap = read_pfm(self.all_gt_depth_files[idx]) 
        # dmap[dmap < 0] = 0
        dmap = dmap - np.min(dmap)
        dmap = dmap.astype(np.float32)/np.max(dmap)
        if self.scene_channels=='dualpix_mono': 
            sample = np.stack((dmap, image), axis=0)
        if self.scene_channels=='stdpix_mono':
            sample = np.stack((dmap, image), axis=0)
        if self.scene_channels=='dualpix_rgb':
            sample = np.stack((dmap, image[:,:,0], image[:,:,0], image[:,:,1], image[:,:,1], image[:,:,2], image[:,:,2]))
        if self.scene_channels=='stdpix_rgb':
            sample = np.stack((dmap, image[:,:,0], image[:,:,1], image[:,:,2]))
        if self.transform:
            sample = self.transform(sample)
        return sample


class endoslamDataset(Dataset):
    def __init__(self, dataroot_dir, train=False, train_val_split=False, N_scenes=0, transform=None, split_seed=0, scene_channels='dualpix_mono'):
        self.dataroot_dir = dataroot_dir
        if train:   # load training dataset files
            all_gt_scene_files_intestine = sorted([join(dataroot_dir, 'small_intestine/Frames', f) \
                                for f in os.listdir(join(dataroot_dir, 'small_intestine/Frames')) \
                                if isfile(join(dataroot_dir, 'small_intestine/Frames', f))])[0:10000:1]#[:len(all_gt_depth_files_stomach)]
            all_gt_depth_files_intestine = sorted([join(dataroot_dir, 'small_intestine/Pixelwise_Depths', f) \
                                for f in os.listdir(join(dataroot_dir, 'small_intestine/Pixelwise_Depths')) \
                                if isfile(join(dataroot_dir, 'small_intestine/Pixelwise_Depths', f))])[0:10000:1]#[:len(all_gt_depth_files_stomach)]
            all_gt_scene_files_colon = [join(dataroot_dir, 'colon/Frames', f) \
                                for f in os.listdir(join(dataroot_dir, 'colon/Frames')) \
                                if isfile(join(dataroot_dir, 'colon/Frames', f))][0:10000:1]
            all_gt_depth_files_colon = [join(dataroot_dir, 'colon/Pixelwise_Depths', f) \
                                for f in os.listdir(join(dataroot_dir, 'colon/Pixelwise_Depths')) \
                                if isfile(join(dataroot_dir, 'colon/Pixelwise_Depths', f))][0:10000:1]
            all_gt_scene_files = np.concatenate((all_gt_scene_files_intestine, all_gt_scene_files_colon))
            all_gt_depth_files = np.concatenate((all_gt_depth_files_intestine, all_gt_depth_files_colon))
        else:       # load validation dataset files
            all_gt_scene_files = sorted([join(dataroot_dir, 'stomach/Frames', f) \
                                for f in os.listdir(join(dataroot_dir, 'stomach/Frames')) \
                                if isfile(join(dataroot_dir, 'stomach/Frames', f))])[0:1000]
            all_gt_depth_files = sorted([join(dataroot_dir, 'stomach/Pixelwise_Depths', f) \
                                for f in os.listdir(join(dataroot_dir, 'stomach/Pixelwise_Depths')) \
                                if isfile(join(dataroot_dir, 'stomach/Pixelwise_Depths', f))])[0:1000]
        
        if N_scenes: 
            indices_list = np.arange(N_scenes)
        else:
            N_scenes = len(all_gt_scene_files)
            indices_list = np.arange(N_scenes)
        old_seed = np.random.get_state()
        np.random.seed(split_seed)
        if train:
            indices_list_perm = np.random.permutation(indices_list)
        else: 
            indices_list_perm = indices_list
        np.random.set_state(old_seed)
        if train_val_split:
            raise NotImplementedError
        else:
            self.all_gt_scene_files = [all_gt_scene_files[x] for x in indices_list_perm]
            self.all_gt_depth_files = [all_gt_depth_files[x] for x in indices_list_perm]
        
        self.transform = transform
        self.scene_channels = scene_channels

        #taken from EndoSLAM paper
        self.K = np.array([[156.0418,0,178.5604],[0,155.7529,181.8043],[0,0,1]])
        self.dist = np.array([-0.2486,0.0614,0,0]) 

    def __len__(self):
        return len(self.all_gt_scene_files)
    
    def __getitem__(self, idx):
        image = skimage.io.imread(self.all_gt_scene_files[idx]) / 255.0
        
        image = cv2.undistort(image, self.K, self.dist, None)
        if self.scene_channels=='dualpix_mono' or self.scene_channels=='stdpix_mono':
            image = image[:,:,1]
        image[image < 0] = 0
        image = image.astype(np.float32)
        image = image / (image.max()+1e-7)

        dpth_filename = self.all_gt_scene_files[idx].replace('image_','aov_image_')
        if "intestine" in dpth_filename:
            dpth_filename = dpth_filename.replace('Frames','Pixelwise_Depths')
        else: 
            dpth_filename = dpth_filename.replace('Frames','Pixelwise_Depths')
        dmap = skimage.io.imread(dpth_filename) / 255.0
        dmap = dmap[:,:,0]
       
        dmap = dmap - np.min(dmap)
        dmap = dmap.astype(np.float32)/(np.max(dmap)+1e-6) 
        dmap = 1.0-dmap
        if self.scene_channels=='dualpix_mono': 
            sample = np.stack((dmap, image), axis=0)
        if self.scene_channels=='stdpix_mono':
            sample = np.stack((dmap, image), axis=0)
        if self.scene_channels=='dualpix_rgb':
            sample = np.stack((dmap, image[:,:,0], image[:,:,0], image[:,:,1], image[:,:,1], image[:,:,2], image[:,:,2]))
        if self.scene_channels=='stdpix_rgb':
            sample = np.stack((dmap, image[:,:,0], image[:,:,1], image[:,:,2]))
        if self.transform:
            sample = self.transform(sample)
        return sample

class TestCanonDPDBlurDataset(Dataset):
    def __init__(self, dataroot_dir, train, N_scenes, transform=None): 
        self.dataroot_dir = dataroot_dir
        self.len = N_scenes
        self.transform = transform
        self.train = train
        if self.train:
            self.all_dp_left_imgfiles = [join(dataroot_dir, 'canon/train_l/source/', f) \
                                for f in os.listdir(join(dataroot_dir, 'canon/train_l/source/')) \
                                if isfile(join(dataroot_dir, 'canon/train_l/source/', f))]
            self.all_dp_left_imgfiles.sort()
            self.all_dp_right_imgfiles = [f.replace('/train_l/','/train_r/').replace('_L.png','_R.png') for f in self.all_dp_left_imgfiles]
            self.all_gt_imgfiles = [join(dataroot_dir, 'canon/train_c/target/', f) \
                                for f in os.listdir(join(dataroot_dir, 'canon/train_c/target/')) \
                                if isfile(join(dataroot_dir, 'canon/train_c/target/', f))]
            self.all_gt_imgfiles.sort()
        else:
            self.all_dp_left_imgfiles = [join(dataroot_dir, 'canon/val_l/source/', f) \
                                for f in os.listdir(join(dataroot_dir, 'canon/val_l/source/')) \
                                if isfile(join(dataroot_dir, 'canon/val_l/source/', f))]
            self.all_dp_left_imgfiles.sort()
            self.all_dp_right_imgfiles = [f.replace('/val_l/','/val_r/').replace('_L.png','_R.png') for f in self.all_dp_left_imgfiles]
            self.all_gt_imgfiles = [join(dataroot_dir, 'canon/val_c/target/', f) \
                                for f in os.listdir(join(dataroot_dir, 'canon/val_c/target/')) \
                                if isfile(join(dataroot_dir, 'canon/val_c/target/', f))]
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
        # img_gt = np.transpose(img_gt, axes=(2,0,1))

        sample = np.stack((img_l[:,:,0], img_r[:,:,0], img_l[:,:,1], img_r[:,:,1], img_l[:,:,2], img_r[:,:,2], img_gt[:,:,0], img_gt[:,:,1], img_gt[:,:,2]), axis=0)
        
        if self.transform:
            sample = self.transform(sample)
        return sample

def prepare_dataset(args, training=True, train_val_split=False):
    if args.dataset_type=='dpdblur':
        if training: 
            tfm_lst = [to_tensor(), tfm.RandomCrop(384), tfm.RandomHorizontalFlip(), tfm.RandomVerticalFlip()]
            dataset = TestCanonDPDBlurDataset(args.dataroot, train=training, N_scenes=args.N_renders_override, transform=tfm.Compose(tfm_lst))
        else: 
            tfm_lst = [to_tensor()]
            dataset = TestCanonDPDBlurDataset(args.dataroot, train=training, N_scenes=74, transform=tfm.Compose(tfm_lst))
    elif args.dataset_type=='nyu2_test':
        tfm_lst = [RescaleDepthMap(args.in_focus_dist_mm, args.max_depth_mm), to_tensor(), tfm.CenterCrop((420,564))]
        dataset = nyuD2TestDataset(args.dataroot, train=False, N_scenes=args.N_renders_override, transform=tfm.Compose(tfm_lst), scene_channels=args.scene_channels)
    elif args.dataset_type=="flyingthings3d":
        if training:
            tfm_lst = [RescaleDepthMapRng(args.min_depth_mm, args.max_depth_mm), to_tensor(), tfm.RandomCrop(316), tfm.RandomHorizontalFlip(), tfm.RandomVerticalFlip()]
            if args.finetune:
                if args.scene_channels=="dualpix_rgb":
                    tfm_lst = [RescaleDepthMapRng(args.min_depth_mm, args.max_depth_mm), to_tensor(), RandomAugment_dualpix_rgb(), tfm.RandomCrop(316), tfm.RandomHorizontalFlip(), tfm.RandomVerticalFlip()]
                elif args.scene_channels=="stdpix_rgb":
                    tfm_lst = [RescaleDepthMapRng(args.min_depth_mm, args.max_depth_mm), to_tensor(), RandomAugment_stdpix_rgb(), tfm.RandomCrop(316), tfm.RandomHorizontalFlip(), tfm.RandomVerticalFlip()]
                else:
                    tfm_lst = [RescaleDepthMapRng(args.min_depth_mm, args.max_depth_mm), to_tensor(), RandomAugment(), tfm.RandomCrop(316), tfm.RandomHorizontalFlip(), tfm.RandomVerticalFlip()]
            dataset = flyingthings3dDataset(args.dataroot, train=training, 
                        N_scenes=args.N_renders_override, train_val_split=train_val_split, 
                        transform=tfm.Compose(tfm_lst),  
                        scene_channels=args.scene_channels)
            # finetuning - # contrast augmentation, brightness augmentation, noise level augmentation
        else:
            dataset = flyingthings3dDataset(args.dataroot, train=training, 
                    N_scenes=int(0.1*args.N_renders_override), train_val_split=train_val_split, 
                    transform=tfm.Compose([RescaleDepthMap(args.min_depth_mm, args.max_depth_mm), to_tensor(), tfm.CenterCrop((540, 956))]), 
                    scene_channels=args.scene_channels)

    elif args.dataset_type=="endoslam":
        if training:
            dataset = endoslamDataset(args.dataroot, train=training, 
                        N_scenes=args.N_renders_override, train_val_split=train_val_split, 
                        transform=tfm.Compose([RescaleDepthMap(args.min_depth_mm, args.max_depth_mm), to_tensor(), tfm.RandomCrop(276), tfm.RandomHorizontalFlip(), tfm.RandomVerticalFlip()]),
                        scene_channels=args.scene_channels)
        else:
            dataset = endoslamDataset(args.dataroot, train=training, 
                    N_scenes=int(0.05*args.N_renders_override), train_val_split=train_val_split, 
                    transform=tfm.Compose([RescaleDepthMap(args.min_depth_mm, args.max_depth_mm), to_tensor(), tfm.CenterCrop(316)]),
                    scene_channels=args.scene_channels)        
    else:
        raise NotImplementedError
    return dataset
