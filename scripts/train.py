"""
Code for training models. 
Code template inspired and taken from Sfp-in-the-wild github code release. 
Author: Bhargav Ghanekar, Pranav Sharma, Salman Siddique Khan
"""
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
import sys; sys.path.append('.') 
import numpy as np 
import math 
import random 
import time 
import argparse
from scipy.io import loadmat 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader 

from configs.config import config_parser
import models as Models 
from src.dataloader import prepare_dataset
from src.learned_mask import LearnedCode
from src.render import Convolve3DFFT, defocus_to_depth
from src.engine import train_one_epoch, validate
from utils.log_utils import init_logging
import utils.train_utils as train_utils

def main(): 
    # create argument parser
    parser = argparse.ArgumentParser()
    parser = config_parser(parser)
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    args.log_dir = os.path.join(args.expt_savedir, 'logs', args.expt_name)
    args.output_dir = os.path.join(args.expt_savedir, 'results', args.expt_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'ckpt_epochs'), exist_ok=True)
    writer, logger = init_logging(args)
    logger.info('Code files copied to {}'.format(os.path.join(args.log_dir, 'code_copy')))
    def print_info(): 
        logger.info('Training CodedDPNet model') 
        logger.info('Arguments: ') 
        for k, v in vars(args).items(): 
            logger.info('\t{}: {}'.format(k, v))
    print_info()

    # set seed
    seed = args.seed
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info('Random seed set to {}'.format(seed))

    # CUDA setup 
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    logger.info('Device: {}'.format(device))
    dtype = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor

    # load and prepare PSF data
    args.aperture_mm = args.focal_length_mm / args.f_number 
    if args.negative_defocus_only: 
        defocus_vals = np.linspace(-1, 0, num=args.num_depth_planes) 
    elif args.positive_defocus_only: 
        defocus_vals = np.linspace(0, +1, num=args.num_depth_planes)
    else: 
        defocus_vals = np.linspace(-1, +1, num=args.num_depth_planes)
    defocus_vals = defocus_vals * args.max_defocus_blur_size_px * args.pixel_pitch_um / 1000
    logger.info("Max. defocus blur: {}px".format(args.max_defocus_blur_size_px))
    args.z_vals_mm = defocus_to_depth(defocus_vals, args.aperture_mm, args.focal_length_mm, args.in_focus_dist_mm)
    args.min_depth_mm = args.z_vals_mm[0]
    args.max_depth_mm = args.z_vals_mm[-1]
    logger.info("Min/Max imaging depth: {}/{} mm\n".format(args.min_depth_mm, args.max_depth_mm))
    logger.info("Using PSFs at depths " + ''.join(['{:.4f}'.format(elem) for elem in args.z_vals_mm]))

    # load PSF data
    if args.finetune: 
        logger.info("PSF has been provided! This will override any code/mask learning functionality.")
    psf_data = loadmat(os.path.join(args.assets_dir, args.psf_data_file)) 
    psfs_meas = psf_data['PSFstack'].astype(np.float32)         # K x Nz x Nh x Nw
    args.num_psf_channels = psfs_meas.shape[0]
    _, Nz, Nh, Nw = psfs_meas.shape
    psfs_meas = torch.nn.Parameter(torch.tensor(psfs_meas).unsqueeze(0), requires_grad=False)
    if cuda_available: 
        psfs_meas = psfs_meas.to(device)
    assert Nz == args.num_depth_planes, "Number of depth planes in PSF data does not match the number of depth planes in the scene."

    # prepare learned code module 
    blur_sizes_data = './assets/circle_of_confusion_v1.npy' 
    coded_mask = LearnedCode(blur_sizes_data, alpha_init=args.initial_mask_alpha, 
                            init_mask_fpath=args.initial_mask_fpath, keep_mask_fixed=args.freeze_mask)
    
    # prepare 3D convolutional model (for rendering images of 3D scenes) 
    model_optics = Convolve3DFFT(args) 

    # get analytical model 
    model_deconv = Models.get_analytical_model(args, psfs_meas)

    # get reconstruction neural network model
    model = Models.get_network_model(args) 
    if cuda_available: 
        coded_mask.cuda()
        model_optics.cuda()
        model_deconv.cuda()
        model.cuda()
    
    # create dataloaders
    dataset_train = prepare_dataset(args, training=True, train_val_split=False)
    dataset_valid = prepare_dataset(args, training=False, train_val_split=False) 
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, 
                                drop_last=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,
                                drop_last=False, num_workers=args.num_workers)
    logger.info('Training dataset size: {}'.format(len(dataset_train)))
    logger.info('Validation dataset size: {}'.format(len(dataset_valid)))

    # create optimizer
    if not args.freeze_mask: 
        optimizer_mask = optim.Adam(coded_mask.parameters(), lr=args.lr_mask)
        logger.info('Mask learning rate: {:.6f}'.format(args.lr_mask))
    else: 
        optimizer_mask = None
    optimizer = optim.Adam(model.parameters(), lr=args.lr_model) 
    logger.info('Model learning rate: {:.6f}'.format(args.lr_model))

    # set up learning rate scheduler
    if not args.freeze_mask: 
        scheduler_mask = optim.lr_scheduler.CosineAnnealingLR(optimizer_mask, T_max=args.num_epochs_for_mask_learning, eta_min=1e-7)
        logger.info('Using cosine decay scheduler for mask learning over {} epochs'.format(args.num_epochs_for_mask_learning))
    if args.scheduler is not None: 
        if args.scheduler == 'cosine': 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)
            logger.info('Using cosine decay scheduler for model learning over {} epochs'.format(args.num_epochs))
        elif args.scheduler == 'cosine-annealing-warm-restart':
            raise NotImplementedError
        else:
            raise ValueError('Invalid scheduler: {}'.format(args.scheduler))
    else:
        scheduler = None
    
    # load pre-training weights 
    loadpath = args.load_wts_file 
    if loadpath is None: 
        start_epoch = 0 
        logger.info('No epoch loaded, starting at {} epoch(s)'.format(start_epoch))
    else: 
        start_epoch = train_utils.load_model_checkpoint(model, loadpath)
        if not args.resume_training: 
            start_epoch = 0 
        if scheduler is not None: 
            scheduler.step(start_epoch)
        if start_epoch<args.num_epochs_for_mask_learning and not args.freeze_mask:
            scheduler_mask.step(start_epoch)
        if start_epoch==args.num_epochs_for_mask_learning and not args.freeze_mask: 
            args.freeze_mask = True
            for param_group in optimizer_mask.param_groups:
                param_group['lr'] = 0.0
            logger.info('Now setting mask LR to zero')
        logger.info('Model loaded; starting training at {} epoch(s)'.format(start_epoch))

    if args.training_mode == 'validation': 
        logger.info('Validation mode selected; will not train. Will simply predict on the validation set')
        test_loss_avg, metric_mean_val = validate(dataloader_valid, None, 
                                                psfs_meas, coded_mask, model_optics, 
                                                model_deconv, model, writer, args, logger)
        exit() 
    
    # start training loop
    metric_best_val = 1e9 
    for epoch in range(start_epoch, args.num_epochs): 
        # train for 1 epoch 
        coded_mask, model_deconv, model = train_one_epoch(dataloader_train, epoch, 
                                                        psfs_meas, coded_mask, model_optics, 
                                                        model_deconv, model, optimizer_mask, 
                                                        optimizer, writer, args, logger)
        
        # update LRs through schedulers if any 
        if scheduler is not None: 
            scheduler.step(epoch) 
        if epoch+1<args.num_epochs_for_mask_learning and not args.freeze_mask: 
            scheduler_mask.step(epoch) 
        if epoch+1==args.num_epochs_for_mask_learning and not args.freeze_mask: 
            args.freeze_mask = True 
            for param_group in optimizer_mask.param_groups: 
                param_group['lr'] = 0.0 
            logger.info('Now setting mask LR to zero')
        
        # save model checkpoints, and run validation pass 
        if (epoch+1)%args.save_freq == 0 or epoch == 0: 
            train_utils.save_model_checkpoint(model, epoch, 
                        save_path=os.path.join(args.output_dir,'ckpt_epochs/ckpt_e{}.pth'.format(epoch)))
            train_utils.save_model_checkpoint(model, epoch, 
                        save_path=os.path.join(args.output_dir,'ckpt.pth'))
            
            _, metric_mean_val = validate(dataloader_valid, epoch, psfs_meas, coded_mask, model_optics, 
                                        model_deconv, model, writer, args, logger)
            
            if metric_mean_val < metric_best_val: 
                metric_best_val = metric_mean_val 
                train_utils.save_model_checkpoint(model, epoch, 
                            save_path=os.path.join(args.output_dir,'ckpt_best_val.pth'))
    writer.close()

if __name__ == '__main__':
    main()

