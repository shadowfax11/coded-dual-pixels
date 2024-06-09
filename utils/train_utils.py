import math 
import torch 
import os 
import logging
from collections import OrderedDict
from src.render import depth_to_defocus

def get_error_maps(pred_dict, gt_dict):
    err_dict = {}
    if 'depth' in pred_dict.keys():
        err_dict['depth'] = torch.sqrt(torch.abs((pred_dict['depth'] - gt_dict['depth'])**2))
    if 'aif' in pred_dict.keys():
        err_dict['aif'] = torch.sqrt(torch.abs((pred_dict['aif'] - gt_dict['aif'])**2))
    return err_dict

def get_in_maps(sample_batch, args):
    scene_gt = sample_batch[:,1:,:,:]
    depth_gt = sample_batch[:,0,:,:]
    defocus_gt = depth_to_defocus(depth_gt, args.aperture_mm, args.focal_length_mm, args.in_focus_dist_mm)
    defocus_gt = defocus_gt.unsqueeze(1)
    return scene_gt, depth_gt, defocus_gt 

def get_out_maps(result, model_output, scene_channels, min_depth_mm=0.0, max_depth_mm=1.0, max_defocus_size_mm=1.0, out_dict=None):
    if out_dict is None:
        out_dict = {}
    if "defocus" in model_output: 
        out_dict['defocus'] = result[:,[0]]*(max_defocus_size_mm)
    if "depth" in model_output:
        out_dict['depth'] = min_depth_mm + (result[:,[0]]*(max_depth_mm - min_depth_mm))
    if "aif" in model_output:
        if scene_channels=='dualpix_rgb' or scene_channels=='stdpix_rgb':
            out_dict['aif'] = torch.clip(result[:,-3:], min=0, max=1)
        else: 
            out_dict['aif'] = torch.clip(result[:,[-1]], min=0, max=1)
    return out_dict

def save_model_checkpoint(model, epoch, save_path=None): 
    """
    Saves a checkpoint under 'args['log_dir'] + '/ckpt.pth'
    """
    if hasattr(model, "module"): 
        model = model.module
    
    save_dict = { 
        'state_dict': model.state_dict(), 
        'epoch': epoch, 
    }
    torch.save(save_dict, save_path)
    del save_dict
    return


def load_model_checkpoint(model, loadpath, epoch=-1): 
    """
    Loads a checkpoint. Either for resuming training or for loading a learned model
    """
    logger = logging.getLogger(__name__)
    if os.path.isfile(loadpath): 
        loc = 'cuda' 
        checkpoint = torch.load(loadpath, map_location=loc) 
        state_dict = OrderedDict() 
        for k, v in checkpoint['state_dict'].items(): 
            # state_dict["module." + k] = v 
            state_dict[k] = v
        logger.info("> Loading weights from {}".format(loadpath))
        model.load_state_dict(state_dict)
        if epoch<0:
            start_epoch = checkpoint['epoch']+1
            logger.info(">> Resuming training at {} epoch".format(start_epoch))
        else: 
            start_epoch = epoch
    else:
        logger.info("No checkpoint at {}".format(loadpath))
        raise KeyError
    return start_epoch
