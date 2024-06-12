from .unet import UNet
from .dpdnet import DPDNet
from .dpdnet import DPDNetParallel
from .dpdnet import ParallelDPDNet
from .deconv import DeconvModel, WienerNet
from .resunet import ResUnet_VB
from .resunet_ppm import ResUnet_VB as PPMResUNet_VB

import torch.nn as nn
import torch 
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class IdentityModel(nn.Module):
    def __init__(self, args):
        super(IdentityModel, self).__init__()
    def forward(self, x):
        return x

def get_analytical_model(args, psf=None):
    if args.analytical_deconv_step=="None":
        model_analytical = IdentityModel(args)
    if args.analytical_deconv_step=="Deconv":
        model_analytical = DeconvModel(args)
    if args.analytical_deconv_step=="Wiener":
        model_analytical = WienerNet(args, psf)
    return model_analytical

def get_network_model(args):
    in_channels = 0
    if args.analytical_deconv_step=="None":
        in_channels = args.num_psf_channels
    if args.analytical_deconv_step=="Deconv":
        in_channels = args.num_depth_planes
    if args.analytical_deconv_step=="Wiener":
        in_channels = args.num_depth_planes*args.num_psf_channels
    if args.add_image_channels_after_analytic_step:
        in_channels += 0
    if hasattr(args, 'positional_encoding_maps'):
        if args.positional_encoding_maps is not None:
            in_channels += 2
    out_channels = 0

    if "depth" in args.model_output and "defocus" in args.model_output: 
        print("Please train either on defocus values OR on depth values, not both.")
        raise ValueError
    
    if "depth" in args.model_output:
        out_channels += 1
    if "defocus" in args.model_output:
        out_channels += 1
    if "aif" in args.model_output: 
        if args.scene_channels == "dualpix_rgb" or args.scene_channels == "stdpix_rgb": 
            out_channels += 3
        else:
            out_channels += 1

    if args.model_type=="unet":
        model_network = UNet(in_channels, out_channels, dim=args.channel_dim, norm=args.normalization, act_fn=args.final_layer_activation)
    if args.model_type=="dpdnet":
        model_network = DPDNet(in_channels, out_channels, dim=args.channel_dim, norm=args.normalization, act_fn_depth=args.final_layer_activation)
    if args.model_type=="dpdnet_parallel":
        out_channels_branch1 = 1 
        out_channels_branch2 = 3 if 'red+blue' in args.scene_channels else 1 
        print("DPDNet parallel decoders output channels {}, {}".format(out_channels_branch1, out_channels_branch2))
        model_network = DPDNetParallel(in_channels, out_channels_branch1, out_channels_branch2, dim=args.channel_dim, norm=args.normalization, act_fn_depth=args.final_layer_activation, act_fn_aif='relu')
    if args.model_type=="dpdnet2":
        model_network = ParallelDPDNet(in_channels, out_channels, dim=args.channel_dim, norm=args.normalization, act_fn_depth=args.final_layer_activation)
    if args.model_type=="resunet":
        model_network = ResUnet_VB(in_channels, dim=args.channel_dim, out_dim=out_channels)
    if args.model_type=="resunet_ppm":
        model_network = PPMResUNet_VB(in_channels, dim=args.channel_dim, out_dim=out_channels)
    # if args.model_type=="ddd_net":
    #     model_network = DDDNet(in_channels, dim=args.channel_dim, out_dim=out_channels)

    return model_network
