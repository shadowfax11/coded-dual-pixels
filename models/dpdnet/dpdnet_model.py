""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch

from .dpdnet_parts import *


class DPDNetParallel(nn.Module):
    def __init__(self, n_channels, n_classes1, n_classes2, dim=64, bilinear=True, kernel_size= 3, norm='bn', act_fn_depth='none', act_fn_aif='relu', dropout_rate=0.4):
        super(DPDNetParallel, self).__init__()
        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        self.n_classes2 = n_classes2
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, dim)
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        self.up1_depth = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2_depth = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3_depth = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4_depth = Up(dim * 2, dim, bilinear)
        self.outc_depth = OutConv(dim, n_classes1, norm=norm, activation=act_fn_depth)
        self.up1_aif = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2_aif = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3_aif = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4_aif = Up(dim * 2, dim, bilinear)
        self.outc_aif = OutConv(dim, n_classes2, norm=norm, activation=act_fn_aif)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x_d = self.up1_depth(x5, x4)
        x_d = self.up2_depth(x_d, x3)
        x_d = self.up3_depth(x_d, x2)
        x_d = self.up4_depth(x_d, x1)
        logits_d = self.outc_depth(x_d)
        x_a = self.up1_aif(x5, x4)
        x_a = self.up2_aif(x_a, x3)
        x_a = self.up3_aif(x_a, x2)
        x_a = self.up4_aif(x_a, x1)
        logits_a = self.outc_aif(x_a)
        return torch.cat((logits_d, logits_a), dim = 1)

class ParallelDPDNet(nn.Module): 
    def __init__(self, n_channels, n_classes, dim=64, bilinear=True, kernel_size=3, norm='bn', act_fn='none'):
        super(ParallelDPDNet, self).__init__()
        self.dpdnet_depth = DPDNet(n_channels, n_classes, dim, bilinear=True, kernel_size=kernel_size, norm=norm, act_fn=act_fn)
        self.dpdnet_aif = DPDNet(n_channels, n_classes, dim, bilinear=True, kernel_size=kernel_size, norm=norm, act_fn='relu')

    def forward(self, x): 
        logits_d = self.dpdnet_depth(x)
        logits_a = self.dpdnet_aif(x)
        return torch.cat((logits_d, logits_a), dim=1)

class DPDNet(nn.Module):
    def __init__(self, n_channels, n_classes, dim=64, bilinear=True, kernel_size= 3, norm='bn', act_fn='relu', dropout_rate=0.4):
        super(DPDNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, dim)
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes, norm=norm, activation=act_fn)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits