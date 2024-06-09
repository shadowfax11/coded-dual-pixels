import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, nb_ch, dropout_rate):
        super(UNet, self).__init__()

        # Downscaling layers
        self.down_conv1 = self.double_conv(3, 64)
        self.down_conv2 = self.double_conv(64, 128)
        self.down_conv3 = self.double_conv(128, 256)
        self.down_conv4 = self.double_conv(256, 512)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Bottleneck
        self.bottleneck_conv = self.double_conv(512, 1024)

        # Upscaling layers
        self.up_conv4 = self.double_conv(1024 + 512, 512)
        self.up_conv3 = self.double_conv(512 + 256, 256)
        self.up_conv2 = self.double_conv(256 + 128, 128)
        self.up_conv1 = self.double_conv(128 + 64, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, nb_ch, 1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downsampling path
        conv1 = self.down_conv1(x)
        x = F.max_pool2d(conv1, 2)
        conv2 = self.down_conv2(x)
        x = F.max_pool2d(conv2, 2)
        conv3 = self.down_conv3(x)
        x = F.max_pool2d(conv3, 2)
        conv4 = self.down_conv4(x)
        x = F.max_pool2d(conv4, 2)

        x = self.bottleneck_conv(x)
        x = self.dropout(x)

        # Upsampling path
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)

        x = self.final_conv(x)

        return x
