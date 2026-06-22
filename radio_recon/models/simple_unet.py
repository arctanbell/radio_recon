"""
Simple U-Net 基线模型

不使用Diffusion，直接监督学习：
输入：[PSF, Dirty, SD] 拼接 → [3, H, W]
输出：[Clean] → [1, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class BottleneckStack(nn.Module):
    """Extra bottleneck processing without changing feature map size."""

    def __init__(self, channels, depth):
        super().__init__()
        layers = [DoubleConv(channels, channels) for _ in range(max(0, depth))]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if len(self.layers) == 0:
            return x
        return self.layers(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size (if needed)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    Simple U-Net for radio reconstruction
    
    Args:
        in_channels: 输入通道数（默认3: PSF + Dirty + SD）
        out_channels: 输出通道数（默认1: Clean）
        features: 特征通道数列表（默认[64, 128, 256, 512]）
        bilinear: 是否使用双线性上采样
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
        bilinear=True,
        residual_output=False,
        residual_source_channel=1,
        bottleneck_depth=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.residual_output = residual_output
        self.residual_source_channel = residual_source_channel
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        bottleneck_channels = features[3] * 2 // factor
        self.bottleneck = BottleneckStack(bottleneck_channels, bottleneck_depth)
        
        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 3, H, W] - condition (PSF + Dirty + SD)
        
        Returns:
            [B, 1, H, W] - predicted clean image
        """
        input_condition = x

        # Encoder
        x1 = self.inc(input_condition)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        out = self.outc(x)

        if self.residual_output:
            if not 0 <= self.residual_source_channel < self.in_channels:
                raise ValueError(
                    f"residual_source_channel={self.residual_source_channel} is out of range for "
                    f"in_channels={self.in_channels}"
                )
            residual_base = input_condition[:, self.residual_source_channel:self.residual_source_channel + 1]
            out = out + residual_base

        return out


def create_simple_unet(config: dict) -> SimpleUNet:
    """从配置创建Simple U-Net"""
    model_config = config.get('model', {})
    
    return SimpleUNet(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 1),
        features=model_config.get('features', [64, 128, 256, 512]),
        bilinear=model_config.get('bilinear', True),
        residual_output=model_config.get('residual_output', False),
        residual_source_channel=model_config.get('residual_source_channel', 1),
        bottleneck_depth=model_config.get('bottleneck_depth', 0),
    )
