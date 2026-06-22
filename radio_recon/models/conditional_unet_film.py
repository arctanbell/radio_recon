"""
Conditional U-Net with FiLM (Feature-wise Linear Modulation)

FiLM 机制:
- PSF 作为条件 (condition)，通过 FiLM 层调制 U-Net 的中间特征
- 核心思想: 让网络在不同尺度学习 PSF 的影响

公式: FiLM(x) = gamma * x + beta
其中 gamma, beta 由 PSF 的特征决定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer
    
    使用条件向量生成 gamma 和 beta，对特征进行调制
    """
    
    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.num_features = num_features
        
        # 从条件向量生成 gamma 和 beta
        self.fc = nn.Sequential(
            nn.Linear(condition_dim, num_features * 2),
        )
        
        # 初始化：gamma 接近 1，beta 接近 0（identity transform）
        nn.init.zeros_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 特征图
            condition: [B, condition_dim] 条件向量
        
        Returns:
            [B, C, H, W] 调制后的特征图
        """
        # 生成 gamma 和 beta
        params = self.fc(condition)  # [B, C*2]
        gamma = params[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = params[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        
        # FiLM: (1 + gamma) * x + beta
        # 使用 1 + gamma 让初始化时是 identity
        return (1 + gamma) * x + beta


class PSFEncoder(nn.Module):
    """
    PSF 编码器：从 PSF 图像提取条件向量
    """
    
    def __init__(self, in_channels: int = 1, condition_dim: int = 256):
        super().__init__()
        
        # 简单的 CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        
        # 投影到 condition_dim
        self.fc = nn.Sequential(
            nn.Linear(256, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
        )
    
    def forward(self, psf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psf: [B, 1, H, W] PSF 图像
        
        Returns:
            [B, condition_dim] 条件向量
        """
        x = self.encoder(psf)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        return self.fc(x)  # [B, condition_dim]


class DoubleConvFiLM(nn.Module):
    """(Conv -> BN -> ReLU -> FiLM) x 2"""
    
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.film1 = FiLMLayer(mid_channels, condition_dim)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film2 = FiLMLayer(out_channels, condition_dim)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.film1(x, condition)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.film2(x, condition)
        
        return x


class DownFiLM(nn.Module):
    """Downscaling with maxpool then double conv + FiLM"""
    
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConvFiLM(in_channels, out_channels, condition_dim)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        return self.conv(x, condition)


class UpFiLM(nn.Module):
    """Upscaling then double conv + FiLM"""
    
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvFiLM(in_channels, out_channels, condition_dim, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvFiLM(in_channels, out_channels, condition_dim)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, condition)


class ConditionalUNetFiLM(nn.Module):
    """
    Conditional U-Net with FiLM modulation
    
    架构:
    - PSF 通过单独的编码器提取条件向量
    - Dirty + SD 作为主要输入
    - FiLM 层在每个尺度调制特征
    
    Args:
        in_channels: 主输入通道数（默认2: Dirty + SD）
        out_channels: 输出通道数（默认1: Clean）
        features: 特征通道数列表
        condition_dim: 条件向量维度
        bilinear: 是否使用双线性上采样
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        condition_dim: int = 256,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # PSF 编码器
        self.psf_encoder = PSFEncoder(in_channels=1, condition_dim=condition_dim)
        
        # Encoder
        self.inc = DoubleConvFiLM(in_channels, features[0], condition_dim)
        self.down1 = DownFiLM(features[0], features[1], condition_dim)
        self.down2 = DownFiLM(features[1], features[2], condition_dim)
        self.down3 = DownFiLM(features[2], features[3], condition_dim)
        
        factor = 2 if bilinear else 1
        self.down4 = DownFiLM(features[3], features[3] * 2 // factor, condition_dim)
        
        # Decoder
        self.up1 = UpFiLM(features[3] * 2, features[3] // factor, condition_dim, bilinear)
        self.up2 = UpFiLM(features[3], features[2] // factor, condition_dim, bilinear)
        self.up3 = UpFiLM(features[2], features[1] // factor, condition_dim, bilinear)
        self.up4 = UpFiLM(features[1], features[0], condition_dim, bilinear)
        
        # Output
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, 3, H, W] - condition (PSF + Dirty + SD)
               PSF 在通道 0，Dirty 在通道 1，SD 在通道 2
        
        Returns:
            [B, 1, H, W] - predicted clean image
        """
        # 分离 PSF 和主输入
        psf = x[:, 0:1, :, :]  # [B, 1, H, W]
        main_input = x[:, 1:, :, :]  # [B, 2, H, W] (Dirty + SD)
        
        # 编码 PSF 得到条件向量
        condition = self.psf_encoder(psf)  # [B, condition_dim]
        
        # Encoder with FiLM
        x1 = self.inc(main_input, condition)
        x2 = self.down1(x1, condition)
        x3 = self.down2(x2, condition)
        x4 = self.down3(x3, condition)
        x5 = self.down4(x4, condition)
        
        # Decoder with FiLM
        x = self.up1(x5, x4, condition)
        x = self.up2(x, x3, condition)
        x = self.up3(x, x2, condition)
        x = self.up4(x, x1, condition)
        
        # Output
        out = self.outc(x)
        
        return out


def create_conditional_unet_film(config: dict) -> ConditionalUNetFiLM:
    """从配置创建 Conditional U-Net FiLM"""
    model_config = config.get('model', {})
    
    return ConditionalUNetFiLM(
        in_channels=model_config.get('in_channels', 2),  # Dirty + SD
        out_channels=model_config.get('out_channels', 1),
        features=model_config.get('features', [64, 128, 256, 512]),
        condition_dim=model_config.get('condition_dim', 256),
        bilinear=model_config.get('bilinear', True),
    )


# 测试代码
if __name__ == '__main__':
    # 创建模型
    model = ConditionalUNetFiLM(
        in_channels=2,
        out_channels=1,
        features=[64, 128, 256, 512],
        condition_dim=256,
    )
    
    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)  # [B, 3, H, W] - PSF + Dirty + SD
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
