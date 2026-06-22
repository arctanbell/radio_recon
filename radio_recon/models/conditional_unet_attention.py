"""
Conditional U-Net with Cross-Attention

Cross-Attention 机制:
- PSF 特征作为 Key 和 Value
- 主输入特征作为 Query
- 让模型"注意"PSF 的关键频率信息

相比 FiLM:
- 更灵活：可以学习空间变化的调制
- 更强大：可以建模复杂的条件依赖
- 计算量稍大，但效果可能更好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention Layer
    
    Q 来自主输入，K/V 来自条件
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q 来自主输入
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        # K, V 来自条件
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor,      # [B, N, C]
        key_value: torch.Tensor,  # [B, M, C]
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N, C] 主输入特征（展平后的）
            key_value: [B, M, C] 条件特征
        
        Returns:
            [B, N, C] 注意力加权后的特征
        """
        B, N, C = query.shape
        M = key_value.shape[1]
        
        # 投影
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.k_proj(key_value).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, M, D]
        v = self.v_proj(key_value).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, M, D]
        
        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权聚合
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.out_proj(out)
        
        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block for 2D feature maps
    
    将 2D 特征图展平，应用 cross-attention，再恢复为 2D
    支持不同通道数的 query 和 key/value
    """
    
    def __init__(
        self,
        query_channels: int,
        kv_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # 投影到相同维度
        self.embed_dim = query_channels
        self.q_proj = nn.Linear(query_channels, query_channels)
        self.k_proj = nn.Linear(kv_channels, query_channels)
        self.v_proj = nn.Linear(kv_channels, query_channels)
        self.out_proj = nn.Linear(query_channels, query_channels)
        
        self.num_heads = num_heads
        self.head_dim = query_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(query_channels)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(query_channels, query_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_channels * 4, query_channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(query_channels)
    
    def forward(
        self, 
        x: torch.Tensor,        # [B, C, H, W]
        condition: torch.Tensor  # [B, C', H', W']
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 主输入特征图
            condition: [B, C', H', W'] 条件特征图（PSF 编码）
        
        Returns:
            [B, C, H, W] 注意力调制后的特征图
        """
        B, C, H, W = x.shape
        C_cond = condition.shape[1]
        
        # 展平
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        cond_flat = condition.flatten(2).transpose(1, 2)  # [B, H'*W', C']
        
        # Cross-Attention
        q = self.q_proj(self.norm1(x_flat))  # [B, N, C]
        k = self.k_proj(cond_flat)  # [B, M, C]
        v = self.v_proj(cond_flat)  # [B, M, C]
        
        # Reshape for multi-head attention
        N, M = q.shape[1], k.shape[1]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        # Residual + FFN
        x_flat = x_flat + out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        
        # 恢复形状
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class PSFEncoderMultiScale(nn.Module):
    """
    多尺度 PSF 编码器
    
    输出多个尺度的特征图，用于不同层的 cross-attention
    """
    
    def __init__(self, in_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        
        self.features = features
        
        # 每个尺度的编码器
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )
        
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(features[0], features[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(features[1], features[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[2], features[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(features[2], features[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[3], features[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, psf: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            psf: [B, 1, H, W] PSF 图像
        
        Returns:
            List of [B, C_i, H_i, W_i] 多尺度特征
        """
        f0 = self.enc0(psf)      # [B, 64, H, W]
        f1 = self.enc1(f0)       # [B, 128, H/2, W/2]
        f2 = self.enc2(f1)       # [B, 256, H/4, W/4]
        f3 = self.enc3(f2)       # [B, 512, H/8, W/8]
        
        return [f0, f1, f2, f3]


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownWithAttention(nn.Module):
    """Downscaling with maxpool, double conv, and cross-attention"""
    
    def __init__(self, in_channels: int, out_channels: int, kv_channels: int, num_heads: int = 8):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.cross_attn = CrossAttentionBlock(out_channels, kv_channels, num_heads)
    
    def forward(self, x: torch.Tensor, psf_feat: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.cross_attn(x, psf_feat)
        return x


class UpWithAttention(nn.Module):
    """Upscaling with double conv and cross-attention"""
    
    def __init__(self, in_channels: int, out_channels: int, kv_channels: int, num_heads: int = 8, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        self.cross_attn = CrossAttentionBlock(out_channels, kv_channels, num_heads)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, psf_feat: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.cross_attn(x, psf_feat)
        return x


class ConditionalUNetAttention(nn.Module):
    """
    Conditional U-Net with Cross-Attention
    
    架构:
    - PSF 通过多尺度编码器提取特征
    - 在每个尺度，主输入通过 cross-attention "关注" PSF 特征
    - 这让模型能够空间自适应地利用 PSF 信息
    
    Args:
        in_channels: 主输入通道数（默认2: Dirty + SD）
        out_channels: 输出通道数（默认1: Clean）
        features: 特征通道数列表
        num_heads: 注意力头数
        bilinear: 是否使用双线性上采样
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        num_heads: int = 8,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # PSF 多尺度编码器
        self.psf_encoder = PSFEncoderMultiScale(in_channels=1, features=features)
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.inc_attn = CrossAttentionBlock(features[0], features[0], num_heads)
        
        self.down1 = DownWithAttention(features[0], features[1], features[1], num_heads)
        self.down2 = DownWithAttention(features[1], features[2], features[2], num_heads)
        self.down3 = DownWithAttention(features[2], features[3], features[3], num_heads)
        
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features[3], features[3] * 2 // factor),
        )
        
        # Decoder
        self.up1 = UpWithAttention(features[3] * 2, features[3] // factor, features[3], num_heads, bilinear)
        self.up2 = UpWithAttention(features[3], features[2] // factor, features[2], num_heads, bilinear)
        self.up3 = UpWithAttention(features[2], features[1] // factor, features[1], num_heads, bilinear)
        self.up4 = UpWithAttention(features[1], features[0], features[0], num_heads, bilinear)
        
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
        psf = x[:, 0:1, :, :]      # [B, 1, H, W]
        main_input = x[:, 1:, :, :]  # [B, 2, H, W] (Dirty + SD)
        
        # 编码 PSF，得到多尺度特征
        psf_feats = self.psf_encoder(psf)  # List of 4 tensors
        
        # Encoder with cross-attention
        x1 = self.inc(main_input)
        x1 = self.inc_attn(x1, psf_feats[0])
        
        x2 = self.down1(x1, psf_feats[1])
        x3 = self.down2(x2, psf_feats[2])
        x4 = self.down3(x3, psf_feats[3])
        x5 = self.down4(x4)
        
        # Decoder with cross-attention
        x = self.up1(x5, x4, psf_feats[3])
        x = self.up2(x, x3, psf_feats[2])
        x = self.up3(x, x2, psf_feats[1])
        x = self.up4(x, x1, psf_feats[0])
        
        # Output
        out = self.outc(x)
        
        return out


def create_conditional_unet_attention(config: dict) -> ConditionalUNetAttention:
    """从配置创建 Conditional U-Net Attention"""
    model_config = config.get('model', {})
    
    return ConditionalUNetAttention(
        in_channels=model_config.get('in_channels', 2),
        out_channels=model_config.get('out_channels', 1),
        features=model_config.get('features', [64, 128, 256, 512]),
        num_heads=model_config.get('num_heads', 8),
        bilinear=model_config.get('bilinear', True),
    )


# 测试代码
if __name__ == '__main__':
    # 创建模型
    model = ConditionalUNetAttention(
        in_channels=2,
        out_channels=1,
        features=[64, 128, 256, 512],
        num_heads=8,
    )
    
    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)  # [B, 3, H, W] - PSF + Dirty + SD
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
