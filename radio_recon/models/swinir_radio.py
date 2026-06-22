from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)

    def _build_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        attn_mask = self._build_mask(hp, wp, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_h or pad_w:
            x = x[:, :h, :w, :]

        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + x

        mlp_in = x.permute(0, 2, 3, 1).contiguous()
        mlp_out = self.mlp(self.norm2(mlp_in))
        mlp_out = mlp_out.permute(0, 3, 1, 2).contiguous()
        return x + mlp_out


class ResidualSwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.blocks(x)
        x = self.conv(x)
        return x + residual


class RadioSwinIR(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        embed_dim: int = 96,
        depths: list[int] | tuple[int, ...] = (6, 6, 6, 6),
        num_heads: list[int] | tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError('depths and num_heads must have the same length')
        if embed_dim % num_heads[0] != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size

        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        self.stages = nn.ModuleList(
            [
                ResidualSwinStage(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for depth, heads in zip(depths, num_heads)
            ]
        )
        self.norm = nn.BatchNorm2d(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_first(x)
        residual = x
        for stage in self.stages:
            x = stage(x)
        x = self.conv_after_body(self.norm(x)) + residual
        return self.conv_last(x)


def create_swinir_radio(config: dict) -> RadioSwinIR:
    model_config = config.get('model', {})
    return RadioSwinIR(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 1),
        embed_dim=model_config.get('embed_dim', 96),
        depths=model_config.get('depths', [6, 6, 6, 6]),
        num_heads=model_config.get('num_heads', [6, 6, 6, 6]),
        window_size=model_config.get('window_size', 8),
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
    )
