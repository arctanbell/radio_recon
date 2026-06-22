"""DiT-style image-to-image Transformer for radio reconstruction.

This is a lightweight, patch-based Transformer that maps condition images
(PSF + dirty + SD) -> reconstructed target.

Implementation notes:
- DiT-inspired AdaLN modulation (from a global pooled conditioning vector).
- Fixed 2D sin-cos positional embedding (no learned positions).
- Deterministic reconstruction network (no diffusion timesteps).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    """Create 1D sin-cos positional embedding.

    Args:
        embed_dim: output dimension (must be even)
        pos: [N] positions

    Returns:
        [N, embed_dim]
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sincos, got {embed_dim}")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=pos.dtype)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (max_period ** omega)
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def _build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Create fixed 2D sin-cos positional embedding.

    Returns:
        pos_embed: [1, grid_h*grid_w, embed_dim] float32 tensor on CPU.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sincos, got {embed_dim}")

    device = torch.device("cpu")
    dtype = torch.float32

    grid_y = torch.arange(grid_h, device=device, dtype=dtype)
    grid_x = torch.arange(grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    half = embed_dim // 2
    emb_y = _build_1d_sincos_pos_embed(half, yy)
    emb_x = _build_1d_sincos_pos_embed(half, xx)
    pos = torch.cat([emb_y, emb_x], dim=1)
    return pos.unsqueeze(0)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN-style modulation.

    x: [B, N, C]
    shift/scale: [B, C]
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero style conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

        # Produce shift/scale/gate for attention and MLP.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # Start near identity for stability.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(cond).chunk(6, dim=1)

        x_norm = _modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out

        x_norm = _modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(x_norm)
        x = x + gate2.unsqueeze(1) * mlp_out
        return x


@dataclass
class DiTRadioConfig:
    in_channels: int = 3
    out_channels: int = 1
    image_size: int = 192
    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class DiTRadio(nn.Module):
    """Patch-based Transformer reconstruction model."""

    def __init__(self, cfg: DiTRadioConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError(f"image_size ({cfg.image_size}) must be divisible by patch_size ({cfg.patch_size})")

        grid = cfg.image_size // cfg.patch_size
        self.grid_size = grid
        self.num_patches = grid * grid

        self.patch_embed = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )

        pos = _build_2d_sincos_pos_embed(cfg.embed_dim, grid, grid)
        self.register_buffer("pos_embed", pos, persistent=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm_final = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        # Conditioning vector from token average.
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim, eps=1e-6),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        self.final_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, 2 * cfg.embed_dim, bias=True),
        )
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)

        patch_out_dim = cfg.out_channels * cfg.patch_size * cfg.patch_size
        self.head = nn.Linear(cfg.embed_dim, patch_out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.cfg.image_size or w != self.cfg.image_size:
            raise ValueError(f"Expected input size {self.cfg.image_size}x{self.cfg.image_size}, got {h}x{w}")

        # Patchify
        x = self.patch_embed(x)  # [B, D, gh, gw]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        # Condition vector
        cond = x.mean(dim=1)
        cond = self.cond_mlp(cond)

        for blk in self.blocks:
            x = blk(x, cond)

        x = self.norm_final(x)
        shift, scale = self.final_mod(cond).chunk(2, dim=1)
        x = _modulate(x, shift, scale)

        # Predict patches and unpatchify
        x = self.head(x)  # [B, N, oc*p*p]

        gh = gw = self.grid_size
        p = self.cfg.patch_size
        oc = self.cfg.out_channels
        x = x.view(b, gh, gw, oc, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, oc, gh * p, gw * p)
        return x


def create_dit_radio(config: dict) -> DiTRadio:
    """Factory from project config dict."""
    mcfg = config.get("model", {})
    cfg = DiTRadioConfig(
        in_channels=int(mcfg.get("in_channels", 3)),
        out_channels=int(mcfg.get("out_channels", 1)),
        image_size=int(mcfg.get("image_size", 192)),
        patch_size=int(mcfg.get("patch_size", 8)),
        embed_dim=int(mcfg.get("embed_dim", 384)),
        depth=int(mcfg.get("depth", 8)),
        num_heads=int(mcfg.get("num_heads", 6)),
        mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
    return DiTRadio(cfg)
