#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network.py

U-Net 2D (H=control-points, W=leaf/detector) + conditionnement FiLM.

Entrée:
  x_drr:     [B, Cin, Hpx, W_in]   avec Hpx = Hcp * cp_height
  angles:    [B, L, 1]
  positions: [B, L, 1]
  film_extra:[B, D] (optionnel)
  present_flags: [B, Cin] (optionnel) si use_missing_token=True

Sortie:
  y:         [B, Cout, Hcp, W_out]

Changement majeur (par rapport à ton ViTRepack):
  - Plus de ViT.
  - Repack EXACT (sans perte) Hpx -> Hcp via "unshuffle" sur H:
        [B,C,Hcp*cp_height,W] -> [B,C*cp_height,Hcp,W]
    puis projection CNN vers base_ch.
  => On passe de 6144x64 à 512x64 sans pooling/stride destructif.

Autre changement fortement recommandé:
  - Up: interpolation bilinéaire + Conv2d (évite artefacts "checkerboard" des ConvTranspose).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.checkpoint import checkpoint
except Exception:  # pragma: no cover
    checkpoint = None


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _safe_gn_groups(ch: int, requested: int) -> int:
    """Retourne un nb de groupes GN compatible avec ch."""
    g = min(int(requested), int(ch))
    while g > 1 and (ch % g) != 0:
        g -= 1
    return max(g, 1)


def _resample_len(x: Optional[torch.Tensor], target_len: int) -> Optional[torch.Tensor]:
    """Resample [B,L,D] -> [B,target_len,D] par interpolation linéaire sur L."""
    if x is None:
        return None
    B, L, D = x.shape
    if L == target_len:
        return x
    x_t = x.permute(0, 2, 1)  # [B,D,L]
    x_t = F.interpolate(x_t, size=int(target_len), mode="linear", align_corners=False)
    return x_t.permute(0, 2, 1).contiguous()


# -------------------------------------------------
# Conditioning
# -------------------------------------------------

class SineCosineEncoding(nn.Module):
    """Embedding sin/cos 1D + MLP pour angles normalisés."""

    def __init__(self, out_dim: int, hidden: int = 256):
        super().__init__()
        self.out_dim = int(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,1] normalisé [0..1] ou [-0.5..0.5], peu importe tant que cohérent
        s = torch.sin(2 * math.pi * x)
        c = torch.cos(2 * math.pi * x)
        z = torch.cat([s, c], dim=-1)  # [B,L,2]
        return self.mlp(z)


class FiLMMap(nn.Module):
    """cond [B,H,cond_dim] -> gamma/beta [B,H,2*C]."""

    def __init__(self, cond_dim: int, channels: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


def _apply_film(x: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
    """Apply FiLM.

    x: [B,C,H,W]
    gamma_beta: [B,H,2*C]
    """
    B, C, H, _ = x.shape
    gb = gamma_beta.view(B, H, 2, C).permute(0, 2, 3, 1).contiguous()  # [B,2,C,H]
    gamma = gb[:, 0].unsqueeze(-1)  # [B,C,H,1]
    beta = gb[:, 1].unsqueeze(-1)   # [B,C,H,1]
    return x * (1.0 + gamma) + beta


class FilmEmbed(nn.Module):
    """Petit embedder interne pour film_extra (D_in -> D_out)."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 64):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.net = nn.Sequential(
            nn.Linear(self.d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------
# Missing channel token (optionnel)
# -------------------------------------------------

class MissingChannelToken(nn.Module):
    """Remplace des canaux d'entrée manquants par un token appris.

    present_flags: [B, Cin] (1 présent / 0 manquant)
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        nn.init.normal_(self.token, std=0.02)

    def forward(self, x: torch.Tensor, present_flags: Optional[torch.Tensor]) -> torch.Tensor:
        if present_flags is None:
            return x
        m = present_flags.to(dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        return x * m + self.token * (1.0 - m)


# -------------------------------------------------
# Stem: CP-aware multi-branches
# -------------------------------------------------


class BranchConv(nn.Module):
    """Petit bloc conv pour une branche après repack."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        anisotropic_leafwise: bool = False,
    ):
        super().__init__()
        g = _safe_gn_groups(out_ch, num_groups)
        k2 = (3, 1) if bool(anisotropic_leafwise) else (3, 3)
        p2 = (1, 0) if bool(anisotropic_leafwise) else (1, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Dropout2d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=k2, padding=p2, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CPStemMultiBranch(nn.Module):
    """
    Repack exact Hpx=Hcp*cp_height -> Hcp en mettant cp_height dans les canaux, puis
    traitement par branches (PTV / External / Shoulder / OAR-others) avant fusion.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cp_height: int,
        w_in: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        pre_k: int = 3,
        film_like_init: float = 0.0,
        shoulder_gate: bool = True,
        shoulder_oar_index_in_oars: int = 9,
        anisotropic_leafwise: bool = False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.cp_height = int(cp_height)
        self.w_in = int(w_in)
        self.shoulder_gate = bool(shoulder_gate)
        self.shoulder_oar_index_in_oars = int(shoulder_oar_index_in_oars)
        self.anisotropic_leafwise = bool(anisotropic_leafwise)

        if self.cp_height <= 0:
            raise ValueError("cp_height must be > 0")
        if pre_k % 2 == 0:
            raise ValueError("pre_k must be odd")

        g_in = _safe_gn_groups(self.in_ch, num_groups)
        self.pre = nn.Sequential(
            nn.Conv2d(
                self.in_ch,
                self.in_ch,
                kernel_size=(pre_k, 1),
                padding=(pre_k // 2, 0),
                groups=self.in_ch,
                bias=False,
            ),
            nn.GroupNorm(g_in, self.in_ch),
            nn.GELU(),
        )

        self.idx_ptv = [0, 1, 2]
        self.idx_ext = [3, 4]
        self.idx_oars = list(range(5, 16))
        if len(self.idx_oars) != 11:
            raise ValueError("Expected 11 OAR channels from 5..15")

        if not (0 <= self.shoulder_oar_index_in_oars < 11):
            raise ValueError("shoulder_oar_index_in_oars must be in [0,10]")
        shoulder_global_idx = self.idx_oars[self.shoulder_oar_index_in_oars]
        self.idx_shoulder = [shoulder_global_idx]

        self.idx_oar_other = [i for i in self.idx_oars if i != shoulder_global_idx]
        if len(self.idx_oar_other) != 10:
            raise ValueError("Expected 10 other OAR channels after removing shoulder")

        ptv_in = len(self.idx_ptv) * self.cp_height
        ext_in = len(self.idx_ext) * self.cp_height
        sh_in = len(self.idx_shoulder) * self.cp_height
        oar_in = len(self.idx_oar_other) * self.cp_height

        ptv_ch = max(16, self.out_ch // 2)
        ext_ch = max(16, self.out_ch // 4)
        sh_ch = max(16, self.out_ch // 4)
        oar_ch = max(16, self.out_ch // 2)

        self.ptv_branch = BranchConv(
            ptv_in, ptv_ch, num_groups=num_groups, dropout=dropout, anisotropic_leafwise=self.anisotropic_leafwise
        )
        self.ext_branch = BranchConv(
            ext_in, ext_ch, num_groups=num_groups, dropout=dropout, anisotropic_leafwise=self.anisotropic_leafwise
        )
        self.sh_branch = BranchConv(
            sh_in, sh_ch, num_groups=num_groups, dropout=dropout, anisotropic_leafwise=self.anisotropic_leafwise
        )
        self.oar_branch = BranchConv(
            oar_in, oar_ch, num_groups=num_groups, dropout=dropout, anisotropic_leafwise=self.anisotropic_leafwise
        )

        fuse_in = ptv_ch + ext_ch + sh_ch + oar_ch
        g_out = _safe_gn_groups(self.out_ch, num_groups)
        kf = (3, 1) if self.anisotropic_leafwise else (3, 3)
        pf = (1, 0) if self.anisotropic_leafwise else (1, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, self.out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(g_out, self.out_ch),
            nn.GELU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=kf, padding=pf, bias=False),
            nn.GroupNorm(g_out, self.out_ch),
            nn.GELU(),
        )

        if self.shoulder_gate:
            gate_in = sh_ch + ext_ch
            self.gate_conv = nn.Sequential(
                nn.Conv2d(gate_in, self.out_ch, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.gate_strength = nn.Parameter(torch.tensor(float(film_like_init)))
        else:
            self.gate_conv = None
            self.gate_strength = None

    def _repack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hpx, W = x.shape
        if C != self.in_ch:
            raise ValueError(f"CPStemMultiBranch expected Cin={self.in_ch}, got {C}")
        if Hpx % self.cp_height != 0:
            raise ValueError(f"Hpx={Hpx} must be divisible by cp_height={self.cp_height}")
        Hcp = Hpx // self.cp_height
        x = x.view(B, C, Hcp, self.cp_height, W).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, C * self.cp_height, Hcp, W)
        return x

    def _select_group(self, x_rep: torch.Tensor, idx: List[int]) -> torch.Tensor:
        B, _, Hcp, W = x_rep.shape
        C = self.in_ch
        x5 = x_rep.view(B, C, self.cp_height, Hcp, W)
        xs = x5[:, idx, :, :, :]
        return xs.reshape(B, len(idx) * self.cp_height, Hcp, W).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x_rep = self._repack(x)

        x_ptv = self._select_group(x_rep, self.idx_ptv)
        x_ext = self._select_group(x_rep, self.idx_ext)
        x_sh = self._select_group(x_rep, self.idx_shoulder)
        x_oar = self._select_group(x_rep, self.idx_oar_other)

        f_ptv = self.ptv_branch(x_ptv)
        f_ext = self.ext_branch(x_ext)
        f_sh = self.sh_branch(x_sh)
        f_oar = self.oar_branch(x_oar)

        fused = self.fuse(torch.cat([f_ptv, f_ext, f_sh, f_oar], dim=1))

        if self.shoulder_gate:
            gate = self.gate_conv(torch.cat([f_sh, f_ext], dim=1))
            strength = torch.sigmoid(self.gate_strength)
            fused = fused * (1.0 - strength * gate)

        return fused


# -------------------------------------------------
# Early width adjustment (optionnel)
# -------------------------------------------------

class EarlyWidthDownsample(nn.Module):
    """Ajuste W_in -> W_out via interpolation area (si besoin)."""

    def __init__(self, w_in: int, w_out: int, mode: str = "nearest"):
        super().__init__()
        self.w_in = int(w_in)
        self.w_out = int(w_out)
        self.mode = str(mode)
        if self.mode not in ("nearest", "area", "bilinear"):
            raise ValueError("mode must be one of: nearest, area, bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.w_in == self.w_out:
            return x
        if self.mode == "nearest":
            return F.interpolate(x, size=(x.shape[-2], self.w_out), mode="nearest")
        if self.mode == "bilinear":
            return F.interpolate(x, size=(x.shape[-2], self.w_out), mode="bilinear", align_corners=False)
        return F.interpolate(x, size=(x.shape[-2], self.w_out), mode="area")


# -------------------------------------------------
# U-Net blocks
# -------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8, anisotropic_leafwise: bool = False):
        super().__init__()
        g = _safe_gn_groups(out_ch, num_groups)
        k = (3, 1) if bool(anisotropic_leafwise) else (3, 3)
        p = (1, 0) if bool(anisotropic_leafwise) else (1, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDoubleConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        anisotropic_leafwise: bool = False,
    ):
        super().__init__()
        g = _safe_gn_groups(out_ch, num_groups)
        k = (3, 1) if bool(anisotropic_leafwise) else (3, 3)
        p = (1, 0) if bool(anisotropic_leafwise) else (1, 1)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.GELU(),
            nn.Dropout2d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        return F.gelu(h + self.proj(x))


class Down(nn.Module):
    """Downsample H par 2, W par pool_w (1/2/4)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pool_w: int = 1,
        use_resblocks: bool = True,
        num_groups: int = 8,
        dropout: float = 0.0,
        anisotropic_leafwise: bool = False,
    ):
        super().__init__()
        pool_w = int(pool_w)
        if pool_w not in (1, 2, 4):
            raise ValueError(f"pool_w must be 1/2/4, got {pool_w}")
        self.pool = nn.MaxPool2d(kernel_size=(2, pool_w), stride=(2, pool_w))
        self.conv = (
            ResidualDoubleConv(
                in_ch,
                out_ch,
                num_groups=num_groups,
                dropout=dropout,
                anisotropic_leafwise=anisotropic_leafwise,
            )
            if use_resblocks
            else DoubleConv(in_ch, out_ch, num_groups=num_groups, anisotropic_leafwise=anisotropic_leafwise)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsample H par 2, W par up_w (1/2/4) avec interpolation + conv (anti-checkerboard).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        up_w: int = 1,
        use_resblocks: bool = True,
        num_groups: int = 8,
        dropout: float = 0.0,
        anisotropic_leafwise: bool = False,
    ):
        super().__init__()
        up_w = int(up_w)
        if up_w not in (1, 2, 4):
            raise ValueError(f"up_w must be 1/2/4, got {up_w}")
        self.up_w = up_w

        # "post-up" conv pour mixer proprement (évite artefacts convtranspose)
        self.post = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=(3, 1) if bool(anisotropic_leafwise) else 3,
            padding=(1, 0) if bool(anisotropic_leafwise) else 1,
            bias=False,
        )

        # après concat: in_ch(from up) + out_ch(from skip) -> out_ch
        self.conv = (
            ResidualDoubleConv(
                in_ch + out_ch,
                out_ch,
                num_groups=num_groups,
                dropout=dropout,
                anisotropic_leafwise=anisotropic_leafwise,
            )
            if use_resblocks
            else DoubleConv(in_ch + out_ch, out_ch, num_groups=num_groups, anisotropic_leafwise=anisotropic_leafwise)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: decoder, x2: skip
        x1 = F.interpolate(x1, scale_factor=(2, self.up_w), mode="bilinear", align_corners=False)
        x1 = self.post(x1)

        # pad si mismatch dû à dimensions impaires
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------------------------------
# Bottleneck (conv-only + option CPTransformer)
# -------------------------------------------------

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        dilated_blocks: int = 1,
        dilated_dropout: float = 0.05,
        horiz_blocks: int = 1,
        horiz_k: int = 11,
        horiz_dropout: float = 0.05,
        num_groups: int = 8,
        anisotropic_leafwise: bool = False,
    ):
        super().__init__()
        g = _safe_gn_groups(ch, num_groups)

        layers: List[nn.Module] = []
        for i in range(int(dilated_blocks)):
            d = 1 + i
            if bool(anisotropic_leafwise):
                k = (3, 1)
                p = (d, 0)
                dil = (d, 1)
            else:
                k = 3
                p = d
                dil = d
            layers += [
                nn.Conv2d(ch, ch, k, padding=p, dilation=dil, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=ch),
                nn.GELU(),
                nn.Dropout2d(p=float(dilated_dropout)) if dilated_dropout and dilated_dropout > 0 else nn.Identity(),
            ]
        self.dilated = nn.Sequential(*layers) if layers else nn.Identity()

        k = int(horiz_k)
        if k % 2 == 0:
            k += 1
        layers2: List[nn.Module] = []
        for _ in range(int(horiz_blocks)):
            layers2 += [
                nn.Conv2d(ch, ch, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                nn.GroupNorm(num_groups=g, num_channels=ch),
                nn.GELU(),
                nn.Dropout2d(p=float(horiz_dropout)) if horiz_dropout and horiz_dropout > 0 else nn.Identity(),
            ]
        self.horiz = nn.Sequential(*layers2) if layers2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dilated(x)
        x = x + self.horiz(x)
        return x


class CPTransformer(nn.Module):
    """Transformer sur la dimension CP (H) en moyennant W (optionnel)."""

    def __init__(
        self,
        ch: int,
        d_model: int = 256,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        layers: int = 4,
        use_ckpt: bool = False,
    ):
        super().__init__()
        self.use_ckpt = bool(use_ckpt)
        self.to_d = nn.Linear(ch, d_model)
        self.to_c = nn.Linear(d_model, ch)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=int(layers))

    def _forward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        x = feat.mean(dim=-1).permute(0, 2, 1).contiguous()  # [B,H,C]
        x = self.to_d(x)
        if self.use_ckpt and checkpoint is not None and self.training:
            x = checkpoint(self._forward_block, x)
        else:
            x = self.enc(x)
        x = self.to_c(x)
        x = x.permute(0, 2, 1).unsqueeze(-1).expand(B, C, H, W)
        return feat + x


# -------------------------------------------------
# Global full-sequence refiner (post-stitching)
# -------------------------------------------------


class _TemporalResBlock1D(nn.Module):
    """Light residual temporal block for CP-axis refinement."""

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        d = max(1, int(dilation))
        pad = (k // 2) * d
        g = _safe_gn_groups(ch, 8)
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, dilation=d, padding=pad, bias=False),
            nn.GroupNorm(g, ch),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv1d(ch, ch, kernel_size=1, bias=False),
            nn.GroupNorm(g, ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class GlobalCPRefiner1D(nn.Module):
    """Global CP-wise residual refiner operating on full stitched sinograms."""

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 4,
        kernel_size: int = 5,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.05,
        cond_dim: int = 32,
        film_dim: int = 0,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.cond_dim = max(0, int(cond_dim))
        self.film_dim = max(0, int(film_dim))

        dils = list(dilations) if dilations is not None and len(dilations) > 0 else [1, 2, 4, 8]
        self.dilations = [max(1, int(d)) for d in dils]

        cond_in = 2 + self.film_dim
        if self.cond_dim > 0 and cond_in > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_in, self.cond_dim),
                nn.GELU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
            extra_ch = self.cond_dim
        else:
            self.cond_proj = None
            extra_ch = 0

        self.in_proj = nn.Conv1d(64 + extra_ch, self.hidden, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            _TemporalResBlock1D(
                ch=self.hidden,
                kernel_size=int(kernel_size),
                dilation=self.dilations[i % len(self.dilations)],
                dropout=float(dropout),
            )
            for i in range(max(1, self.layers))
        ])
        self.out_proj = nn.Conv1d(self.hidden, 64, kernel_size=1, bias=True)

        # Start from identity behavior: delta ~= 0 at initialization.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    @staticmethod
    def _ensure_seq_3d(x: Optional[torch.Tensor], ncp: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if x is None:
            return torch.zeros((1, ncp, 1), device=device, dtype=dtype)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.shape[1] != ncp:
            x = _resample_len(x, ncp)
        return x.to(device=device, dtype=dtype)

    def _build_condition(
        self,
        bsz: int,
        ncp: int,
        dtype: torch.dtype,
        device: torch.device,
        angles: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        film: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.cond_proj is None:
            return None

        a = self._ensure_seq_3d(angles, ncp, dtype, device)
        p = self._ensure_seq_3d(positions, ncp, dtype, device)

        if a.shape[0] == 1 and bsz > 1:
            a = a.expand(bsz, ncp, 1)
        if p.shape[0] == 1 and bsz > 1:
            p = p.expand(bsz, ncp, 1)

        if self.film_dim > 0:
            if film is None:
                f = torch.zeros((bsz, self.film_dim), device=device, dtype=dtype)
            else:
                if film.dim() == 3:
                    f = film[:, 0, :]
                else:
                    f = film
                f = f.to(device=device, dtype=dtype)
                if f.shape[-1] != self.film_dim:
                    if f.shape[-1] > self.film_dim:
                        f = f[:, : self.film_dim]
                    else:
                        pad = self.film_dim - f.shape[-1]
                        f = F.pad(f, (0, pad), value=0.0)
            f = f.unsqueeze(1).expand(bsz, ncp, self.film_dim)
            cond_in = torch.cat([a, p, f], dim=-1)
        else:
            cond_in = torch.cat([a, p], dim=-1)

        return self.cond_proj(cond_in)

    def forward(
        self,
        y_base_full: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        film: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_is_3d = y_base_full.dim() == 3
        if input_is_3d:
            y_base = y_base_full.unsqueeze(1)
        elif y_base_full.dim() == 4:
            y_base = y_base_full
        else:
            raise ValueError(f"GlobalCPRefiner1D expects [B,Ncp,64] or [B,1,Ncp,64], got {tuple(y_base_full.shape)}")

        if y_base.shape[1] != 1:
            raise ValueError(f"Expected channel dimension 1 for y_base_full, got {tuple(y_base.shape)}")
        if y_base.shape[-1] != 64:
            raise ValueError(f"Expected detector width 64 for refiner input, got {tuple(y_base.shape)}")

        bsz, _, ncp, _ = y_base.shape
        x = y_base.squeeze(1).transpose(1, 2).contiguous()  # [B,64,Ncp]

        cond = self._build_condition(
            bsz=bsz,
            ncp=ncp,
            dtype=x.dtype,
            device=x.device,
            angles=angles,
            positions=positions,
            film=film,
        )
        if cond is not None:
            cond_ch = cond.transpose(1, 2).contiguous()  # [B,cond_dim,Ncp]
            x = torch.cat([x, cond_ch], dim=1)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        delta = self.out_proj(h).transpose(1, 2).unsqueeze(1).contiguous()  # [B,1,Ncp,64]

        y_refined = y_base + self.alpha.to(dtype=y_base.dtype) * delta
        if input_is_3d:
            return y_refined.squeeze(1)
        return y_refined


class PatchCPHead1D(nn.Module):
    """Local CP-only residual head applied on patch predictions [B,1,Ncp,W]."""

    def __init__(
        self,
        detector_width: int = 64,
        hidden: int = 128,
        layers: int = 3,
        kernel_size: int = 5,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.05,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.detector_width = int(detector_width)
        self.hidden = int(hidden)
        dils = list(dilations) if dilations is not None and len(dilations) > 0 else [1, 2, 4]
        self.dilations = [max(1, int(d)) for d in dils]

        self.in_proj = nn.Conv1d(self.detector_width, self.hidden, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            _TemporalResBlock1D(
                ch=self.hidden,
                kernel_size=int(kernel_size),
                dilation=self.dilations[i % len(self.dilations)],
                dropout=float(dropout),
            )
            for i in range(max(1, int(layers)))
        ])
        self.out_proj = nn.Conv1d(self.hidden, self.detector_width, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        if y0.dim() != 4 or y0.shape[1] != 1:
            raise ValueError(f"PatchCPHead1D expects [B,1,Ncp,W], got {tuple(y0.shape)}")
        if y0.shape[-1] != self.detector_width:
            raise ValueError(f"PatchCPHead1D expected detector width {self.detector_width}, got {y0.shape[-1]}")

        x = y0.squeeze(1).transpose(1, 2).contiguous()  # [B,W,Ncp]
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        delta = self.out_proj(h).transpose(1, 2).unsqueeze(1).contiguous()  # [B,1,Ncp,W]
        return y0 + self.alpha.to(dtype=y0.dtype) * delta


class TwoStageSinoModel(nn.Module):
    """Composite model: patch-wise backbone + optional global full-sequence refiner."""

    @classmethod
    def from_configs(cls, model_cfg, data_cfg):
        backbone = G1TransUnet.from_configs(model_cfg, data_cfg)
        use_global_refiner = bool(getattr(model_cfg, "use_global_refiner", False))
        refiner = None
        if use_global_refiner:
            refiner = GlobalCPRefiner1D(
                hidden=int(getattr(model_cfg, "refiner_hidden", 128)),
                layers=int(getattr(model_cfg, "refiner_layers", 4)),
                kernel_size=int(getattr(model_cfg, "refiner_kernel_size", 5)),
                dilations=list(getattr(model_cfg, "refiner_dilations", [1, 2, 4, 8])),
                dropout=float(getattr(model_cfg, "refiner_dropout", 0.05)),
                cond_dim=int(getattr(model_cfg, "refiner_cond_dim", 32)),
                film_dim=int(getattr(model_cfg, "film_extra_dim", 0)),
                alpha_init=float(getattr(model_cfg, "refiner_alpha_init", 0.0)),
            )
        return cls(backbone=backbone, global_refiner=refiner)

    def __init__(self, backbone: nn.Module, global_refiner: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.global_refiner = global_refiner
        self.use_global_refiner = global_refiner is not None

    def forward(
        self,
        x_drr: torch.Tensor,
        angles: torch.Tensor,
        positions: torch.Tensor,
        film_extra: Optional[torch.Tensor] = None,
        present_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_patch(x_drr, angles, positions, film_extra, present_flags)

    def predict_patch(
        self,
        x_drr: torch.Tensor,
        angles: torch.Tensor,
        positions: torch.Tensor,
        film_extra: Optional[torch.Tensor] = None,
        present_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backbone(x_drr, angles, positions, film_extra, present_flags)

    def refine_full(
        self,
        y_base_full: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        film: Optional[torch.Tensor] = None,
        null_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_global_refiner and self.global_refiner is not None:
            y_out = self.global_refiner(y_base_full, angles=angles, positions=positions, film=film)
        else:
            y_out = y_base_full

        if null_mask is not None:
            m = null_mask.to(dtype=y_out.dtype, device=y_out.device)
            if m.dim() == 1:
                m = m.view(-1, 1, 1, 1)
            y_out = y_out * (1.0 - m)
        return y_out

    def get_refiner_parameters(self) -> List[nn.Parameter]:
        if not self.use_global_refiner or self.global_refiner is None:
            return []
        return list(self.global_refiner.parameters())


# -------------------------------------------------
# Main model
# -------------------------------------------------

class G1TransUnet(nn.Module):
    @classmethod
    def from_configs(cls, model_cfg, data_cfg):
        """Build helper using dataclass/dict configs to avoid a long kwargs list at call site."""
        return cls(
            in_ch_drr=int(model_cfg.in_ch_drr),
            out_ch=int(model_cfg.out_ch),
            img_size=(int(getattr(data_cfg, "patch_in_cp", data_cfg.patch_cp)) * int(data_cfg.cp_height), int(data_cfg.W_in)),
            out_width=int(data_cfg.W),
            base_ch=int(model_cfg.base_ch),
            depth=int(model_cfg.depth),
            cp_height=int(data_cfg.cp_height),
            use_sincos_angles=bool(model_cfg.use_sincos_angles),
            film_hidden=int(model_cfg.film_hidden),
            film_extra_dim=int(model_cfg.film_extra_dim),
            film_embed_dim=int(getattr(model_cfg, "film_embed_dim", 32)),
            film_embed_hidden=int(getattr(model_cfg, "film_embed_hidden", 64)),
            pool_w_levels=int(model_cfg.pool_w_levels),
            pool_w_factor=int(model_cfg.pool_w_factor),
            use_transformer=bool(model_cfg.use_transformer),
            d_model=int(model_cfg.d_model),
            nhead=int(model_cfg.nhead),
            mlp_ratio=float(model_cfg.mlp_ratio),
            transformer_layers=int(model_cfg.transformer_layers),
            use_ckpt=bool(model_cfg.use_ckpt),
            bneck_dilated_blocks=int(model_cfg.bneck_dilated_blocks),
            bneck_dropout=float(model_cfg.bneck_dropout),
            bneck_horiz_blocks=int(model_cfg.bneck_horiz_blocks),
            bneck_horiz_k=int(model_cfg.bneck_horiz_k),
            bneck_horiz_dropout=float(model_cfg.bneck_horiz_dropout),
            use_film_all_levels=bool(model_cfg.use_film_all_levels),
            film_on_decoder=bool(getattr(model_cfg, "film_on_decoder", False)),
            use_resblocks=bool(model_cfg.use_resblocks),
            res_dropout=float(model_cfg.res_dropout),
            use_missing_token=bool(getattr(model_cfg, "use_missing_token", False)),
            num_groups=int(model_cfg.num_groups),
            shape_debug=bool(model_cfg.shape_debug),
            stem_pre_k=int(getattr(model_cfg, "stem_pre_k", 3)),
            shoulder_gate=bool(getattr(model_cfg, "shoulder_gate", True)),
            shoulder_oar_index_in_oars=int(getattr(model_cfg, "shoulder_oar_index_in_oars", 9)),
            anisotropic_leafwise=bool(getattr(model_cfg, "anisotropic_leafwise", False)),
            film_light_mode=bool(getattr(model_cfg, "film_light_mode", False)),
            early_width_mode=str(getattr(model_cfg, "early_width_mode", "nearest")),
            use_patch_cp_head=bool(getattr(model_cfg, "use_patch_cp_head", False)),
            patch_cp_head_hidden=int(getattr(model_cfg, "patch_cp_head_hidden", 128)),
            patch_cp_head_layers=int(getattr(model_cfg, "patch_cp_head_layers", 3)),
            patch_cp_head_kernel_size=int(getattr(model_cfg, "patch_cp_head_kernel_size", 5)),
            patch_cp_head_dilations=list(getattr(model_cfg, "patch_cp_head_dilations", [1, 2, 4])),
            patch_cp_head_dropout=float(getattr(model_cfg, "patch_cp_head_dropout", 0.05)),
            patch_cp_head_alpha_init=float(getattr(model_cfg, "patch_cp_head_alpha_init", 0.0)),
        )

    def __init__(
        self,
        in_ch_drr: int,
        out_ch: int,
        img_size: Tuple[int, int] = (6144, 64),
        out_width: int = 64,
        base_ch: int = 64,
        depth: int = 3,
        cp_height: int = 12,
        use_sincos_angles: bool = True,
        film_hidden: int = 512,
        film_extra_dim: int = 0,
        film_embed_dim: int = 32,
        film_embed_hidden: int = 64,
        pool_w_levels: int = 0,
        pool_w_factor: int = 2,
        use_transformer: bool = False,
        d_model: int = 256,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        transformer_layers: int = 4,
        use_ckpt: bool = False,
        bneck_dilated_blocks: int = 1,
        bneck_dropout: float = 0.05,
        bneck_horiz_blocks: int = 1,
        bneck_horiz_k: int = 11,
        bneck_horiz_dropout: float = 0.05,
        use_film_all_levels: bool = True,
        film_on_decoder: bool = False,
        use_resblocks: bool = True,
        res_dropout: float = 0.0,
        use_missing_token: bool = False,
        num_groups: int = 8,
        shape_debug: bool = False,
        stem_pre_k: int = 3,
        shoulder_gate: bool = True,
        shoulder_oar_index_in_oars: int = 9,
        anisotropic_leafwise: bool = False,
        film_light_mode: bool = False,
        early_width_mode: str = "nearest",
        use_patch_cp_head: bool = False,
        patch_cp_head_hidden: int = 128,
        patch_cp_head_layers: int = 3,
        patch_cp_head_kernel_size: int = 5,
        patch_cp_head_dilations: Optional[List[int]] = None,
        patch_cp_head_dropout: float = 0.05,
        patch_cp_head_alpha_init: float = 0.0,
    ):
        super().__init__()

        self.shape_debug = bool(shape_debug)
        self.use_film_all_levels = bool(use_film_all_levels)
        self.film_on_decoder = bool(film_on_decoder)
        self.use_sincos_angles = bool(use_sincos_angles)
        self.film_light_mode = bool(film_light_mode)
        self.anisotropic_leafwise = bool(anisotropic_leafwise)

        Hpx, W_in = int(img_size[0]), int(img_size[1])
        self.cp_height = int(cp_height)
        if Hpx % self.cp_height != 0:
            raise ValueError(f"img_size[0]={Hpx} must be divisible by cp_height={self.cp_height}")
        self.Hcp = Hpx // self.cp_height
        self.W_in = W_in
        self.W_out = int(out_width)

        self.missing_token = MissingChannelToken(in_ch_drr) if bool(use_missing_token) else None

        self.stem = CPStemMultiBranch(
            in_ch=in_ch_drr,
            out_ch=base_ch,
            cp_height=self.cp_height,
            w_in=self.W_in,
            num_groups=num_groups,
            dropout=res_dropout,
            pre_k=stem_pre_k,
            film_like_init=0.0,
            shoulder_gate=shoulder_gate,
            shoulder_oar_index_in_oars=shoulder_oar_index_in_oars,
            anisotropic_leafwise=self.anisotropic_leafwise,
        )
        self.early_w = EarlyWidthDownsample(w_in=self.W_in, w_out=self.W_out, mode=early_width_mode)

        self.inc = (
            ResidualDoubleConv(
                base_ch,
                base_ch,
                num_groups=num_groups,
                dropout=res_dropout,
                anisotropic_leafwise=self.anisotropic_leafwise,
            )
            if use_resblocks
            else DoubleConv(base_ch, base_ch, num_groups=num_groups, anisotropic_leafwise=self.anisotropic_leafwise)
        )

        pool_w_levels = max(int(pool_w_levels), 0)
        if self.anisotropic_leafwise:
            pool_w_levels = 0
        pool_w_factor = int(pool_w_factor)
        if pool_w_factor not in (2, 4):
            pool_w_factor = 2

        pool_w_per_level: List[int] = []
        for i in range(max(depth - 1, 0)):
            pool_w_per_level.append(pool_w_factor if i < pool_w_levels else 1)

        w_tmp = self.W_out
        for i in range(len(pool_w_per_level)):
            pw = pool_w_per_level[i]
            if pw != 1 and (w_tmp % pw) != 0:
                pool_w_per_level[i] = 1
            w_tmp = w_tmp // pool_w_per_level[i]

        self.downs = nn.ModuleList()
        ch = base_ch
        enc_channels = [ch]
        for i in range(depth - 1):
            outc = ch * 2
            self.downs.append(
                Down(
                    ch,
                    outc,
                    pool_w=pool_w_per_level[i],
                    use_resblocks=use_resblocks,
                    num_groups=num_groups,
                    dropout=res_dropout,
                    anisotropic_leafwise=self.anisotropic_leafwise,
                )
            )
            ch = outc
            enc_channels.append(ch)

        if self.anisotropic_leafwise:
            bneck_horiz_blocks = 0

        self.bottleneck = BottleneckBlock(
            ch,
            dilated_blocks=bneck_dilated_blocks,
            dilated_dropout=bneck_dropout,
            horiz_blocks=bneck_horiz_blocks,
            horiz_k=bneck_horiz_k,
            horiz_dropout=bneck_horiz_dropout,
            num_groups=num_groups,
            anisotropic_leafwise=self.anisotropic_leafwise,
        )

        self.tr = CPTransformer(
            ch,
            d_model=d_model,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            layers=transformer_layers,
            use_ckpt=use_ckpt,
        ) if use_transformer else None

        self.ups = nn.ModuleList()
        pool_w_rev = list(reversed(pool_w_per_level))
        for i in range(depth - 1):
            self.ups.append(
                Up(
                    ch,
                    ch // 2,
                    up_w=pool_w_rev[i],
                    use_resblocks=use_resblocks,
                    num_groups=num_groups,
                    dropout=res_dropout,
                    anisotropic_leafwise=self.anisotropic_leafwise,
                )
            )
            ch = ch // 2

        self.outc = OutConv(ch, out_ch)
        self.patch_cp_head = None
        if bool(use_patch_cp_head):
            if int(out_ch) != 1:
                raise ValueError("use_patch_cp_head=True requires out_ch=1")
            self.patch_cp_head = PatchCPHead1D(
                detector_width=self.W_out,
                hidden=int(patch_cp_head_hidden),
                layers=int(patch_cp_head_layers),
                kernel_size=int(patch_cp_head_kernel_size),
                dilations=patch_cp_head_dilations,
                dropout=float(patch_cp_head_dropout),
                alpha_init=float(patch_cp_head_alpha_init),
            )

        self.loc_proj = nn.Linear(1, d_model)
        self.angle_embed = SineCosineEncoding(out_dim=d_model) if self.use_sincos_angles else nn.Linear(1, d_model)

        self.film_in_dim = int(film_extra_dim)
        self.film_embed_dim = int(film_embed_dim) if (self.film_in_dim > 0 and int(film_embed_dim) > 0) else 0
        if self.film_in_dim > 0 and self.film_embed_dim > 0:
            self.film_embed = FilmEmbed(self.film_in_dim, self.film_embed_dim, hidden=int(film_embed_hidden))
        else:
            self.film_embed = None

        self.cond_dim = int(d_model) + int(d_model) + (self.film_embed_dim if self.film_embed is not None else 0)

        self.film_global = FiLMMap(self.cond_dim, base_ch, hidden=film_hidden)
        self.film_inc = FiLMMap(self.cond_dim, base_ch, hidden=film_hidden)
        self.film_down = nn.ModuleList([
            FiLMMap(self.cond_dim, c, hidden=film_hidden) for c in enc_channels[1:]
        ])
        self.film_bneck = FiLMMap(self.cond_dim, enc_channels[-1], hidden=film_hidden)
        self.film_dec = nn.ModuleList([
            FiLMMap(self.cond_dim, c, hidden=film_hidden) for c in reversed(enc_channels[:-1])
        ])

    def _cond(
        self,
        angles: torch.Tensor,
        positions: torch.Tensor,
        target_len: int,
        film_extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        angles = _resample_len(angles, target_len)
        positions = _resample_len(positions, target_len)

        a = self.angle_embed(angles)
        p = self.loc_proj(positions)

        if self.film_embed is None:
            return torch.cat([a, p], dim=-1)

        B = angles.shape[0]
        if film_extra is None:
            e = torch.zeros((B, target_len, self.film_in_dim), device=angles.device, dtype=a.dtype)
        else:
            if film_extra.dim() == 2:
                e = film_extra[:, None, :].to(device=angles.device).expand(B, target_len, self.film_in_dim)
            else:
                e = film_extra
                if e.shape[1] != target_len:
                    e = _resample_len(e, target_len)
                e = e.to(device=angles.device)
            e = e.to(dtype=a.dtype)

        e2 = self.film_embed(e.reshape(B * target_len, self.film_in_dim)).reshape(B, target_len, self.film_embed_dim)
        return torch.cat([a, p, e2], dim=-1)

    def forward(
        self,
        x_drr: torch.Tensor,
        angles: torch.Tensor,
        positions: torch.Tensor,
        film_extra: Optional[torch.Tensor] = None,
        present_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.missing_token is not None:
            x_drr = self.missing_token(x_drr, present_flags)

        x = self.stem(x_drr)
        x = self.early_w(x)
        if self.shape_debug:
            print("stem+earlyW:", x.shape)

        cond_cache: Dict[int, torch.Tensor] = {}

        def get_cond(h: int) -> torch.Tensor:
            if h not in cond_cache:
                cond_cache[h] = self._cond(angles, positions, target_len=h, film_extra=film_extra)
            return cond_cache[h]

        cond0 = get_cond(x.shape[2])
        x = _apply_film(x, self.film_global(cond0))

        x1 = self.inc(x)
        if self.use_film_all_levels and not self.film_light_mode:
            x1 = _apply_film(x1, self.film_inc(cond0))
        if self.shape_debug:
            print("inc:", x1.shape)

        skips = [x1]
        xi = x1
        for i, down in enumerate(self.downs):
            xi = down(xi)
            if self.use_film_all_levels and not self.film_light_mode:
                cond_i = get_cond(xi.shape[2])
                xi = _apply_film(xi, self.film_down[i](cond_i))
            skips.append(xi)
            if self.shape_debug:
                print(f"down{i}:", xi.shape)

        xb = self.bottleneck(xi)
        if self.tr is not None:
            xb = self.tr(xb)
        if self.use_film_all_levels:
            cond_b = get_cond(xb.shape[2])
            xb = _apply_film(xb, self.film_bneck(cond_b))
        if self.shape_debug:
            print("bneck:", xb.shape)

        xu = xb
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            xu = up(xu, skip)
            if self.use_film_all_levels and self.film_on_decoder:
                cond_u = get_cond(xu.shape[2])
                xu = _apply_film(xu, self.film_dec[i](cond_u))
            if self.shape_debug:
                print(f"up{i}:", xu.shape)

        y = self.outc(xu)
        if self.patch_cp_head is not None:
            y = self.patch_cp_head(y)
        if self.shape_debug:
            print("outc:", y.shape)
        return y
