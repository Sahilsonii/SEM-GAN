"""
Conditional GAN texture refiner (stage 5).

The renderer gives geometry: exact masks, exact boxes, exact classes. What it
cannot give is FESEM texture - its defects read as clean drawn blobs. This
module learns that texture and paints it into the rendered region, leaving the
geometry (and therefore the labels) untouched.

Division of responsibility:
    renderer  ->  WHERE the defect is, and how big / how severe   (controllable)
    refiner   ->  what it LOOKS like                              (learned)

Why the previous pipeline's generator could not work
----------------------------------------------------
It computed `L1(fake * (1 - mask), real * (1 - mask))` while the generator
already returned `bg * (1 - mask) + out * mask`. Outside the mask the two
arguments are bit-identical, so that term was exactly zero with zero gradient
for every batch. Inside the mask there was no reconstruction term at all, which
left `0.01 * adversarial` as the only signal shaping the region the model was
supposed to author.

Here the reconstruction loss is applied INSIDE the mask against real defect
crops, which is the only place the refiner actually authors pixels. Ablation A1
(`--no-fft`) drops the Fourier branch of the discriminator; that comparison is
hypothesis H2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.discriminator import (FourierFrequencyDiscriminator,
                                  SpatialPatchDiscriminator)


class FiLM(nn.Module):
    """Per-channel scale/shift conditioned on (class, severity).

    Conditioning is what makes the refiner controllable rather than a generic
    texture model: it has to be told which defect it is painting, otherwise the
    class label and the pixels drift apart - exactly the failure that made the
    old synthetic labels meaningless.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.SiLU(), nn.Linear(64, channels * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int, down: bool = False):
        super().__init__()
        stride = 2 if down else 1
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, cout)          # batch-independent: batches are tiny on 4 GB
        self.film = FiLM(cond_dim, cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond):
        return self.act(self.film(self.norm(self.conv(x)), cond))


class DefectRefiner(nn.Module):
    """U-Net that repaints the masked region, conditioned on (class, severity).

    Input  : rendered composite (3) + defect mask (1)
    Output : the same image with the masked region re-textured

    The final composite is `bg * (1 - m) + out * m`, so the refiner is
    structurally incapable of moving a defect or changing its extent. Geometry
    stays exactly what the renderer emitted, which is what keeps the labels true.
    """

    def __init__(self, base: int = 32, n_classes: int = 2, cond_dim: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.cond_proj = nn.Linear(n_classes + 1, cond_dim)

        self.e1 = ConvBlock(4, base, cond_dim)
        self.e2 = ConvBlock(base, base * 2, cond_dim, down=True)
        self.e3 = ConvBlock(base * 2, base * 4, cond_dim, down=True)
        self.mid = ConvBlock(base * 4, base * 4, cond_dim)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2, cond_dim)
        self.d1 = ConvBlock(base * 2 + base, base, cond_dim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor,
                cls_onehot: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(torch.cat([cls_onehot, severity[:, None]], dim=1))

        x = torch.cat([img, mask], dim=1)
        e1 = self.e1(x, cond)
        e2 = self.e2(e1, cond)
        e3 = self.e3(e2, cond)
        m = self.mid(e3, cond)

        u2 = F.interpolate(m, size=e2.shape[-2:], mode="nearest")
        d2 = self.d2(torch.cat([u2, e2], dim=1), cond)
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.d1(torch.cat([u1, e1], dim=1), cond)

        residual = torch.tanh(self.out(d1))
        painted = torch.clamp(img + residual, 0.0, 1.0)
        return img * (1 - mask) + painted * mask


class RefinerDiscriminator(nn.Module):
    """Spatial PatchGAN, optionally with the Fourier branch (ablation A1 / H2)."""

    def __init__(self, in_channels: int = 3, ndf: int = 32, use_fft: bool = True):
        super().__init__()
        self.use_fft = use_fft
        self.spatial = SpatialPatchDiscriminator(in_channels, ndf)
        self.fourier = (FourierFrequencyDiscriminator(in_channels, ndf // 2)
                        if use_fft else None)

    def forward(self, x):
        out = [self.spatial(x)]
        if self.fourier is not None:
            out.append(self.fourier(x))
        return out


def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L1 restricted to the mask, normalised by mask area.

    Normalising matters here: defects cover well under 1% of the frame, so an
    unnormalised masked L1 is numerically dominated by whichever sample happens
    to carry the largest blob.
    """
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() * pred.shape[1] + eps)


def spectral_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between log magnitude spectra.

    Applied to mask-centred crops rather than whole frames: comparing full
    images whose pixels are ~99% identical yields a near-zero loss that teaches
    nothing, which is what happened in the previous pipeline.
    """
    fp = torch.fft.fftshift(torch.fft.fft2(pred, dim=(-2, -1)), dim=(-2, -1))
    ft = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))
    return F.l1_loss(torch.log(fp.abs() + 1e-8), torch.log(ft.abs() + 1e-8))
