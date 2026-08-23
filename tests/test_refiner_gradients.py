"""
Regression tests for the defect refiner.

The previous pipeline's generator trained for 10 epochs with a reconstruction
term that was identically zero. It computed

    L1(fake * (1 - mask), real * (1 - mask))

while the generator already returned `bg * (1 - mask) + out * mask`, so outside
the mask the two arguments were bit-identical tensors. Autograd dutifully
returned zero and the checkpoint was shaped almost entirely by a 0.01-weighted
adversarial term.

These tests exist so that specific failure cannot come back unnoticed.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth.refiner import (DefectRefiner, RefinerDiscriminator, masked_l1,
                           spectral_l1)

B, C, P = 2, 3, 64
N_CLASSES = 2


def _batch(mask_frac: float = 0.05):
    torch.manual_seed(0)
    img = torch.rand(B, C, P, P)
    real = torch.rand(B, C, P, P)
    mask = torch.zeros(B, 1, P, P)
    side = max(2, int(P * mask_frac ** 0.5))
    mask[:, :, :side, :side] = 1.0
    cls = torch.zeros(B, N_CLASSES)
    cls[:, 0] = 1.0
    sev = torch.full((B,), 0.7)
    return img, real, mask, cls, sev


def test_reconstruction_loss_is_nonzero_and_has_gradient():
    """The exact bug that killed the previous generator."""
    img, real, mask, cls, sev = _batch()
    G = DefectRefiner(base=8, n_classes=N_CLASSES)
    out = G(img, mask, cls, sev)

    loss = masked_l1(out, real, mask)
    assert loss.item() > 0, "masked reconstruction loss is zero - the old bug is back"

    loss.backward()
    total = sum(p.grad.abs().sum().item() for p in G.parameters() if p.grad is not None)
    assert total > 0, "no gradient reached the generator from the reconstruction term"


def test_outside_mask_is_preserved_exactly():
    """Composite must not touch background - that is what keeps labels true."""
    img, _, mask, cls, sev = _batch()
    G = DefectRefiner(base=8, n_classes=N_CLASSES)
    with torch.no_grad():
        out = G(img, mask, cls, sev)
    outside = (mask == 0).expand_as(img)
    assert torch.allclose(out[outside], img[outside], atol=1e-6), \
        "refiner modified pixels outside the mask - geometry/labels no longer agree"


def test_inside_mask_actually_changes():
    """The complement: if nothing changes inside, the refiner is a no-op."""
    img, _, mask, cls, sev = _batch()
    G = DefectRefiner(base=8, n_classes=N_CLASSES)
    with torch.no_grad():
        out = G(img, mask, cls, sev)
    inside = (mask > 0).expand_as(img)
    assert not torch.allclose(out[inside], img[inside], atol=1e-6), \
        "refiner left the masked region untouched"


def test_conditioning_changes_the_output():
    """Class/severity must reach the pixels, or the labels drift from content."""
    img, _, mask, _, _ = _batch()
    G = DefectRefiner(base=8, n_classes=N_CLASSES).eval()

    c0 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    c1 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    sev_lo, sev_hi = torch.full((B,), 0.1), torch.full((B,), 0.9)
    with torch.no_grad():
        by_class = (G(img, mask, c0, sev_lo) - G(img, mask, c1, sev_lo)).abs().sum()
        by_sev = (G(img, mask, c0, sev_lo) - G(img, mask, c0, sev_hi)).abs().sum()
    assert by_class > 0, "class conditioning has no effect on the output"
    assert by_sev > 0, "severity conditioning has no effect on the output"


def test_spectral_loss_is_nonzero_on_different_images():
    a, b = torch.rand(B, C, P, P), torch.rand(B, C, P, P)
    assert spectral_l1(a, b).item() > 0
    assert spectral_l1(a, a).item() < 1e-5


def test_fft_branch_is_ablatable():
    """H2 / ablation A1 must be a real structural difference, not a flag no-op."""
    x = torch.rand(B, C, P, P)
    with_fft = RefinerDiscriminator(use_fft=True)
    without = RefinerDiscriminator(use_fft=False)
    assert len(with_fft(x)) == 2, "expected spatial + fourier outputs"
    assert len(without(x)) == 1, "fourier branch still active with use_fft=False"
    assert without.fourier is None


def test_discriminator_gradients_flow():
    img, real, mask, cls, sev = _batch()
    G = DefectRefiner(base=8, n_classes=N_CLASSES)
    D = RefinerDiscriminator(ndf=8, use_fft=True)
    fake = G(img, mask, cls, sev)
    loss = sum(o.mean() for o in D(fake))
    loss.backward()
    total = sum(p.grad.abs().sum().item() for p in D.parameters() if p.grad is not None)
    assert total > 0, "no gradient reached the discriminator"
