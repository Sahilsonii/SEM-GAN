"""
Train the defect texture refiner on 192x192 patches.

Training signal
---------------
A batch item is built by taking a REAL annotated patch containing a real defect,
then rendering a synthetic defect of the same class at the same place on the
same background. The refiner sees the rendered version and is asked to make the
masked region look like the real one.

That is the whole idea: the target is a real FESEM defect, so the gradient
inside the mask is real texture rather than a proxy. The old pipeline's L1 term
compared two tensors that were identical by construction and contributed exactly
nothing.

Patches, not whole frames: 192x192 at batch 16 fits 4 GB comfortably, whereas the
previous 512x512 at batch 2 was VRAM-bound while giving the model less defect
signal per step.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.adversarial import GANLoss
from synth.refiner import (DefectRefiner, RefinerDiscriminator, masked_l1,
                           spectral_l1)
from synth.renderer import DefectParams, PBI2, PINHOLE, render

CURATED = ROOT / "data" / "curated"
SPLITS = ROOT / "data" / "splits"
CKPT = ROOT / "checkpoints"
CLASS_KIND = {0: PBI2, 1: PINHOLE}
N_CLASSES = 2


class DefectPatchDataset(Dataset):
    """Real defect patch + a rendered stand-in of the same defect."""

    def __init__(self, split: str = "train", patch: int = 192, min_side_px: int = 6):
        self.patch = patch
        recs = json.loads((SPLITS / f"{split}.json").read_text(encoding="utf-8"))["records"]
        self.items = []
        for r in recs:
            if not r["n_boxes"]:
                continue
            w, h = r["size"]
            for c, cx, cy, bw, bh in r["boxes"]:
                side = (bw * w * bh * h) ** 0.5
                if side < min_side_px:
                    continue           # below the refiner's ability to author anything
                self.items.append((r["file"], c, cx, cy, bw, bh, w, h))
        if not self.items:
            raise RuntimeError(f"no usable defect patches in split '{split}'")
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.items)

    def _image(self, rel: str) -> np.ndarray:
        if rel not in self._cache:
            if len(self._cache) > 48:
                self._cache.pop(next(iter(self._cache)))
            self._cache[rel] = cv2.imread(str(CURATED / "images" / rel), cv2.IMREAD_COLOR)
        return self._cache[rel]

    def __getitem__(self, idx):
        rel, c, cx, cy, bw, bh, W, H = self.items[idx]
        img = self._image(rel)
        p = self.patch

        # centre the patch on the defect, jittered so it is not always dead centre
        px = int(cx * W) + random.randint(-p // 6, p // 6)
        py = int(cy * H) + random.randint(-p // 6, p // 6)
        x0 = int(np.clip(px - p // 2, 0, max(0, img.shape[1] - p)))
        y0 = int(np.clip(py - p // 2, 0, max(0, img.shape[0] - p)))
        real = img[y0:y0 + p, x0:x0 + p]
        if real.shape[:2] != (p, p):
            real = cv2.copyMakeBorder(real, 0, p - real.shape[0], 0, p - real.shape[1],
                                      cv2.BORDER_REFLECT)

        # defect position within the patch
        dcx = (cx * W - x0) / p
        dcy = (cy * H - y0) / p
        side_px = float((bw * W * bh * H) ** 0.5)
        severity = float(np.random.uniform(0.45, 0.95))

        # a clean canvas: the real patch with the defect region inpainted away,
        # so the renderer draws onto plausible background rather than onto the
        # very defect it is meant to be replacing
        params = DefectParams(kind=CLASS_KIND[c], cx=float(np.clip(dcx, 0.05, 0.95)),
                              cy=float(np.clip(dcy, 0.05, 0.95)),
                              size_px=max(4.0, side_px), severity=severity,
                              morphology=random.choice(["circular", "irregular"]))
        res = render(real, [params], seed=random.randint(0, 2 ** 31 - 1))
        mask = (res["mask"] > 0).astype(np.float32)
        if mask.sum() < 4:
            mask = np.zeros((p, p), np.float32)
            mask[p // 2 - 3:p // 2 + 3, p // 2 - 3:p // 2 + 3] = 1.0

        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        onehot = torch.zeros(N_CLASSES)
        onehot[c] = 1.0
        return {
            "rendered": to_t(res["image"]),
            "real": to_t(real),
            "mask": torch.from_numpy(mask)[None],
            "cls": onehot,
            "severity": torch.tensor(severity, dtype=torch.float32),
        }


def train(epochs: int = 30, batch: int = 16, patch: int = 192, lr_g: float = 2e-4,
          lr_d: float = 1e-4, use_fft: bool = True, device: str | None = None,
          lambda_rec: float = 10.0, lambda_fft: float = 1.0,
          lambda_adv: float = 1.0, tag: str = "fft") -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = DefectPatchDataset("train", patch=patch)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

    G = DefectRefiner(n_classes=N_CLASSES).to(device)
    D = RefinerDiscriminator(use_fft=use_fft).to(device)
    gan = GANLoss().to(device) if hasattr(GANLoss(), "to") else GANLoss()

    optG = torch.optim.AdamW(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    scaler_g = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    scaler_d = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    CKPT.mkdir(exist_ok=True)
    print(f"[refiner] {len(ds)} defect patches | patch={patch} batch={batch} "
          f"fft={'on' if use_fft else 'OFF (ablation A1)'} device={device}")

    history = []
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        agg = {"rec": 0.0, "adv": 0.0, "fft": 0.0, "d": 0.0}
        t0 = time.time()

        for b in dl:
            real = b["real"].to(device)
            rendered = b["rendered"].to(device)
            mask = b["mask"].to(device)
            cls = b["cls"].to(device)
            sev = b["severity"].to(device)

            # ---- discriminator ----
            optD.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                with torch.no_grad():
                    fake = G(rendered, mask, cls, sev)
                d_real = D(real)
                d_fake = D(fake.detach())
                loss_d = 0.5 * sum(
                    torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
                    for r, f in zip(d_real, d_fake))
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optD); scaler_d.update()

            # ---- generator ----
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                fake = G(rendered, mask, cls, sev)
                d_out = D(fake)
                loss_adv = -sum(torch.mean(o) for o in d_out)
                # THE term the old pipeline was missing: supervision INSIDE the
                # mask, against a real defect
                loss_rec = masked_l1(fake, real, mask)
                loss_fft = spectral_l1(fake, real) if use_fft else torch.zeros((), device=device)
                loss_g = (lambda_rec * loss_rec + lambda_adv * loss_adv
                          + lambda_fft * loss_fft)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optG); scaler_g.update()

            agg["rec"] += float(loss_rec); agg["adv"] += float(loss_adv)
            agg["fft"] += float(loss_fft); agg["d"] += float(loss_d)

        n = len(dl)
        row = {k: round(v / n, 4) for k, v in agg.items()}
        row.update(epoch=epoch, seconds=round(time.time() - t0, 1))
        history.append(row)
        print(f"  epoch {epoch:>3}/{epochs}  rec={row['rec']:.4f}  adv={row['adv']:.3f}  "
              f"fft={row['fft']:.4f}  d={row['d']:.3f}  ({row['seconds']}s)")

    out = CKPT / f"refiner_{tag}.pth"
    torch.save({"model": G.state_dict(), "use_fft": use_fft,
                "n_classes": N_CLASSES, "patch": patch}, out)
    (CKPT / f"refiner_{tag}_history.json").write_text(
        json.dumps(history, indent=1), encoding="utf-8")
    print(f"[refiner] saved -> {out}")
    return {"checkpoint": str(out), "history": history}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--no-fft", action="store_true",
                    help="ablation A1 / H2: drop the Fourier discriminator branch")
    a = ap.parse_args()
    train(epochs=a.epochs, batch=a.batch, patch=a.patch,
          use_fft=not a.no_fft, tag="nofft" if a.no_fft else "fft")
