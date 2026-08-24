"""
Build a refined synthetic pool: renderer geometry + learned FESEM texture.

The renderer emits exact masks and boxes; the refiner repaints only inside those
masks. Labels are copied through unchanged because the composite guarantees the
geometry is untouched - which is the whole reason the two stages are separate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth.refiner import DefectRefiner

SYNTH = ROOT / "data" / "synthetic"
CKPT = ROOT / "checkpoints"
N_CLASSES = 2


def refine_pool(src_pool: str = "controlled", dst_pool: str = "refined",
                checkpoint: str = "refiner_fft.pth", device: str | None = None,
                batch: int = 4) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = SYNTH / src_pool, SYNTH / dst_pool
    assert (src / "manifest.jsonl").exists(), f"no source pool at {src}"

    ck = torch.load(CKPT / checkpoint, map_location=device, weights_only=False)
    G = DefectRefiner(n_classes=ck.get("n_classes", N_CLASSES)).to(device).eval()
    G.load_state_dict(ck["model"])

    for sub in ("images", "labels", "masks"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    entries = [json.loads(l) for l in
               (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    n = 0
    pbar = tqdm(entries, desc=f"refine[{dst_pool}]", unit="img", dynamic_ncols=True)
    with torch.no_grad():
        for e in pbar:
            stem = e["stem"]
            img_p = src / "images" / f"{stem}.jpg"
            msk_p = src / "masks" / f"{stem}.png"
            lbl_p = src / "labels" / f"{stem}.txt"
            if not (img_p.exists() and msk_p.exists()):
                continue

            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)

            t_img = torch.from_numpy(img.astype(np.float32) / 255.0)\
                         .permute(2, 0, 1)[None].to(device)
            t_msk = torch.from_numpy((msk > 0).astype(np.float32))[None, None].to(device)

            # dominant class and mean severity for this image, from the manifest
            params = e.get("params", [])
            kinds = [p["kind"] for p in params]
            cls_idx = 0 if kinds.count("pbi2") > kinds.count("pinhole") else 1
            sev = float(np.mean([p["severity"] for p in params])) if params else 0.6

            onehot = torch.zeros(1, N_CLASSES, device=device)
            onehot[0, cls_idx] = 1.0
            out = G(t_img, t_msk, onehot, torch.tensor([sev], device=device))

            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), arr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(msk_p, dst / "masks" / f"{stem}.png")
            if lbl_p.exists():
                shutil.copy2(lbl_p, dst / "labels" / f"{stem}.txt")
            n += 1
            pbar.set_postfix({"refined": n})

    shutil.copy2(src / "manifest.jsonl", dst / "manifest.jsonl")
    summary = {"pool": dst_pool, "source_pool": src_pool, "images": n,
               "checkpoint": checkpoint, "use_fft": ck.get("use_fft"),
               "note": "geometry and labels identical to source pool; texture refined"}
    (dst / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[refine] {src_pool} -> {dst_pool}: {n} images refined "
          f"(fft={ck.get('use_fft')})")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="controlled")
    ap.add_argument("--dst", default="refined")
    ap.add_argument("--checkpoint", default="refiner_fft.pth")
    a = ap.parse_args()
    refine_pool(src_pool=a.src, dst_pool=a.dst, checkpoint=a.checkpoint)
