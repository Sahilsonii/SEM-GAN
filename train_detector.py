"""
Detector training for one regime x seed.

YOLO11s at 640 px on a 4 GB card. The one architectural choice that matters here
is the P2 head: the default P3/P4/P5 pyramid starts at stride 8, and ~21% of
train boxes are smaller than 8 px at 640 - i.e. smaller than a single cell of
the finest default detection level. `--p2` swaps in the yolo11-p2 topology, and
running with and without it is ablation A4.

Everything lands in experiments/<exp_id>/ with its resolved config, so a run can
be reproduced from the directory alone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

EXPERIMENTS = ROOT / "experiments"
MASTER_CSV = ROOT / "outputs" / "master_results.csv"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def train(regime: str = "real_only", seed: int = 0, epochs: int = 100,
          imgsz: int = 640, batch: int = 8, model: str = "yolo11s",
          p2: bool = False, synth_pool: str | None = None,
          synth_ratio: float = 1.0, device: str = "0",
          patience: int = 30) -> dict:
    from ultralytics import YOLO

    import yolo_export

    data_yaml = yolo_export.build(regime=regime, synth_pool=synth_pool,
                                  synth_ratio=synth_ratio)

    exp_id = f"{regime}_{model}{'-p2' if p2 else ''}_seed{seed}"
    exp_dir = EXPERIMENTS / exp_id
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # yolo11s.pt carries COCO-pretrained weights; the -p2 variant has no
    # published checkpoint, so it trains from the YAML topology instead.
    weights = f"{model}-p2.yaml" if p2 else f"{model}.pt"

    config = {
        "exp_id": exp_id, "regime": regime, "seed": seed, "epochs": epochs,
        "imgsz": imgsz, "batch": batch, "model": model, "p2_head": p2,
        "weights": weights, "synth_pool": synth_pool, "synth_ratio": synth_ratio,
        "device": device, "git_sha": git_sha(),
        "data_yaml": str(data_yaml),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    print(f"[train] {exp_id}  weights={weights}  imgsz={imgsz} batch={batch} "
          f"epochs={epochs}")

    t0 = time.time()
    net = YOLO(weights)
    net.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        device=device,
        project=str(exp_dir),
        name="run",
        exist_ok=True,
        patience=patience,
        amp=True,               # 4 GB card - mixed precision is not optional
        workers=2,
        val=True,
        plots=False,
        # small-object friendly augmentation; heavy scale jitter destroys 6 px defects
        scale=0.3,
        mosaic=0.5,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        translate=0.1,
        erasing=0.0,
    )
    elapsed = time.time() - t0

    metrics = net.val(data=str(data_yaml), split="val", imgsz=imgsz,
                      batch=batch, device=device, plots=False)
    box = metrics.box
    result = {
        **config,
        "train_seconds": round(elapsed, 1),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "params_M": round(sum(p.numel() for p in net.model.parameters()) / 1e6, 2),
    }
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    _append_master(result)
    print(f"[train] {exp_id}: mAP50={result['mAP50']:.4f} "
          f"mAP50-95={result['mAP50_95']:.4f} P={result['precision']:.3f} "
          f"R={result['recall']:.3f}  ({elapsed/60:.1f} min)")
    return result


def _append_master(row: dict) -> None:
    import csv
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["exp_id", "regime", "seed", "model", "p2_head", "synth_pool",
              "synth_ratio", "epochs", "imgsz", "batch", "mAP50", "mAP50_95",
              "precision", "recall", "params_M", "train_seconds", "git_sha"]
    exists = MASTER_CSV.exists()
    with MASTER_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="real_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="yolo11s")
    ap.add_argument("--p2", action="store_true", help="enable the stride-4 head")
    ap.add_argument("--synth-pool", default=None)
    ap.add_argument("--synth-ratio", type=float, default=1.0)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    train(regime=a.regime, seed=a.seed, epochs=a.epochs, imgsz=a.imgsz,
          batch=a.batch, model=a.model, p2=a.p2, synth_pool=a.synth_pool,
          synth_ratio=a.synth_ratio, device=a.device)
