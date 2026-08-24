"""
Stage 9 - LOCKED test-set evaluation. Run this once, manually, at the end.

Every other script in this project reads train/val. This is the only one that
reads data/splits/test.json, and it requires an explicit --i-am-sure flag plus
typing the exact confirmation phrase, because there is no undo for "the test
set has now informed a decision" - once you've looked, every subsequent choice
you make is contaminated by having seen it.

Usage (after you've settled on final checkpoints via val):

    py -3.10 eval/final_eval.py --checkpoints exp1=path1.pt exp2=path2.pt \
        --i-am-sure --confirm "I am done tuning"

Writes outputs/FINAL_TEST_RESULTS.md - the one table that goes in the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from eval.detection import evaluate as score_detections
from eval.tiny_defect import load_bins

OUT = ROOT / "outputs"
CURATED = ROOT / "data" / "curated" / "images"
CONFIRM_PHRASE = "I am done tuning"


def _test_records(known_classes=(0, 1)):
    """Load the locked test split, remapped to the same class ids used in training."""
    known = set(known_classes)
    remap = {c: i for i, c in enumerate(sorted(known))}
    recs = json.loads((ROOT / "data" / "splits" / "test.json")
                      .read_text(encoding="utf-8"))["records"]
    out = []
    for r in recs:
        boxes = [[remap[b[0]], *b[1:]] for b in r["boxes"] if b[0] in known]
        if r["n_boxes"] > 0 and not boxes:
            continue          # unknown-only image under this vocabulary
        out.append({**r, "boxes": boxes})
    return out


def run_one_checkpoint(name: str, checkpoint: str, known_classes=(0, 1),
                       conf: float = 0.001) -> dict:
    from ultralytics import YOLO

    net = YOLO(checkpoint)
    records = _test_records(known_classes)
    samples = []
    t0 = time.time()
    for r in records:
        img = cv2.imread(str(CURATED / r["file"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = net.predict(img, conf=conf, verbose=False)[0]
        pred = []
        scores = []
        if len(res.boxes):
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for (cx, cy, bw, bh), c, s in zip(xywhn, cls, cf):
                pred.append([int(c), float(cx), float(cy), float(bw), float(bh)])
                scores.append(float(s))
        samples.append({"gt": r["boxes"], "pred": pred,
                        "scores": np.array(scores, np.float32), "wh": (w, h)})
    elapsed = time.time() - t0

    metrics = score_detections(samples)
    metrics.update({"checkpoint_name": name, "checkpoint_path": checkpoint,
                    "images": len(samples), "seconds": round(elapsed, 1),
                    "known_classes": sorted(known_classes)})
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path.pt pairs, e.g. real_only=experiments/.../best.pt")
    ap.add_argument("--known-classes", default="0,1")
    ap.add_argument("--i-am-sure", action="store_true")
    ap.add_argument("--confirm", default="")
    a = ap.parse_args()

    if not a.i_am_sure or a.confirm != CONFIRM_PHRASE:
        print("REFUSING TO RUN.")
        print(f"This reads the LOCKED test split. Pass --i-am-sure and "
              f"--confirm \"{CONFIRM_PHRASE}\" to proceed.")
        print("Make sure every hyperparameter, checkpoint, and threshold has "
              "already been chosen using val only.")
        return 1

    known = tuple(int(x) for x in a.known_classes.split(","))
    pairs = [c.split("=", 1) for c in a.checkpoints]

    print("=" * 70)
    print("  FINAL LOCKED TEST-SET EVALUATION")
    print("  (this is the only time this project reads data/splits/test.json)")
    print("=" * 70)

    results = {}
    for name, path in pairs:
        print(f"\n[final] {name}  <-  {path}")
        results[name] = run_one_checkpoint(name, path, known_classes=known)
        m = results[name]
        print(f"    mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50_95']:.4f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"({m['images']} imgs, {m['seconds']:.1f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_test_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")

    lines = ["# FINAL TEST RESULTS (locked split, read once)", "",
            "| checkpoint | mAP50 | mAP50-95 | P | R |", "|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| `{name}` | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} "
                     f"| {m['precision']:.3f} | {m['recall']:.3f} |")
    lines += ["", "## Per scale bin", "",
             "| checkpoint | " + " | ".join(b["name"] for b in load_bins()[1]) + " |",
             "|---|" + "---|" * len(load_bins()[1])]
    for name, m in results.items():
        cells = [f"{v['recall']:.3f}" for v in m["per_bin_at50"].values()]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    (OUT / "FINAL_TEST_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[final] wrote {OUT/'FINAL_TEST_RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
