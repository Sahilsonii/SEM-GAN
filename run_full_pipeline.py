#!/usr/bin/env python
"""
run_full_pipeline.py - single entry point for the whole study.

    py -3.10 run_full_pipeline.py --check      # environment + what is runnable
    py -3.10 run_full_pipeline.py --stage all  # everything currently implemented
    py -3.10 run_full_pipeline.py --stage 2    # one stage

Stages run in dependency order and each refuses to start unless its inputs
exist, so a half-finished run cannot silently produce numbers. Stages that need
packages which are not installed yet are reported as BLOCKED with the exact
install command, rather than crashing halfway.

Design rule carried through every stage: the test split is written once by
stage 1 and is not read again until stage 9. Nothing between those points may
touch it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

PY = sys.executable
EXPERIMENTS = ROOT / "experiments"
OUT = ROOT / "outputs"


# --------------------------------------------------------------------------
# stage registry
# --------------------------------------------------------------------------
class Stage:
    def __init__(self, num, key, title, needs=(), fn=None, note=""):
        self.num, self.key, self.title = num, key, title
        self.needs, self.fn, self.note = needs, fn, note


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _missing(mods) -> list[str]:
    return [m for m in mods if not _have(m)]


# import name -> pip name, where they differ
PIP_NAME = {"sklearn": "scikit-learn", "cv2": "opencv-python", "skimage": "scikit-image"}


def _pip_cmd(mods) -> str:
    return "py -3.10 -m pip install " + " ".join(PIP_NAME.get(m, m) for m in mods)


# ---- stage bodies --------------------------------------------------------

def stage0_snapshot(args):
    import snapshot
    snapshot.create(force=args.force)
    snapshot.verify()


def stage1_dataset(args):
    import build_dataset
    import splits
    build_dataset.curate(keep_3class=args.keep_3class)
    splits.build(seed=args.seed)


def stage2_bins(args):
    from eval.tiny_defect import profile_split
    out = {}
    for sp in ("train", "val", "test"):
        p = profile_split(sp)
        out[sp] = p
        parts = " ".join(f"{k}={v}({p['share'][k]*100:.1f}%)" for k, v in p["counts"].items())
        print(f"  {sp:<5} n={p['total_boxes']:<5} {parts}")
    (ROOT / "outputs" / "defect_scale_profile.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")


def stage3_synth(args):
    """Renderer pool. --per-class N switches to balanced N-per-class generation
    (e.g. --per-class 5000 -> 10,000 images, both classes); otherwise --n-synth
    total, pinhole-only. Measured throughput ~0.61s/image, so 5000/class is
    ~100 minutes of rendering alone - see the README section on bulk scale."""
    from synth.generate import generate, generate_balanced, severity_ladder
    if args.per_class:
        generate_balanced(per_class=args.per_class, render_px=args.render_px,
                          seed=args.seed, pool_name="controlled")
    else:
        generate(n_images=args.n_synth, pool_name="controlled",
                seed=args.seed, render_px=args.render_px, pbi2_fraction=0.0)
    severity_ladder(render_px=args.render_px)


def stage4_microdefectcv(args):
    """R1: tune on val, then report. The test split stays locked until stage 9."""
    from eval.microdefectcv_baseline import run_baseline, sweep
    if args.sweep:
        best = sweep(split="val", limit=args.sweep_limit)
        run_baseline(split="val", min_area=best["min_area"])
    else:
        run_baseline(split="val", min_area=args.mdcv_min_area,
                     sensitivity=args.mdcv_sensitivity)


def stage5_refiner(args):
    """Train the conditional GAN texture refiner: FFT-branch ON (main) and
    OFF (ablation A1/H2), same budget, so the two checkpoints are comparable.
    Operates on 192px patches from real defect crops - independent of how big
    the renderer pool from stage 3 is."""
    from train_refiner import train as train_refiner
    fft = train_refiner(epochs=args.refiner_epochs, batch=args.refiner_batch,
                        use_fft=True, tag="fft")
    nofft = train_refiner(epochs=args.refiner_epochs, batch=args.refiner_batch,
                          use_fft=False, tag="nofft")
    print(f"[stage5] fft   final rec={fft['history'][-1]['rec']:.4f}")
    print(f"[stage5] nofft final rec={nofft['history'][-1]['rec']:.4f}")


def stage6_quality(args):
    """Repaint the renderer pool's geometry with the two refiner checkpoints,
    producing data/synthetic/refined and refined_nofft. Labels are copied
    through unchanged - the refiner cannot move a defect or change its extent,
    only what it looks like."""
    from synth.apply_refiner import refine_pool
    refine_pool("controlled", "refined", "refiner_fft.pth")
    refine_pool("controlled", "refined_nofft", "refiner_nofft.pth")


def stage7_detector(args):
    """E-A (real only) and E-D (real + synthetic), same budget, same seed.
    --refined uses the GAN-refined pool from stage 6 instead of raw renderer
    output. --ratios runs the scaling ladder (one seed) instead of the matrix.
    --target-steps caps wall-clock when the synthetic pool is large - see the
    docstring on train_detector.train for why a fixed epoch count does not
    scale sanely with a 5000/class pool."""
    from train_detector import train
    pool = "refined" if args.refined else "controlled"

    if args.ratios:
        rows = []
        for ratio in (float(r) for r in args.ratios.split(",")):
            rows.append(train(regime=f"scale_{int(ratio*100):03d}", seed=args.seed,
                              epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                              p2=args.p2, synth_pool=pool, synth_ratio=ratio,
                              target_steps=args.target_steps))
    else:
        rows = [train(regime="real_only", seed=args.seed, epochs=args.epochs,
                      imgsz=args.imgsz, batch=args.batch, p2=args.p2,
                      target_steps=args.target_steps)]
        if not args.skip_synth_regime:
            rows.append(train(regime=f"real_plus_{pool}", seed=args.seed,
                              epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                              p2=args.p2, synth_pool=pool, target_steps=args.target_steps))
    print()
    print("  regime               mAP50    mAP50-95   P       R")
    for r in rows:
        print(f"  {r['regime']:<20} {r['mAP50']:<8.4f} {r['mAP50_95']:<10.4f} "
              f"{r['precision']:<7.3f} {r['recall']:.3f}")


def stage8_uncertainty(args):
    """Open-set (PbI2 held out) + calibration. Trains a pinhole-only detector
    if no checkpoint is supplied via --openset-checkpoint."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from eval.calibration import calibration_report
    from eval.open_set import evaluate as openset_evaluate
    from eval.detection import iou_matrix, xywhn_to_xyxy

    ckpt = args.openset_checkpoint
    if not ckpt:
        from train_detector import train
        r = train(regime="openset_probe", seed=args.seed, epochs=args.epochs,
                  batch=args.batch, known_classes=(1,))
        ckpt = str(EXPERIMENTS / r["exp_id"] / "run" / "weights" / "best.pt")
        print(f"[stage8] trained pinhole-only checkpoint -> {ckpt}")

    openset_evaluate(ckpt, split="val")

    # calibration: per-detection confidence vs IoU-matched correctness, val only
    net = YOLO(ckpt)
    recs = json.loads((ROOT / "data" / "splits" / "val.json").read_text(encoding="utf-8"))
    confs, corrects = [], []
    for r in recs["records"]:
        gt = [b for b in r["boxes"] if b[0] == 1]
        img = cv2.imread(str(ROOT / "data" / "curated" / "images" / r["file"]))
        if img is None:
            continue
        w, h = img.shape[1], img.shape[0]
        res = net.predict(img, conf=0.05, verbose=False)[0]
        if not len(res.boxes) or not gt:
            continue
        gt_xyxy = np.stack([xywhn_to_xyxy(b, w, h) for b in gt])
        for box, cf in zip(res.boxes.xywhn.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            pr_xyxy = xywhn_to_xyxy([1, *box], w, h)[None]
            best = float(iou_matrix(pr_xyxy, gt_xyxy).max())
            confs.append(float(cf)); corrects.append(1.0 if best >= 0.5 else 0.0)

    if confs:
        calibration_report(np.array(confs), np.array(corrects),
                           out_path=OUT / "calibration_val.json")
    else:
        print("[stage8] no detections on val to calibrate against")


def stage9_final(args):
    print("Stage 9 is deliberately NOT auto-run. It reads the locked test split")
    print("exactly once, and must be invoked by hand after every checkpoint,")
    print("hyperparameter, and threshold has been chosen using val only:")
    print()
    print('  py -3.10 eval/final_eval.py --checkpoints name=path.pt ... \\')
    print('      --i-am-sure --confirm "I am done tuning"')
    raise NotImplementedError("run eval/final_eval.py by hand - see message above")


STAGES = [
    Stage(0, "snapshot",  "Vendor + verify immutable dataset snapshot",
          (), stage0_snapshot),
    Stage(1, "dataset",   "Curate corpus + leakage-safe grouped splits (N0)",
          (), stage1_dataset),
    Stage(2, "bins",      "Pre-registered defect-scale profile (N3)",
          (), stage2_bins),
    Stage(3, "synth",     "Parametric synthetic pool + counterfactual ladder (N1, N4)",
          ("cv2", "numpy"), stage3_synth),
    Stage(4, "classical", "MicroDefectCV zero-training baseline (R1)",
          ("microdefectcv",), stage4_microdefectcv),
    Stage(5, "refiner",   "Conditional GAN texture refiner (H2 FFT ablation)",
          ("torch",), stage5_refiner),
    Stage(6, "quality",   "Build refined synthetic pools from stage-5 checkpoints",
          ("torch",), stage6_quality),
    Stage(7, "detector",  "Detector matrix E-A..E-E (YOLO11s+P2, RF-DETR)",
          ("ultralytics",), stage7_detector),
    Stage(8, "uncertain", "Open-set PbI2 + calibration (ECE, Brier, risk-coverage)",
          ("ultralytics", "sklearn"), stage8_uncertainty),
    Stage(9, "final",     "LOCKED real test-set evaluation + master results table",
          ("ultralytics",), stage9_final),
]
FIRST_UNIMPLEMENTED = 5   # kept for reference; superseded by IMPLEMENTED below
IMPLEMENTED = {0, 1, 2, 3, 4, 5, 6, 7, 8}
# stage 9 is intentionally excluded: it always raises NotImplementedError here by
# design and must be run by hand via eval/final_eval.py with its confirmation gate
BY_KEY = {s.key: s for s in STAGES}
BY_NUM = {str(s.num): s for s in STAGES}


# --------------------------------------------------------------------------
def preflight() -> None:
    print("=" * 74)
    print("  ENVIRONMENT")
    print("=" * 74)
    print(f"  python      : {sys.version.split()[0]}  ({PY})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "-"
        vram = (f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB"
                if cuda else "-")
        print(f"  torch       : {torch.__version__}  cuda={cuda}  {dev} {vram}")
    except Exception as exc:
        print(f"  torch       : NOT AVAILABLE ({exc})")

    for mod in ("cv2", "numpy", "ultralytics", "microdefectcv", "sklearn",
                "pandas", "skimage", "timm", "pytest"):
        spec = importlib.util.find_spec(mod)
        print(f"  {mod:<12}: {'ok' if spec else 'MISSING'}")

    print("=" * 74)
    print("  STAGES")
    print("=" * 74)
    for s in STAGES:
        miss = _missing(s.needs)
        if s.num not in IMPLEMENTED:
            status = "TODO"
        elif miss:
            status = "BLOCKED"
        else:
            status = "READY"
        print(f"  [{s.num}] {s.key:<10} {status:<8} {s.title}")
        if miss:
            print(f"      missing {', '.join(miss)}  ->  {_pip_cmd(miss)}")
    print("=" * 74)


def run_stage(s: Stage, args) -> bool:
    miss = _missing(s.needs)
    if miss:
        hint = s.note or f"py -3.10 -m pip install {' '.join(miss)}"
        print(f"\n[{s.num}] {s.title}\n    BLOCKED - missing {', '.join(miss)}\n    -> {hint}")
        return False

    print("\n" + "-" * 74)
    print(f"[{s.num}] {s.title}")
    print("-" * 74)
    t0 = time.time()
    try:
        s.fn(args)
    except NotImplementedError as exc:
        print(f"    TODO - {exc}")
        return False
    print(f"    done in {time.time() - t0:.1f}s")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    help="'all', a number (0-9), a key, or a range like 0-3")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-synth", type=int, default=200)
    ap.add_argument("--per-class", type=int, default=None,
                    help="stage 3: balanced N-per-class generation (e.g. 5000), "
                         "instead of --n-synth total")
    ap.add_argument("--render-px", type=int, default=512)
    ap.add_argument("--keep-3class", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-copy the snapshot")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="stage 4: hyperparameter search for MicroDefectCV on val")
    ap.add_argument("--sweep-limit", type=int, default=8)
    ap.add_argument("--mdcv-min-area", type=int, default=120,
                    help="tuned on val: interior optimum of the min_area sweep")
    ap.add_argument("--mdcv-sensitivity", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--p2", action="store_true", help="enable the stride-4 head")
    ap.add_argument("--skip-synth-regime", action="store_true")
    ap.add_argument("--openset-checkpoint", default=None,
                    help="stage 8: reuse an existing pinhole-only .pt instead of training one")
    ap.add_argument("--refiner-epochs", type=int, default=30)
    ap.add_argument("--refiner-batch", type=int, default=16)
    ap.add_argument("--refined", action="store_true",
                    help="stage 7: use the GAN-refined pool (stage 6) instead of raw renderer output")
    ap.add_argument("--ratios", default=None,
                    help="stage 7: comma-separated synthetic ratios for the scaling ladder, "
                         "e.g. 0.25,0.5,1.0,2.0 - runs the ladder instead of E-A/E-D")
    ap.add_argument("--target-steps", type=int, default=None,
                    help="stage 7: cap wall-clock by deriving epochs from a step budget - "
                         "see train_detector.train docstring; important once the synthetic "
                         "pool is large (e.g. after --per-class 5000)")
    args = ap.parse_args()

    if args.check:
        preflight()
        return 0

    # resolve requested stages
    sel = args.stage.strip().lower()
    if sel == "all":
        chosen = STAGES
    elif "-" in sel and sel.replace("-", "").isdigit():
        a, b = (int(x) for x in sel.split("-"))
        chosen = [s for s in STAGES if a <= s.num <= b]
    elif sel in BY_NUM:
        chosen = [BY_NUM[sel]]
    elif sel in BY_KEY:
        chosen = [BY_KEY[sel]]
    else:
        print(f"unknown stage '{args.stage}'. options: all, 0-9, "
              f"{', '.join(BY_KEY)}")
        return 2

    preflight()
    ran, skipped = [], []
    for s in chosen:
        (ran if run_stage(s, args) else skipped).append(s)

    if not args.skip_tests and any(s.num <= 3 for s in ran):
        print("\n" + "-" * 74)
        print("verification")
        print("-" * 74)
        rc = subprocess.call([PY, "-m", "pytest", str(ROOT / "tests"), "-q"])
        if rc != 0:
            print("    TESTS FAILED - do not trust anything above")
            return 1

    print("\n" + "=" * 74)
    print(f"  completed {len(ran)} stage(s): {', '.join(s.key for s in ran) or '-'}")
    if skipped:
        print(f"  not run   {len(skipped)}: {', '.join(s.key for s in skipped)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
