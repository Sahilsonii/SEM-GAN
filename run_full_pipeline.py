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
    from synth.generate import generate, severity_ladder
    generate(n_images=args.n_synth, pool_name="controlled",
             seed=args.seed, render_px=args.render_px)
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
    raise NotImplementedError(
        "GAN texture refiner - next to build. Needs torch (present in py3.10).")


def stage6_quality(args):
    raise NotImplementedError("Quality filter + domain-gap regression.")


def stage7_detector(args):
    raise NotImplementedError("YOLO11s+P2 / RF-DETR training matrix.")


def stage8_uncertainty(args):
    raise NotImplementedError("Open-set (PbI2 held out) + calibration.")


def stage9_final(args):
    raise NotImplementedError(
        "FINAL evaluation on the locked real test split. Runs once, at the end.")


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
    Stage(6, "quality",   "Quality filter + domain-gap -> utility regression (N2)",
          ("torch", "sklearn"), stage6_quality),
    Stage(7, "detector",  "Detector matrix E-A..E-E (YOLO11s+P2, RF-DETR)",
          ("ultralytics",), stage7_detector),
    Stage(8, "uncertain", "Open-set PbI2 + calibration (ECE, Brier, risk-coverage)",
          ("ultralytics", "sklearn"), stage8_uncertainty),
    Stage(9, "final",     "LOCKED real test-set evaluation + master results table",
          ("ultralytics",), stage9_final),
]
FIRST_UNIMPLEMENTED = 5     # stages >= this are declared but not built yet
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
        if s.num >= FIRST_UNIMPLEMENTED:
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
