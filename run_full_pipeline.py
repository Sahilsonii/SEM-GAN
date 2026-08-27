#!/usr/bin/env python
"""
run_full_pipeline.py - single entry point for the whole study.

    py -3.10 run_full_pipeline.py --check      # environment + what is runnable
    py -3.10 run_full_pipeline.py --stage all  # everything currently implemented
    py -3.10 run_full_pipeline.py --stage all --resume  # skip finished stages, continue
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
    # best-epoch rec, not final-epoch: GAN training on ~2600 patches can and
    # does destabilise in the back half, so the last epoch is not necessarily
    # the one worth using - see train_refiner.train's checkpoint selection.
    print(f"[stage5] fft   best epoch={fft['best_epoch']}/{args.refiner_epochs} "
          f"rec={fft['best_rec']:.4f}")
    print(f"[stage5] nofft best epoch={nofft['best_epoch']}/{args.refiner_epochs} "
          f"rec={nofft['best_rec']:.4f}")


def stage6_quality(args):
    """Repaint the renderer pool's geometry with the two refiner checkpoints,
    producing data/synthetic/refined and refined_nofft. Labels are copied
    through unchanged - the refiner cannot move a defect or change its extent,
    only what it looks like."""
    from synth.apply_refiner import refine_pool
    refine_pool("controlled", "refined", "refiner_fft_best.pth")
    refine_pool("controlled", "refined_nofft", "refiner_nofft_best.pth")


def stage7_detector(args):
    """E-A (real only) and E-D (real + synthetic), same budget, same seed.
    --refined uses the GAN-refined pool from stage 6 instead of raw renderer
    output. --ratios runs the scaling ladder (one seed) instead of the matrix.
    --target-steps caps wall-clock when the synthetic pool is large - see the
    docstring on train_detector.train for why a fixed epoch count does not
    scale sanely with a 5000/class pool."""
    from train_detector import train
    pool = args.pool or ("refined" if args.refined else "controlled")

    if args.ratios:
        rows = []
        for ratio in (float(r) for r in args.ratios.split(",")):
            tag = "" if pool in ("refined", "controlled") else f"_{pool}"
            rows.append(train(regime=f"scale_{int(ratio*100):03d}{tag}", seed=args.seed,
                              epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                              p2=args.p2, synth_pool=pool, synth_ratio=ratio,
                              target_steps=args.target_steps, resume=args.resume))
    else:
        rows = [train(regime="real_only", seed=args.seed, epochs=args.epochs,
                      imgsz=args.imgsz, batch=args.batch, p2=args.p2,
                      target_steps=args.target_steps, resume=args.resume)]
        if not args.skip_synth_regime:
            rows.append(train(regime=f"real_plus_{pool}", seed=args.seed,
                              epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                              p2=args.p2, synth_pool=pool, target_steps=args.target_steps,
                              resume=args.resume))
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
                  batch=args.batch, known_classes=(1,), resume=args.resume)
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


def _best_ckpt(args) -> str:
    """Checkpoint the diagnostic stages operate on."""
    if args.diag_checkpoint:
        return args.diag_checkpoint
    pool = args.pool or ("refined" if args.refined else "controlled")
    cand = [EXPERIMENTS / f"scale_005_yolo11s_seed{args.seed}" / "run" / "weights" / "best.pt",
            EXPERIMENTS / f"real_plus_{pool}_yolo11s_seed{args.seed}" / "run" / "weights" / "best.pt",
            EXPERIMENTS / f"real_only_yolo11s_seed{args.seed}" / "run" / "weights" / "best.pt"]
    for c in cand:
        if c.exists():
            return str(c)
    raise RuntimeError("no trained checkpoint found - run stage 7, or pass "
                       "--diag-checkpoint")


def stage10_domaingap(args):
    """N2: real-vs-synthetic gap at 4 levels, per pool. Persists the result so
    the report can read it - measure() alone only prints."""
    from eval.domain_gap import measure
    pools = (args.gap_pools or "controlled,refined,refined_nofft").split(",")
    res = {p.strip(): measure(p.strip(), n_real=args.gap_n, n_synth=args.gap_n,
                              skip_l4=args.skip_l4) for p in pools}
    (OUT / "domain_gap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[gap] wrote {OUT / 'domain_gap.json'}")


def stage11_counterfactual(args):
    """N4: monotonic response to severity on a fixed background."""
    from eval.counterfactual import probe
    probe(_best_ckpt(args), conf=max(args.diag_conf, 0.10))


def stage12_robustness(args):
    """Section 14: does confidence fall when the image degrades?"""
    from eval.robustness import sweep
    sweep(_best_ckpt(args), conf=args.diag_conf)


def stage13_failures(args):
    """Section 15: pre-declared failure taxonomy + representative crops."""
    from eval.failure_analysis import analyse
    analyse(_best_ckpt(args), conf=max(args.diag_conf, 0.10))


def stage14_explain(args):
    """Section 12: attribution faithfulness, not heatmap pictures."""
    from eval.explain import evaluate
    evaluate(_best_ckpt(args), n_images=args.explain_images,
             patch=96, stride=64, conf=args.diag_conf)


def stage15_interpret(args):
    """Section 13: image-derived morphology indices (NOT measurements)."""
    from interpret.run_interpretation import run
    run(_best_ckpt(args), split="val", conf=max(args.diag_conf, 0.10),
        use_ground_truth=args.interpret_gt)


def stage16_report(args):
    """Regenerates the auto-table only.

    outputs/RESULTS.md is HAND-WRITTEN and holds the narrative, caveats and the
    analyses from stages 10-15 that make_report does not know about. Writing it
    from make_report would silently discard all of that, so the generated table
    goes to a separate file and RESULTS.md is left alone.
    """
    import make_report
    make_report.OUT = OUT
    text = make_report.build()
    (OUT / "auto_table.md").write_text(text, encoding="utf-8")
    print(f"[report] auto table -> {OUT / 'auto_table.md'} "
          f"(RESULTS.md is hand-written and was not touched)")


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
    Stage(10, "domaingap", "Domain gap real vs synthetic, 4 levels (N2)",
          ("cv2",), stage10_domaingap),
    Stage(11, "counterfact", "Counterfactual severity monotonicity (N4)",
          ("ultralytics",), stage11_counterfactual),
    Stage(12, "robustness", "Perturbation sweep: does confidence track degradation",
          ("ultralytics",), stage12_robustness),
    Stage(13, "failures",  "Pre-declared failure taxonomy + crops (section 15)",
          ("ultralytics",), stage13_failures),
    Stage(14, "explain",   "Attribution faithfulness (section 12)",
          ("ultralytics",), stage14_explain),
    Stage(15, "interpret", "Image-derived morphology indices (section 13)",
          ("cv2",), stage15_interpret),
    Stage(16, "report",    "Assemble outputs/RESULTS.md",
          (), stage16_report),
]
FIRST_UNIMPLEMENTED = 5   # kept for reference; superseded by IMPLEMENTED below
IMPLEMENTED = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16}
# stage 9 is intentionally excluded: it always raises NotImplementedError here by
# design and must be run by hand via eval/final_eval.py with its confirmation gate
BY_KEY = {s.key: s for s in STAGES}
BY_NUM = {str(s.num): s for s in STAGES}
CKPT = ROOT / "checkpoints"
SYNTH = ROOT / "data" / "synthetic"
RAW_SNAPSHOT = ROOT / "data" / "raw_snapshot"
CURATED = ROOT / "data" / "curated"
SPLITS = ROOT / "data" / "splits"


def _detector_exp_ids(args) -> list[str]:
    """Experiment dirs stage 7 must finish for the current CLI flags."""
    pool = args.pool or ("refined" if args.refined else "controlled")
    suffix = "-p2" if args.p2 else ""
    tail = f"_yolo11s{suffix}_seed{args.seed}"
    if args.ratios:
        return [f"scale_{int(float(r) * 100):03d}{tail}"
                for r in args.ratios.split(",")]
    ids = [f"real_only{tail}"]
    if not args.skip_synth_regime:
        ids.append(f"real_plus_{pool}{tail}")
    return ids


def stage_is_done(s: Stage, args) -> tuple[bool, str]:
    """True when this stage's expected outputs for *args* already exist."""
    if s.num == 0:
        if (RAW_SNAPSHOT / "SNAPSHOT.json").exists():
            return True, "data/raw_snapshot/SNAPSHOT.json"
        return False, "snapshot manifest missing"

    if s.num == 1:
        need = (CURATED / "curated.json", SPLITS / "splits_manifest.json",
                SPLITS / "train.json", SPLITS / "val.json", SPLITS / "test.json")
        if not all(p.exists() for p in need):
            return False, "curated.json or split files missing"
        manifest = json.loads((SPLITS / "splits_manifest.json")
                              .read_text(encoding="utf-8"))
        if manifest.get("seed") != args.seed:
            return False, f"splits seed={manifest.get('seed')} != --seed {args.seed}"
        return True, f"splits (seed={args.seed})"

    if s.num == 2:
        p = OUT / "defect_scale_profile.json"
        return (True, str(p.relative_to(ROOT))) if p.exists() else (False, "defect_scale_profile.json missing")

    if s.num == 3:
        ladder = SYNTH / "counterfactual" / "ladder.json"
        summary_path = SYNTH / "controlled" / "summary.json"
        if not ladder.exists():
            return False, "counterfactual/ladder.json missing"
        if not summary_path.exists():
            return False, "controlled/summary.json missing"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if args.per_class:
            want = args.per_class * 2
            got = summary.get("total_images", 0)
            if got < want:
                return False, f"controlled pool {got}/{want} images (--per-class {args.per_class})"
        else:
            want = args.n_synth
            got = summary.get("images") or summary.get("total_images", 0)
            if got < want:
                return False, f"controlled pool {got}/{want} images (--n-synth {want})"
        return True, f"controlled ({got} imgs) + counterfactual ladder"

    if s.num == 4:
        if args.sweep:
            p = OUT / "microdefectcv_sweep_val.json"
        else:
            p = OUT / "microdefectcv_baseline_val.json"
        return ((True, str(p.relative_to(ROOT))) if p.exists()
                else (False, f"{p.name} missing"))

    if s.num == 5:
        fft = CKPT / "refiner_fft_best.pth"
        nofft = CKPT / "refiner_nofft_best.pth"
        if fft.exists() and nofft.exists():
            return True, "refiner_fft_best.pth + refiner_nofft_best.pth"
        missing = [p.name for p in (fft, nofft) if not p.exists()]
        return False, ", ".join(missing) + " missing"

    if s.num == 6:
        refined = SYNTH / "refined" / "summary.json"
        nofft = SYNTH / "refined_nofft" / "summary.json"
        if refined.exists() and nofft.exists():
            return True, "refined + refined_nofft pools"
        missing = [p.parent.name for p in (refined, nofft) if not p.exists()]
        return False, f"summary.json missing in {', '.join(missing)}"

    if s.num == 7:
        missing = [e for e in _detector_exp_ids(args)
                   if not (EXPERIMENTS / e / "metrics.json").exists()]
        if missing:
            return False, "incomplete: " + ", ".join(missing)
        return True, f"detector matrix ({len(_detector_exp_ids(args))} experiments)"

    if s.num == 8:
        need = OUT / "open_set_val.json", OUT / "calibration_val.json"
        if all(p.exists() for p in need):
            return True, "open_set_val.json + calibration_val.json"
        missing = [p.name for p in need if not p.exists()]
        return False, ", ".join(missing) + " missing"

    # stage 9 is manual-only; never auto-skip via --resume
    return False, "not auto-runnable"


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
    ap.add_argument("--force", action="store_true",
                    help="re-run requested stages even if --resume would skip them; "
                         "also re-copy the snapshot (stage 0)")
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
    ap.add_argument("--diag-checkpoint", default=None,
                    help="checkpoint for the diagnostic stages 11-15")
    ap.add_argument("--diag-conf", type=float, default=0.05)
    ap.add_argument("--gap-pools", default=None)
    ap.add_argument("--gap-n", type=int, default=40)
    ap.add_argument("--skip-l4", action="store_true",
                    help="stage 10: skip the DINOv2 feature level")
    ap.add_argument("--explain-images", type=int, default=4)
    ap.add_argument("--interpret-gt", action="store_true",
                    help="stage 15: interpret expert boxes instead of detections")
    ap.add_argument("--pool", default=None,
                    help="explicit synthetic pool name, e.g. refined_nofft for the H2 ablation")
    ap.add_argument("--refined", action="store_true",
                    help="stage 7: use the GAN-refined pool (stage 6) instead of raw renderer output")
    ap.add_argument("--ratios", default=None,
                    help="stage 7: comma-separated synthetic ratios for the scaling ladder, "
                         "e.g. 0.25,0.5,1.0,2.0 - runs the ladder instead of E-A/E-D")
    ap.add_argument("--target-steps", type=int, default=None,
                    help="stage 7: cap wall-clock by deriving epochs from a step budget - "
                         "see train_detector.train docstring; important once the synthetic "
                         "pool is large (e.g. after --per-class 5000)")
    ap.add_argument("--resume", action="store_true",
                    help="skip pipeline stages whose outputs already exist; "
                         "within stage 7/8 resume detector training from last.pt "
                         "and skip finished experiments (metrics.json)")
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
    ran, skipped, skipped_done = [], [], []
    for s in chosen:
        if args.resume and not args.force and s.num in IMPLEMENTED:
            done, marker = stage_is_done(s, args)
            if done:
                print("\n" + "-" * 74)
                print(f"[{s.num}] {s.title}")
                print("-" * 74)
                print(f"    SKIP (already done: {marker})")
                skipped_done.append(s)
                continue
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
    if skipped_done:
        print(f"  skipped   {len(skipped_done)} (already done): "
              f"{', '.join(s.key for s in skipped_done)}")
    if skipped:
        print(f"  not run   {len(skipped)}: {', '.join(s.key for s in skipped)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
