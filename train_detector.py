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
import contextlib
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


def _last_completed_epoch(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    try:
        lines = [ln for ln in results_csv.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("epoch")]
        return int(lines[-1].split(",", 1)[0]) if lines else 0
    except Exception:
        return 0


class _Tee:
    """Mirror writes to console + log file. isatty follows the real console."""

    def __init__(self, console, log_fh):
        self.console = console
        self.log_fh = log_fh

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        self.log_fh.write(data)
        self.log_fh.flush()

    def flush(self):
        self.console.flush()
        self.log_fh.flush()

    def isatty(self):
        return self.console.isatty()

    def fileno(self):
        return self.console.fileno()


@contextlib.contextmanager
def _capture_console(log_path: Path, append: bool = False):
    """Duplicate stdout/stderr to experiments/<exp_id>/console.log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    fh = open(log_path, mode, encoding="utf-8", errors="replace")
    fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    fh.flush()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, fh)
    sys.stderr = _Tee(old_err, fh)
    try:
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        fh.close()


def train(regime: str = "real_only", seed: int = 0, epochs: int = 100,
          imgsz: int = 640, batch: int = 8, model: str = "yolo11s",
          p2: bool = False, synth_pool: str | None = None,
          synth_ratio: float = 1.0, device: str = "0",
          patience: int = 30, known_classes: tuple = (0, 1),
          target_steps: int | None = None, min_epochs: int = 3,
          resume: bool = False) -> dict:
    """
    target_steps caps wall-clock at scale. Convergence tracks gradient STEPS
    seen, not epochs - at 160 real images, 150 epochs is 3000 steps; at 10160
    images (5000/class synthetic), 150 epochs is 190,000 steps, ~12 hours on
    this card for one run alone. When target_steps is set, epochs is derived
    as target_steps / (n_train/batch), floored at min_epochs, so a run's cost
    stays roughly constant as the synthetic pool grows instead of scaling
    linearly with it. Leave target_steps=None for the small real-only regime,
    where a fixed epoch count is what you want.

    resume=True continues from experiments/<exp_id>/run/weights/last.pt
    (Ultralytics checkpoint). Finished runs with metrics.json are returned
    as-is so a pipeline re-run skips completed regimes.

    Console stdout/stderr is mirrored to experiments/<exp_id>/console.log
    for the duration of training (appended on --resume).
    """
    from ultralytics import RTDETR, YOLO

    # Paradigm comparison (plan 7.4) without new dependencies: the installed
    # ultralytics already ships RT-DETR, so the "does a DETR-family model handle
    # sub-stride defects better than an anchor-free CNN?" row costs an argument
    # rather than a pip install that could move torch mid-project.
    Net = RTDETR if model.startswith("rtdetr") else YOLO
    if p2 and Net is RTDETR:
        raise ValueError("--p2 is a YOLO topology; rtdetr has no -p2 variant")

    import yolo_export

    data_yaml = yolo_export.build(regime=regime, synth_pool=synth_pool,
                                  synth_ratio=synth_ratio,
                                  known_classes=known_classes)
    export = json.loads((data_yaml.parent / "export_manifest.json")
                        .read_text(encoding="utf-8"))

    if target_steps is not None:
        n_train = export["counts"]["train"]
        steps_per_epoch = max(1, n_train // batch)
        epochs = max(min_epochs, round(target_steps / steps_per_epoch))
        actual_steps = epochs * steps_per_epoch
        print(f"[train] step-budget: n_train={n_train} steps/epoch={steps_per_epoch} "
              f"-> epochs={epochs} (target_steps={target_steps}, "
              f"actual_steps={actual_steps})")
        # A min_epochs floor silently breaks the whole point of a step budget:
        # with 160 real vs 10,160 real+synthetic images the two regimes differ
        # 63x in steps/epoch, so a floor of 20 gave 6,000 vs 25,400 steps - a
        # 4.2x mismatch dressed up as a matched comparison. Say so loudly
        # rather than letting the number look controlled when it is not.
        if actual_steps > target_steps * 1.25:
            print(f"[train] WARNING: min_epochs={min_epochs} floor raised this run to "
                  f"{actual_steps} steps, {actual_steps/target_steps:.1f}x the "
                  f"{target_steps} budget. It is NOT step-matched against a regime "
                  f"that hit the budget. Lower --min-epochs to compare fairly.")

    exp_id = f"{regime}_{model}{'-p2' if p2 else ''}_seed{seed}"
    exp_dir = EXPERIMENTS / exp_id
    last_pt = exp_dir / "run" / "weights" / "last.pt"
    metrics_path = exp_dir / "metrics.json"
    console_log = exp_dir / "console.log"

    if resume and metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[train] {exp_id}: already finished -> skip "
              f"(mAP50={result.get('mAP50', '?')})")
        return result

    do_resume = resume and last_pt.exists()
    if not do_resume:
        if resume:
            print(f"[train] {exp_id}: --resume set but no last.pt -> fresh start")
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

    with _capture_console(console_log, append=do_resume):
        try:
            print(f"[train] console log -> {console_log}")
            if do_resume:
                # keep exp_dir; Ultralytics resume restores optimizer + epoch from last.pt
                cfg_path = exp_dir / "config.json"
                config = (json.loads(cfg_path.read_text(encoding="utf-8"))
                          if cfg_path.exists() else {
                              "exp_id": exp_id, "regime": regime, "seed": seed,
                              "epochs": epochs, "imgsz": imgsz, "model": model,
                              "p2_head": p2, "synth_pool": synth_pool,
                              "synth_ratio": synth_ratio, "device": device,
                              "git_sha": git_sha(), "target_steps": target_steps,
                              "data_yaml": str(data_yaml),
                          })
                config["resumed"] = True
                config["batch"] = batch  # allow smaller batch after OOM
                cfg_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
                done_epochs = _last_completed_epoch(exp_dir / "run" / "results.csv")
                print(f"[train] {exp_id}  RESUME from {last_pt}  "
                      f"after epoch {done_epochs}  batch={batch}")
                t0 = time.time()
                # 4 GB cards die during Ultralytics' AMP self-check on resume:
                # last.pt is already on GPU, then check_amp loads yolo11n on top.
                # Original run already validated AMP; skip the re-check.
                import torch
                import ultralytics.engine.trainer as _ul_trainer
                import ultralytics.utils.checks as _ul_checks
                torch.cuda.empty_cache()
                _ul_checks.check_amp = lambda model: True
                _ul_trainer.check_amp = lambda model: True
                net = Net(str(last_pt))
                net.train(resume=True, batch=batch, device=device)
            else:
                # yolo11s.pt carries COCO-pretrained weights; the -p2 variant has no
                # published checkpoint, so it trains from the YAML topology instead.
                weights = f"{model}-p2.yaml" if p2 else f"{model}.pt"

                config = {
                    "exp_id": exp_id, "regime": regime, "seed": seed, "epochs": epochs,
                    "imgsz": imgsz, "batch": batch, "model": model, "p2_head": p2,
                    "weights": weights, "synth_pool": synth_pool, "synth_ratio": synth_ratio,
                    "device": device, "git_sha": git_sha(), "target_steps": target_steps,
                    "data_yaml": str(data_yaml),
                }
                (exp_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

                print(f"[train] {exp_id}  weights={weights}  imgsz={imgsz} batch={batch} "
                      f"epochs={epochs}")

                t0 = time.time()
                net = Net(weights)
                # Epoch-based schedules must scale with the epoch count, or a
                # step-matched run on a large pool is mostly warmup: at 10,160
                # images a 6,000-step budget is ~5 epochs, and Ultralytics'
                # default warmup_epochs=3 would then be 60% of training while
                # close_mosaic=10 exceeds the run entirely.
                warmup = min(3.0, max(1.0, epochs / 10.0))
                close_mosaic = min(10, max(0, epochs // 5))
                if epochs < 20:
                    print(f"[train] short run ({epochs} ep): warmup={warmup:.1f} "
                          f"close_mosaic={close_mosaic}")
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
                    warmup_epochs=warmup,
                    # small-object friendly augmentation; heavy scale jitter destroys 6 px defects
                    scale=0.3,
                    mosaic=0.5,
                    close_mosaic=close_mosaic,
                    fliplr=0.5,
                    flipud=0.5,
                    degrees=10.0,
                    translate=0.1,
                    erasing=0.0,
                )
            elapsed = time.time() - t0

            # plots=True here, not during training. Ultralytics re-renders its
            # plot set every epoch when train(plots=True), which is wasteful on a
            # 4 GB laptop; but this val pass runs ONCE and is the only thing that
            # emits confusion_matrix.png, PR_curve.png, F1_curve.png and
            # P/R_curve.png into experiments/<exp_id>/run/. Those were silently
            # absent from every run because this said plots=False.
            metrics = net.val(data=str(data_yaml), split="val", imgsz=imgsz,
                              batch=batch, device=device, plots=True)
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

            # Per-class AP, carried alongside the training support that produced it.
            # A class with 5 training images will score near zero; recording the two
            # together is what stops that number being read as a property of the method.
            class_names = export["class_names"]
            support = export["train_images_per_class"]
            per_class = {}
            for i, cname in enumerate(class_names):
                try:
                    ap50, ap = float(box.ap50[i]), float(box.ap[i])
                except (IndexError, TypeError):
                    ap50 = ap = 0.0
                per_class[cname] = {
                    "AP50": round(ap50, 4),
                    "AP50_95": round(ap, 4),
                    "train_images": support.get(cname, 0),
                    "interpretable": support.get(cname, 0) >= 10,
                }
            result["per_class"] = per_class
            result["low_support_classes"] = export.get("low_support_classes", {})
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

            _append_master(result)
            print(f"[train] {exp_id}: mAP50={result['mAP50']:.4f} "
                  f"mAP50-95={result['mAP50_95']:.4f} P={result['precision']:.3f} "
                  f"R={result['recall']:.3f}  ({elapsed/60:.1f} min)")
            for cname, v in per_class.items():
                flag = "" if v["interpretable"] else "   <- NOT INTERPRETABLE (low support)"
                print(f"         {cname:<10} AP50={v['AP50']:.4f} AP50-95={v['AP50_95']:.4f} "
                      f"(train imgs={v['train_images']}){flag}")
            return result
        except Exception:
            # print while still teed so OOM / crashes land in console.log
            import traceback
            traceback.print_exc()
            raise


MASTER_FIELDS = ["exp_id", "regime", "seed", "model", "p2_head", "synth_pool",
                 "synth_ratio", "epochs", "imgsz", "batch", "mAP50", "mAP50_95",
                 "precision", "recall", "params_M", "train_seconds", "git_sha"]


def _append_master(row: dict) -> None:
    """UPSERT by exp_id - a re-run REPLACES its row rather than stacking a duplicate.

    This was append-only, which quietly corrupted every aggregate. exp_id is
    deterministic (regime_model_seed), so re-running an experiment - after a
    crash, a resume, or a protocol change - appended a second row for the same
    experiment. The file reached 14 rows for 10 distinct experiments, and
    real_only_seed42 appeared twice with DIFFERENT values (0.0583 at 300 epochs,
    0.0513 at 100). Naive grouping then reported real_only as n=4 over
    "seeds 1,2,42,42", averaging a stale 300-epoch result into a 100-epoch mean.
    Silent, and exactly the kind of thing that survives into a results table.
    """
    import csv

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if MASTER_CSV.exists():
        with MASTER_CSV.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("exp_id"):
                    existing[r["exp_id"]] = r          # later row wins on read too

    existing[row["exp_id"]] = {k: row.get(k, "") for k in MASTER_FIELDS}

    tmp = MASTER_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for eid in sorted(existing):
            w.writerow(existing[eid])
    tmp.replace(MASTER_CSV)                            # atomic, so a crash mid-write
                                                       # cannot truncate the results


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
    ap.add_argument("--known-classes", default="0,1",
                    help="closed-set class ids; '1' alone = open-set (PbI2 held out)")
    ap.add_argument("--target-steps", type=int, default=None,
                    help="cap wall-clock: derive epochs from this / steps-per-epoch")
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=30,
                    help="Ultralytics early-stop patience. RAISE THIS above the "
                         "epoch count for a step-matched run: at 160 images a "
                         "8,300-step budget is 415 epochs, and the default 30 "
                         "would stop it early and silently un-match the "
                         "comparison it exists to make.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from experiments/<exp_id>/run/weights/last.pt "
                         "(skips wipe; finished runs with metrics.json are skipped)")
    a = ap.parse_args()
    train(regime=a.regime, seed=a.seed, epochs=a.epochs, imgsz=a.imgsz,
          batch=a.batch, model=a.model, p2=a.p2, synth_pool=a.synth_pool,
          synth_ratio=a.synth_ratio, device=a.device,
          known_classes=tuple(int(x) for x in a.known_classes.split(",")),
          target_steps=a.target_steps, min_epochs=a.min_epochs,
          patience=a.patience, resume=a.resume)
