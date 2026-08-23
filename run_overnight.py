#!/usr/bin/env python
"""
Unattended overnight run.

    py -3.10 run_overnight.py                 # full schedule
    py -3.10 run_overnight.py --dry-run       # print the plan and exit
    py -3.10 run_overnight.py --from 3        # resume from step 3

Design for running while nobody is watching:

  * every step is isolated - a failure is logged and the schedule continues,
    because losing one step should not cost the whole night;
  * every step appends to outputs/overnight.log and outputs/overnight_state.json
    as it completes, so partial results survive a crash or a power cut;
  * nothing here touches the TEST split. Stage 9 stays manual and deliberate.

The schedule follows the order that avoids redundant work: establish the honest
baseline, improve the generator, then measure how much synthetic to use. Running
the scaling ladder before the refiner would mean running it twice, since the
ladder characterises one specific synthetic distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

OUT = ROOT / "outputs"
LOG = OUT / "overnight.log"
STATE = OUT / "overnight_state.json"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------- steps ----

def step_synth(cfg):
    """Regenerate the renderer pool at the working size."""
    from synth.generate import generate, severity_ladder
    s = generate(n_images=cfg["n_synth"], pool_name="controlled",
                 seed=cfg["seed"], render_px=cfg["render_px"], pbi2_fraction=0.0)
    severity_ladder(render_px=cfg["render_px"])
    return s


def step_refiner_fft(cfg):
    from train_refiner import train
    return train(epochs=cfg["refiner_epochs"], batch=cfg["refiner_batch"],
                 use_fft=True, tag="fft")["history"][-1]


def step_refiner_nofft(cfg):
    """Ablation A1 / hypothesis H2: same budget, Fourier branch removed."""
    from train_refiner import train
    return train(epochs=cfg["refiner_epochs"], batch=cfg["refiner_batch"],
                 use_fft=False, tag="nofft")["history"][-1]


def step_apply_refiner(cfg):
    from synth.apply_refiner import refine_pool
    a = refine_pool("controlled", "refined", "refiner_fft.pth")
    b = refine_pool("controlled", "refined_nofft", "refiner_nofft.pth")
    return {"fft": a, "nofft": b}


def step_matrix_refined(cfg):
    """E-C (refined synth) and the H2 ablation, at matched budget."""
    from train_detector import train
    rows = []
    for seed in cfg["seeds"]:
        rows.append(train(regime="real_plus_refined", seed=seed,
                          epochs=cfg["epochs"], batch=cfg["batch"],
                          synth_pool="refined"))
        rows.append(train(regime="real_plus_refined_nofft", seed=seed,
                          epochs=cfg["epochs"], batch=cfg["batch"],
                          synth_pool="refined_nofft"))
    return [{k: r[k] for k in ("exp_id", "mAP50", "mAP50_95")} for r in rows]


def step_scaling(cfg):
    """How much synthetic actually helps - one seed per ratio to find the shape."""
    from train_detector import train
    rows = []
    for ratio in cfg["ratios"]:
        rows.append(train(regime=f"scale_{int(ratio*100):03d}", seed=cfg["seeds"][0],
                          epochs=cfg["epochs"], batch=cfg["batch"],
                          synth_pool="refined", synth_ratio=ratio))
    return [{"ratio": r, "exp_id": x["exp_id"], "mAP50": x["mAP50"],
             "mAP50_95": x["mAP50_95"]} for r, x in zip(cfg["ratios"], rows)]


def step_classical(cfg):
    from eval.microdefectcv_baseline import run_baseline
    return run_baseline(split="val", min_area=cfg["mdcv_min_area"])


def step_report(cfg):
    import make_report
    return make_report.build()


STEPS = [
    ("synth", "Regenerate renderer pool + counterfactual ladder", step_synth, 15),
    ("refiner_fft", "Train refiner (Fourier branch ON)", step_refiner_fft, 30),
    ("refiner_nofft", "Train refiner (Fourier OFF - ablation A1/H2)", step_refiner_nofft, 30),
    ("apply_refiner", "Build refined synthetic pools", step_apply_refiner, 15),
    ("matrix_refined", "E-C + H2 ablation across seeds", step_matrix_refined, 100),
    ("scaling", "Synthetic scaling ladder", step_scaling, 90),
    ("classical", "MicroDefectCV R1 baseline on val", step_classical, 10),
    ("report", "Assemble results report", step_report, 2),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--render-px", type=int, default=512)
    ap.add_argument("--refiner-epochs", type=int, default=30)
    ap.add_argument("--refiner-batch", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ratios", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mdcv-min-area", type=int, default=120)
    a = ap.parse_args()

    cfg = dict(epochs=a.epochs, batch=a.batch, n_synth=a.n_synth,
               render_px=a.render_px, refiner_epochs=a.refiner_epochs,
               refiner_batch=a.refiner_batch, seed=a.seed,
               seeds=[int(s) for s in a.seeds.split(",")],
               ratios=[float(r) for r in a.ratios.split(",")],
               mdcv_min_area=a.mdcv_min_area)

    todo = STEPS[a.start:]
    total = sum(s[3] for s in todo)
    eta = datetime.now() + timedelta(minutes=total)

    print("=" * 74)
    print("  OVERNIGHT SCHEDULE")
    print("=" * 74)
    for i, (key, title, _, mins) in enumerate(STEPS):
        mark = "  " if i >= a.start else "= "   # '=' means skipped via --from
        print(f"{mark}[{i}] {key:<16} ~{mins:>3} min   {title}")
    print("-" * 74)
    print(f"  estimated total: {total} min (~{total/60:.1f} h)   ETA {eta:%H:%M}")
    print(f"  config: epochs={a.epochs} batch={a.batch} n_synth={a.n_synth} "
          f"seeds={cfg['seeds']} ratios={cfg['ratios']}")
    print("=" * 74)
    if a.dry_run:
        return 0

    state = {"started": datetime.now(), "config": cfg, "steps": {}}
    save_state(state)
    log(f"START overnight run, {len(todo)} steps, ETA {eta:%H:%M}")

    ok, failed = [], []
    for i, (key, title, fn, mins) in enumerate(STEPS):
        if i < a.start:
            continue
        log(f"--- [{i}] {key}: {title}")
        t0 = time.time()
        try:
            result = fn(cfg)
            dt = time.time() - t0
            state["steps"][key] = {"status": "ok", "minutes": round(dt / 60, 1),
                                   "result": result}
            ok.append(key)
            log(f"    OK  {key} in {dt/60:.1f} min")
        except Exception as exc:
            dt = time.time() - t0
            tb = traceback.format_exc()
            state["steps"][key] = {"status": "FAILED", "minutes": round(dt / 60, 1),
                                   "error": str(exc), "traceback": tb}
            failed.append(key)
            log(f"    FAILED {key} after {dt/60:.1f} min: {exc}")
            log("    " + tb.replace("\n", "\n    "))
            log("    continuing with the next step")
        save_state(state)

    state["finished"] = datetime.now()
    save_state(state)

    log("=" * 60)
    log(f"DONE. {len(ok)} ok, {len(failed)} failed.")
    if failed:
        log(f"failed steps: {', '.join(failed)}  (see {STATE} for tracebacks)")
    log(f"results: {OUT/'master_results.csv'}  report: {OUT/'RESULTS.md'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
