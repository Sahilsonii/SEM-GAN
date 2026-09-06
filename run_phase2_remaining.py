"""
Every remaining Phase 2 run, in order, in one command:

    py -3.10 run_phase2_remaining.py

Each training run is a separate subprocess with --resume, so a finished run
is skipped, a crashed one does not stop the rest, and the live Ultralytics
log is your terminal. Re-running this script after a crash continues where
it stopped. At the end every checkpoint that exists is scored ONCE on the
locked test split (final_eval merges, never overwrites) and the test suite
runs. A one-line-per-step summary lands in outputs/phase2_remaining.log.

Order is by what can change a claim:
  1. copy-paste baseline, 3 seeds     - is defect APPEARANCE the lever?
  2. yolov8s vs yolov8s-P2, seed 42   - ablation A4 (yolo11 has no P2 yaml)
  3. real-only and balanced-synthetic at 8,300 steps, 3 seeds each - the
     empty cell of the budget x data 2x2 (RESULTS.md §17). ~45 min per run.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable]
LOG = ROOT / "outputs" / "phase2_remaining.log"

CP = ["--synth-pool", "copypaste", "--synth-ratio", "1.0", "--target-steps", "2000"]
BAL = ["--synth-pool", "refined", "--synth-ratio", "0.05"]
S8 = ["--target-steps", "8300", "--patience", "500"]
RUNS = [
    # (exp_id, train_detector args)
    ("copypaste_500_steps2000_yolo11s_seed42", ["--regime", "copypaste_500_steps2000", "--seed", "42", *CP]),
    ("copypaste_500_steps2000_yolo11s_seed1",  ["--regime", "copypaste_500_steps2000", "--seed", "1", *CP]),
    ("copypaste_500_steps2000_yolo11s_seed2",  ["--regime", "copypaste_500_steps2000", "--seed", "2", *CP]),
    ("real_only_yolov8s_seed42",               ["--regime", "real_only", "--seed", "42", "--model", "yolov8s"]),
    ("real_only_p2_yolov8s-p2_seed42",         ["--regime", "real_only_p2", "--seed", "42", "--model", "yolov8s", "--p2"]),
    # 3. the empty cell of the 2x2: both arms at 8,300 steps. Real-only at
    #    160 images is 415 epochs by arithmetic (8,300 / 20 per epoch);
    #    patience 500 so early stopping cannot un-match the budget.
    ("real_only_steps8300_yolo11s_seed42",     ["--regime", "real_only_steps8300", "--seed", "42", *S8]),
    ("real_only_steps8300_yolo11s_seed1",      ["--regime", "real_only_steps8300", "--seed", "1", *S8]),
    ("real_only_steps8300_yolo11s_seed2",      ["--regime", "real_only_steps8300", "--seed", "2", *S8]),
    ("scale_005_bal_steps8300_yolo11s_seed42", ["--regime", "scale_005_bal_steps8300", "--seed", "42", *BAL, *S8]),
    ("scale_005_bal_steps8300_yolo11s_seed1",  ["--regime", "scale_005_bal_steps8300", "--seed", "1", *BAL, *S8]),
    ("scale_005_bal_steps8300_yolo11s_seed2",  ["--regime", "scale_005_bal_steps8300", "--seed", "2", *BAL, *S8]),
]


def log(line: str):
    print(f"\n[phase2] {line}\n", flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    LOG.parent.mkdir(exist_ok=True)
    log("=== start ===")
    for exp_id, args in RUNS:
        t0 = time.time()
        rc = run(PY + ["train_detector.py", *args, "--resume"])
        log(f"{exp_id}: exit={rc} ({(time.time() - t0) / 60:.1f} min)")

    ckpts = []
    for exp_id, _ in RUNS:
        best = ROOT / "experiments" / exp_id / "run" / "weights" / "best.pt"
        if (ROOT / "experiments" / exp_id / "metrics.json").exists() and best.exists():
            ckpts.append(f"{exp_id}={best.relative_to(ROOT).as_posix()}")
        else:
            log(f"{exp_id}: NOT finished - excluded from test scoring")
    if ckpts:
        rc = run(PY + ["eval/final_eval.py", "--i-am-sure", "--confirm", "I am done tuning",
                       "--checkpoints", *ckpts])
        log(f"final_eval on {len(ckpts)} checkpoints: exit={rc}")

    rc = run(PY + ["-m", "pytest", "tests", "-q"])
    log(f"pytest: exit={rc}")
    log("=== done - tell Claude ===")
