"""
Every Phase 3 run, in order, in one command:

    py -3.10 phase3/run_phase3.py

Six detector runs at the 8,300-step budget - the cell where real-only is
strongest (0.148) and where dilution is gone - three seeds per pool, then one
merged final_eval on the locked test split and the test suites. Each run is a
subprocess with --resume, so a finished run is skipped and a crash does not
stop the rest; re-run the same command to continue.

Comparators already on disk: real_only_steps8300 (0.1480 ± 0.0367) and
scale_005_bal_steps8300 (0.1347 ± 0.0052). Pre-registered reading is in
phase3/RESULTS_phase3.md, written before this ran.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable]
LOG = ROOT / "outputs" / "phase3.log"

S8 = ["--target-steps", "8300", "--patience", "500", "--synth-ratio", "1.0"]
RUNS = []
for pool in ("layout_transfer", "augment_defective"):
    for seed in (42, 1, 2):
        RUNS.append((f"{pool}_steps8300_yolo11s_seed{seed}",
                     ["--regime", f"{pool}_steps8300", "--seed", str(seed),
                      "--synth-pool", pool, *S8]))


def log(line):
    print(f"\n[phase3] {line}\n", flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")


def run(cmd):
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    LOG.parent.mkdir(exist_ok=True)
    for pool in ("layout_transfer", "augment_defective"):
        if not (ROOT / "data" / "synthetic" / pool / "manifest.jsonl").exists():
            sys.exit(f"pool '{pool}' missing - run: py -3.10 phase3/synth_compose.py --pool {pool}")
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
    rc = run(PY + ["-m", "pytest", "tests", "phase3/tests", "-q"])
    log(f"pytest: exit={rc}")
    log("=== done - tell Claude ===")
