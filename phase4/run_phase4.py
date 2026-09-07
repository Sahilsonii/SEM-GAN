"""
Every Phase 4 run, in one command:

    py -3.10 phase4/run_phase4.py

Six detector runs at 8,300 steps: the diffusion pool ALONE (releasability)
and MIXED with the 160 real images (augmentation), three seeds each, then one
merged final_eval on the locked test split and the test suites. Each run is a
subprocess with --resume; re-run the same command to continue after a crash.
Pre-registered reading: phase4/RESULTS_phase4.md.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable]
LOG = ROOT / "outputs" / "phase4.log"
POOL = "diffusion_cn"
S8 = ["--synth-pool", POOL, "--synth-ratio", "1.0", "--target-steps", "8300", "--patience", "500"]
RUNS = []
for seed in (42, 1, 2):
    RUNS.append((f"{POOL}_only_steps8300_yolo11s_seed{seed}",
                 ["--regime", f"{POOL}_only_steps8300", "--seed", str(seed), "--synth-only", *S8]))
for seed in (42, 1, 2):
    RUNS.append((f"{POOL}_steps8300_yolo11s_seed{seed}",
                 ["--regime", f"{POOL}_steps8300", "--seed", str(seed), *S8]))


def log(line):
    print(f"\n[phase4] {line}\n", flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")


def run(cmd):
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    LOG.parent.mkdir(exist_ok=True)
    import json
    summ = ROOT / "data/synthetic" / POOL / "summary.json"
    if not summ.exists() or json.loads(summ.read_text())["images"] < 500:
        sys.exit(f"pool '{POOL}' not complete - finish: py -3.10 phase4/generate_diffusion.py")
    log("=== start ===")
    for exp_id, args in RUNS:
        t0 = time.time()
        rc = run(PY + ["train_detector.py", *args, "--resume"])
        log(f"{exp_id}: exit={rc} ({(time.time() - t0) / 60:.1f} min)")
    ckpts = []
    for exp_id, _ in RUNS:
        best = ROOT / "experiments" / exp_id / "run/weights/best.pt"
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
