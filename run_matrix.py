"""Run one experiment matrix: regimes x seeds, identical budget."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="real_only,real_plus_synth")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--p2", action="store_true")
    a = ap.parse_args()

    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for regime in a.regimes.split(","):
            rows.append(train(
                regime=regime, seed=seed, epochs=a.epochs, batch=a.batch,
                imgsz=a.imgsz, p2=a.p2,
                synth_pool="controlled" if "synth" in regime else None))
    print("\n" + "=" * 70)
    print("  MATRIX COMPLETE")
    print("=" * 70)
    for r in rows:
        pc = " ".join(f"{k}={v['AP50']:.4f}" for k, v in r["per_class"].items())
        print(f"  {r['exp_id']:<38} mAP50={r['mAP50']:.4f}  {pc}")
