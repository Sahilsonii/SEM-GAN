"""
Regenerate the per-run Ultralytics diagnostic plots for runs that already
finished.

train_detector passed plots=False to its final val() call, so NO run ever
produced confusion_matrix.png, PR_curve.png, F1_curve.png or P/R_curve.png -
the standard detection diagnostics. That default is now fixed for future runs,
but the eleven completed runs would otherwise have to be retrained to get them.
They do not: the weights are saved, so re-running validation alone reproduces
every plot in a couple of minutes per run.

Writes into experiments/<exp_id>/run/val_plots/ rather than run/, so nothing
that the original training wrote can be overwritten.

GPU: --device 0 by default. Pass --device cpu if the card is busy with a
training job - it is slower but touches no VRAM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
# Ultralytics 8.4 prefixes the curve plots "Box" (BoxPR_curve.png, not
# PR_curve.png). Getting this list wrong does not lose any file, it only
# undercounts them in the index, which is exactly how it went unnoticed.
WANTED = ["confusion_matrix.png", "confusion_matrix_normalized.png",
          "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png",
          "BoxF1_curve.png"]


def regen_one(exp_dir: Path, device: str, imgsz: int | None = None) -> dict:
    cfg_p = exp_dir / "config.json"
    best = exp_dir / "run" / "weights" / "best.pt"
    if not cfg_p.exists() or not best.exists():
        return {"exp_id": exp_dir.name, "status": "skipped: no config or best.pt"}

    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    data_yaml = Path(cfg["data_yaml"])
    if not data_yaml.exists():
        # e.g. real_plus_controlled, whose export directory was rebuilt under a
        # different regime name. Report it rather than guessing a substitute.
        return {"exp_id": exp_dir.name,
                "status": f"skipped: data yaml missing ({data_yaml})"}

    from ultralytics import RTDETR, YOLO
    Net = RTDETR if str(cfg.get("model", "")).startswith("rtdetr") else YOLO

    net = Net(str(best))
    m = net.val(data=str(data_yaml), split="val",
                imgsz=imgsz or cfg.get("imgsz", 640),
                batch=cfg.get("batch", 8), device=device, plots=True,
                project=str(exp_dir), name="val_plots", exist_ok=True)

    out = exp_dir / "val_plots"
    made = [f for f in WANTED if (out / f).exists()]
    return {"exp_id": exp_dir.name, "status": "ok",
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "plots": made, "dir": str(out.relative_to(ROOT))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0",
                    help="'0' for GPU, 'cpu' if the card is busy training")
    ap.add_argument("--only", default=None,
                    help="substring filter on exp_id")
    ap.add_argument("--imgsz", type=int, default=None)
    a = ap.parse_args()

    dirs = sorted(d for d in EXP.iterdir() if d.is_dir())
    if a.only:
        dirs = [d for d in dirs if a.only in d.name]
    if not dirs:
        print("[regen] no experiments matched")
        return

    results = []
    for d in dirs:
        print(f"\n[regen] {d.name}  device={a.device}")
        try:
            r = regen_one(d, a.device, a.imgsz)
        except Exception as exc:                     # one bad run must not stop the rest
            r = {"exp_id": d.name, "status": f"FAILED: {type(exc).__name__}: {exc}"}
        results.append(r)
        print(f"        {r['status']}"
              + (f"  mAP50={r['mAP50']}  {len(r['plots'])} plots" if r["status"] == "ok" else ""))

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n[regen] {len(ok)}/{len(results)} runs re-validated with plots")
    for r in results:
        if r["status"] != "ok":
            print(f"        {r['exp_id']}: {r['status']}")

    (ROOT / "outputs" / "val_plots_index.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print(f"[regen] index -> outputs/val_plots_index.json")


if __name__ == "__main__":
    main()
