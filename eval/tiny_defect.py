"""
Scale-stratified evaluation (contribution N3).

Bins come from configs/tiny_defect_bins.yaml and are anchored to detector
strides, not to the data distribution. Nothing here reads results before
assigning a box to a bin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tiny_defect_bins.yaml"


def load_bins() -> tuple[int, list[dict]]:
    """Minimal parse - avoids a PyYAML dependency for a fixed-shape file."""
    ref, bins = 640, []
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("reference_px:"):
            ref = int(s.split(":")[1].split("#")[0].strip())
        elif s.startswith("- {name:"):
            body = s[s.index("{") + 1:s.rindex("}")]
            d = {}
            for part in body.split(", "):
                if ":" not in part:
                    continue
                k, v = part.split(":", 1)
                d[k.strip()] = v.strip().strip('"')
            bins.append({
                "name": d["name"],
                "min_px": float(d["min_px"]),
                "max_px": math.inf if d["max_px"] in ("null", "None") else float(d["max_px"]),
            })
    return ref, bins


def box_side_px(w: float, h: float, reference_px: int) -> float:
    """Equivalent square side, in pixels at the detector input resolution."""
    return math.sqrt(max(w, 0.0) * max(h, 0.0)) * reference_px


def assign_bin(w: float, h: float, reference_px: int, bins: list[dict]) -> str:
    side = box_side_px(w, h, reference_px)
    for b in bins:
        if b["min_px"] <= side < b["max_px"]:
            return b["name"]
    return bins[-1]["name"]


def profile_split(split: str = "train") -> dict:
    """Bin histogram for a split - the descriptive stat, not a result."""
    ref, bins = load_bins()
    recs = json.loads((ROOT / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    counts = {b["name"]: 0 for b in bins}
    for r in recs["records"]:
        for _, _, _, w, h in r["boxes"]:
            counts[assign_bin(w, h, ref, bins)] += 1
    total = sum(counts.values()) or 1
    return {
        "split": split,
        "reference_px": ref,
        "counts": counts,
        "share": {k: round(v / total, 4) for k, v in counts.items()},
        "total_boxes": total,
    }


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        p = profile_split(split)
        parts = " ".join(f"{k}={v}({p['share'][k]*100:.1f}%)" for k, v in p["counts"].items())
        print(f"[bins] {split:<5} n={p['total_boxes']:<5} {parts}")
