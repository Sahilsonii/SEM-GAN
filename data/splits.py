"""
Stage 1b - Leakage-Safe Grouped Evaluation Protocol (contribution N0).

Splits at the level of the *source group*, never the file. A group is one
physical specimen region; its augmented copies, its re-filings under a second
class folder, and its `(2)` re-exports all travel together.

Also audits the ORIGINAL published splits so the cost of not doing this is
measured rather than asserted. That audit is the evidence for N0 and is the
only thing that ever reads raw_snapshot/{train,val,test}.txt.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_dataset import CURATED, group_key
from snapshot import DEST as SNAP

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "splits"
RATIOS = {"train": 0.60, "val": 0.15, "test": 0.25}


def audit_original() -> dict:
    """Quantify leakage in the published train/val/test.txt."""
    stems = {}
    for name in ("train", "val", "test"):
        f = SNAP / f"{name}.txt"
        if not f.exists():
            return {"available": False}
        keys = defaultdict(list)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                base = Path(line.strip().replace("\\", "/")).name
                keys[group_key(Path(base).stem)].append(base)
        stems[name] = keys

    tr, va, te = (set(stems[k]) for k in ("train", "val", "test"))
    audit = {
        "available": True,
        "groups": {k: len(v) for k, v in stems.items()},
        "train_val_overlap": len(tr & va),
        "train_test_overlap": len(tr & te),
        "val_test_overlap": len(va & te),
        "examples": [
            {"group": g, "train": stems["train"][g][:2], "test": stems["test"][g][:2]}
            for g in sorted(tr & te)[:5]
        ],
    }
    audit["total_leaked_groups"] = len((tr & va) | (tr & te) | (va & te))
    print(f"[audit] published splits leak: train/val={audit['train_val_overlap']}, "
          f"train/test={audit['train_test_overlap']}, val/test={audit['val_test_overlap']} "
          f"({audit['total_leaked_groups']} distinct groups affected)")
    return audit


def _stratum(recs: list[dict]) -> str:
    """Stratify on defect presence and dominant class - keeps rare PbI2 spread out."""
    boxes = [b[0] for r in recs for b in r["boxes"]]
    if not boxes:
        return "background"
    return f"defect_c{Counter(boxes).most_common(1)[0][0]}"


def build(seed: int = 42) -> dict:
    meta = json.loads((CURATED / "curated.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for r in meta["records"]:
        groups[r["group"]].append(r)

    strata = defaultdict(list)
    for g, recs in groups.items():
        strata[_stratum(recs)].append(g)

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for stratum, gs in sorted(strata.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        n = len(gs)
        n_tr = round(n * RATIOS["train"])
        n_va = round(n * RATIOS["val"])
        for i, g in enumerate(gs):
            assign[g] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    out = {name: {"seed": seed, "groups": [], "records": []} for name in RATIOS}
    for g, split in assign.items():
        out[split]["groups"].append(g)
        out[split]["records"].extend(groups[g])

    # ---- hard guarantees -------------------------------------------------
    gsets = {k: set(v["groups"]) for k, v in out.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (gsets[a] & gsets[b]), f"GROUP LEAK between {a} and {b}"
    msets = {k: {r["md5"] for r in v["records"]} for k, v in out.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (msets[a] & msets[b]), f"MD5 LEAK between {a} and {b}"
    assert not any("_aug" in r["stem"] for v in out.values() for r in v["records"]), \
        "augmented file survived into a split"

    SPLITS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, payload in out.items():
        (SPLITS / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        cls = Counter(b[0] for r in payload["records"] for b in r["boxes"])
        summary[name] = {
            "groups": len(payload["groups"]),
            "images": len(payload["records"]),
            "boxes": sum(r["n_boxes"] for r in payload["records"]),
            "boxes_per_class": {meta["class_names"][c]: n for c, n in sorted(cls.items())},
            "images_with_defects": sum(1 for r in payload["records"] if r["n_boxes"]),
        }

    manifest = {
        "seed": seed,
        "ratios": RATIOS,
        "class_names": meta["class_names"],
        "split_level": "source_group",
        "summary": summary,
        "leakage_assertions": ["group_disjoint", "md5_disjoint", "no_augmented_files"],
        "original_split_audit": audit_original(),
    }
    (SPLITS / "splits_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"[splits] seed={seed}  group-level, leakage assertions passed")
    for name in ("train", "val", "test"):
        s = summary[name]
        print(f"  {name:<5} {s['groups']:>3} groups  {s['images']:>3} imgs  "
              f"{s['images_with_defects']:>3} with defects  {s['boxes']:>5} boxes  {s['boxes_per_class']}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    build(**vars(ap.parse_args()))
