"""
Materialise a YOLO-format dataset directory for one training regime.

Ultralytics wants images/ and labels/ trees plus a data.yaml. This builds them
from data/splits/*.json and, optionally, a synthetic pool - which is the only
place the E-A..E-E regimes differ.

Two things are asserted rather than assumed, because a leak here would be
invisible in the training log and would quietly inflate every number downstream:

  * no source group appears in more than one split;
  * every synthetic image's background group came from TRAIN.

The test split is materialised only when explicitly asked for (stage 9).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CURATED = ROOT / "curated"
SPLITS = ROOT / "splits"
SYNTH = ROOT / "synthetic"
EXPORTS = ROOT / "yolo"


def _load(name: str) -> dict:
    return json.loads((SPLITS / f"{name}.json").read_text(encoding="utf-8"))


def _write_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> bool:
    if not img_src.exists():
        return False
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst)
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("", encoding="utf-8")
    return True


def build(regime: str = "real_only", synth_pool: str | None = None,
          synth_ratio: float = 1.0, include_test: bool = False,
          include_backgrounds: bool = True) -> Path:
    """Write data/yolo/<regime>/ and return the path to its data.yaml.

    synth_ratio is a fraction of the available synthetic pool, so the scaling
    experiment is a parameter rather than a separate code path.
    """
    meta = json.loads((CURATED / "curated.json").read_text(encoding="utf-8"))
    names = meta["class_names"]

    out = EXPORTS / regime
    if out.exists():
        shutil.rmtree(out)

    wanted = ["train", "val"] + (["test"] if include_test else [])
    splits = {s: _load(s) for s in wanted}

    # leakage assertion, re-checked at the point of materialisation
    gsets = {s: set(v["groups"]) for s, v in splits.items()}
    for a in gsets:
        for b in gsets:
            if a < b:
                assert not (gsets[a] & gsets[b]), f"GROUP LEAK {a}/{b}"

    counts = {}
    for split, payload in splits.items():
        n = 0
        for r in payload["records"]:
            if not include_backgrounds and r["n_boxes"] == 0:
                continue
            rel = Path(r["file"])
            ok = _write_pair(
                CURATED / "images" / rel,
                (CURATED / "labels" / rel).with_suffix(".txt"),
                out / "images" / split / rel.name,
                out / "labels" / split / rel.with_suffix(".txt").name,
            )
            n += int(ok)
        counts[split] = n

    # ---- synthetic images join TRAIN only ---------------------------------
    n_synth = 0
    if synth_pool:
        pool = SYNTH / synth_pool
        manifest = pool / "manifest.jsonl"
        assert manifest.exists(), f"no synthetic pool at {pool}"

        train_groups = gsets["train"]
        entries = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        for e in entries:
            assert e["background_group"] in train_groups, (
                f"synthetic image {e['stem']} was built on background group "
                f"'{e['background_group']}', which is not in train")

        keep = entries[: max(0, int(len(entries) * synth_ratio))]
        for e in keep:
            n_synth += int(_write_pair(
                pool / "images" / f"{e['stem']}.jpg",
                pool / "labels" / f"{e['stem']}.txt",
                out / "images" / "train" / f"{e['stem']}.jpg",
                out / "labels" / "train" / f"{e['stem']}.txt",
            ))
        counts["train"] += n_synth

    yaml_path = out / "data.yaml"
    lines = [
        f"# regime: {regime}",
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines += [f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "export_manifest.json").write_text(json.dumps({
        "regime": regime,
        "counts": counts,
        "synthetic_pool": synth_pool,
        "synthetic_ratio": synth_ratio,
        "synthetic_images": n_synth,
        "include_test": include_test,
        "class_names": names,
    }, indent=1), encoding="utf-8")

    real_train = counts["train"] - n_synth
    print(f"[yolo] {regime}: train={counts['train']} ({real_train} real + {n_synth} synth) "
          f"val={counts['val']}" + (f" test={counts['test']}" if include_test else ""))
    return yaml_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="real_only")
    ap.add_argument("--synth-pool", default=None)
    ap.add_argument("--synth-ratio", type=float, default=1.0)
    ap.add_argument("--include-test", action="store_true")
    a = ap.parse_args()
    build(regime=a.regime, synth_pool=a.synth_pool,
          synth_ratio=a.synth_ratio, include_test=a.include_test)
