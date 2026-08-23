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


def _fmt_boxes(boxes) -> str:
    """YOLO label text: one `cls cx cy w h` line per box."""
    return "".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes)


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
          include_backgrounds: bool = True,
          known_classes: tuple = (0, 1)) -> Path:
    """Write data/yolo/<regime>/ and return the path to its data.yaml.

    known_classes is the closed-set vocabulary. Everything else is an UNKNOWN
    morphology reserved for open-set evaluation and must never be trained on.

    Default (0, 1) = two-class detection, pbi2 + pinhole.

    CAVEAT, and it must travel with every PbI2 number this produces: only 9
    source groups in the corpus carry PbI2 and just 5 PbI2 images land in train.
    PbI2 AP is therefore expected to be near zero and is not interpretable as a
    measure of the method - it measures the annotation budget. Report it, label
    it, do not build a claim on it. The count is written into every export
    manifest and metrics file so the caveat cannot get separated from the number.

    Pass known_classes=(1,) for the open-set configuration, where PbI2 becomes
    the held-out unknown morphology instead of a trained class.

    An image whose only annotations are unknown-class is EXCLUDED rather than
    emitted with an empty label. Emitting it would reintroduce exactly the false
    negative this pipeline just spent a commit removing.
    """
    known = set(known_classes)
    remap = {c: i for i, c in enumerate(sorted(known))}
    meta = json.loads((CURATED / "curated.json").read_text(encoding="utf-8"))
    all_names = meta["class_names"]
    names = [all_names[c] for c in sorted(known)]
    unknown_names = [n for i, n in enumerate(all_names) if i not in known]

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

    counts, held_out = {}, {}
    for split, payload in splits.items():
        n, skipped = 0, []
        for r in payload["records"]:
            boxes = [b for b in r["boxes"] if b[0] in known]
            has_unknown = len(boxes) != r["n_boxes"]

            if r["n_boxes"] > 0 and not boxes:
                skipped.append(r["file"])      # unknown-only image -> open-set pool
                continue
            if not include_backgrounds and not boxes:
                continue

            rel = Path(r["file"])
            img_src = CURATED / "images" / rel
            if not img_src.exists():
                continue
            img_dst = out / "images" / split / rel.name
            lbl_dst = (out / "labels" / split / rel.name).with_suffix(".txt")
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_src, img_dst)
            lbl_dst.write_text(
                _fmt_boxes([[remap[b[0]], b[1], b[2], b[3], b[4]] for b in boxes]),
                encoding="utf-8")
            n += 1
        counts[split] = n
        held_out[split] = skipped

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
            src_lbl = pool / "labels" / f"{e['stem']}.txt"
            if not src_lbl.exists():
                continue
            boxes = []
            for line in src_lbl.read_text().splitlines():
                q = line.split()
                if len(q) >= 5 and int(q[0]) in known:
                    boxes.append([remap[int(q[0])]] + [float(v) for v in q[1:5]])
            if not boxes:
                continue      # synthetic image of an unknown class - not for training
            img_dst = out / "images" / "train" / f"{e['stem']}.jpg"
            lbl_dst = out / "labels" / "train" / f"{e['stem']}.txt"
            shutil.copy2(pool / "images" / f"{e['stem']}.jpg", img_dst)
            lbl_dst.write_text(_fmt_boxes(boxes), encoding="utf-8")
            n_synth += 1
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

    # per-class training support - the number the PbI2 caveat rests on
    train_per_class = {n: 0 for n in names}
    for r in splits["train"]["records"]:
        for c in {b[0] for b in r["boxes"]} & known:
            train_per_class[all_names[c]] += 1
    low_support = {k: v for k, v in train_per_class.items() if v < 10}

    (out / "export_manifest.json").write_text(json.dumps({
        "regime": regime,
        "counts": counts,
        "synthetic_pool": synth_pool,
        "synthetic_ratio": synth_ratio,
        "synthetic_images": n_synth,
        "include_test": include_test,
        "known_classes": sorted(known),
        "class_names": names,
        "held_out_unknown_classes": unknown_names,
        "held_out_images": held_out,
        "train_images_per_class": train_per_class,
        "low_support_classes": low_support,
    }, indent=1), encoding="utf-8")

    real_train = counts["train"] - n_synth
    n_held = sum(len(v) for v in held_out.values())
    print(f"[yolo] {regime}: train={counts['train']} ({real_train} real + {n_synth} synth) "
          f"val={counts['val']}" + (f" test={counts['test']}" if include_test else "")
          + f"  | known={names}"
          + (f" held-out={unknown_names} ({n_held} imgs reserved)" if unknown_names else ""))
    print(f"[yolo] train images per class: {train_per_class}")
    for k, v in low_support.items():
        print(f"[yolo] WARNING low support: '{k}' has {v} training images - "
              f"its AP will not be interpretable")
    return yaml_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="real_only")
    ap.add_argument("--synth-pool", default=None)
    ap.add_argument("--synth-ratio", type=float, default=1.0)
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--known-classes", default="0,1",
                    help="closed-set class ids; '1' alone = open-set (PbI2 held out)")
    a = ap.parse_args()
    build(regime=a.regime, synth_pool=a.synth_pool,
          synth_ratio=a.synth_ratio, include_test=a.include_test,
          known_classes=tuple(int(x) for x in a.known_classes.split(",")))
