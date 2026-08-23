"""
Stage 1a - curate the raw snapshot into an honest corpus.

Three defects in the raw corpus, each removed here:

  1. 25 byte-identical duplicate groups (440 files -> 415 unique).
  2. 102 `_aug###` files that are geometric copies of other files. Augmentation
     belongs in the training loop, not the dataset; keeping them lets one source
     image appear on both sides of a split.
  3. Folder label contradicts box label - `class0_pbI2` images carry 41 pinhole
     boxes, `class2` images carry 346 PbI2 boxes. Only the boxes are trusted;
     the folder label is dropped entirely.

Box classes 1 (3D_pinholes) and 2 (3D-2D_pinholes) are merged into a single
`pinhole` class by default: they are the same defect, the 1-vs-2 distinction
encodes film morphology (a specimen property), and class 2 has only 14 real
source images. `--keep-3class` preserves the original scheme for ablation A7.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from snapshot import DEST as SNAP, md5

ROOT = Path(__file__).resolve().parent
CURATED = ROOT / "curated"

# raw box class ids -> merged scheme
MERGED_MAP = {0: 0, 1: 1, 2: 1}          # PbI2 -> 0, both pinhole kinds -> 1
MERGED_NAMES = ["pbi2", "pinhole"]
RAW_NAMES = ["pbi2", "pinhole_3d", "pinhole_3d2d"]

_AUG = re.compile(r"_aug\d+")
_CLSSUF = re.compile(r"_class\d[^.]*")
_PAREN = re.compile(r"\s*\(\d+\)$")


def group_key(stem: str) -> str:
    """Collapse a filename to the physical source it came from.

    `01-07`, `01-07_aug000`, `01-07_class3_3D_background`, `01-07 (2)` are all
    the same specimen region and must never straddle a split.
    """
    s = _AUG.sub("", stem)
    s = _CLSSUF.sub("", s)
    return _PAREN.sub("", s).strip()


def read_boxes(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append([int(parts[0]), *(float(v) for v in parts[1:5])])
    return out


def curate(keep_3class: bool = False) -> dict:
    images = sorted(p for p in (SNAP / "images").rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})

    seen_md5: dict[str, str] = {}
    records, dropped = [], Counter()

    for img in images:
        rel = img.relative_to(SNAP / "images")
        stem = img.stem

        if _AUG.search(stem):
            dropped["augmented_copy"] += 1
            continue

        h = md5(img)
        if h in seen_md5:
            dropped["byte_duplicate"] += 1
            continue
        seen_md5[h] = rel.as_posix()

        boxes = read_boxes(SNAP / "labels" / rel.with_suffix(".txt"))
        if not keep_3class:
            boxes = [[MERGED_MAP[b[0]], *b[1:]] for b in boxes]

        records.append({
            "file": rel.as_posix(),
            "stem": stem,
            "group": group_key(stem),
            "md5": h,
            "boxes": boxes,
            "n_boxes": len(boxes),
        })

    groups = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)

    cls_counter = Counter(b[0] for r in records for b in r["boxes"])
    names = RAW_NAMES if keep_3class else MERGED_NAMES

    meta = {
        "class_names": names,
        "n_raw_files": len(images),
        "n_curated": len(records),
        "n_groups": len(groups),
        "n_groups_with_defects": sum(1 for g in groups.values()
                                     if any(r["n_boxes"] for r in g)),
        "n_boxes": sum(r["n_boxes"] for r in records),
        "boxes_per_class": {names[c]: n for c, n in sorted(cls_counter.items())},
        "dropped": dict(dropped),
        "records": records,
    }

    CURATED.mkdir(parents=True, exist_ok=True)
    (CURATED / "curated.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"[curate] {meta['n_raw_files']} raw -> {meta['n_curated']} curated "
          f"({dropped['augmented_copy']} augmented, {dropped['byte_duplicate']} duplicate dropped)")
    print(f"[curate] {meta['n_groups']} source groups, "
          f"{meta['n_groups_with_defects']} carry defects, {meta['n_boxes']} boxes")
    print(f"[curate] boxes per class: {meta['boxes_per_class']}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-3class", action="store_true",
                    help="keep raw 3-class box scheme (ablation A7)")
    curate(**vars(ap.parse_args()))
