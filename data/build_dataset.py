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

import cv2
import numpy as np

from sem_bar import find_bar_top
from snapshot import DEST as SNAP, md5

ROOT = Path(__file__).resolve().parent
CURATED = ROOT / "curated"

# raw box class ids -> merged scheme
MERGED_MAP = {0: 0, 1: 1, 2: 1}          # PbI2 -> 0, both pinhole kinds -> 1

# Folders whose images are genuine defect-free negatives. An image in ANY other
# folder with an empty label file is UNLABELLED, not negative - see label_status.
BACKGROUND_FOLDERS = {"class3_3D_background", "class4_3D-2D_background"}
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

        # --- strip the FESEM metadata banner, once, for the whole project ----
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            dropped["unreadable"] += 1
            continue
        H0, W0 = bgr.shape[:2]
        cut = find_bar_top(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        cropped = bgr[:cut]
        H1 = cropped.shape[0]

        kept_boxes = []
        for c, cx, cy, bw, bh in boxes:
            y0 = (cy - bh / 2) * H0
            y1 = (cy + bh / 2) * H0
            y1 = min(y1, cut)                 # clip a box that straddles the banner
            if y1 - y0 < 1.0:                 # box lived entirely inside the banner
                dropped["box_in_banner"] += 1
                continue
            kept_boxes.append([c, cx, ((y0 + y1) / 2) / H1, bw, (y1 - y0) / H1])

        out_img = CURATED / "images" / rel
        out_lbl = (CURATED / "labels" / rel).with_suffix(".txt")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), cropped, [cv2.IMWRITE_JPEG_QUALITY, 96])
        out_lbl.write_text(
            "".join("%d %.6f %.6f %.6f %.6f" % tuple(b) + "\n" for b in kept_boxes),
            encoding="utf-8")

        # An empty label file means two completely different things depending on
        # where the image sits. In class3/class4 it is a curated defect-free
        # background - a true negative. In class0/class1/class2 a human filed it
        # as containing that defect and then nobody drew boxes: 65 of the 73
        # class0_pbI2 images are like this. Training on those as negatives would
        # teach the detector that PbI2 morphology is "nothing here", which is
        # worse than not using them at all.
        folder = rel.parts[0] if len(rel.parts) > 1 else ""
        if kept_boxes:
            label_status = "annotated"
        elif folder in BACKGROUND_FOLDERS:
            label_status = "true_background"
        else:
            label_status = "unlabelled"
            dropped["unlabelled_defect_class_image"] += 1

        records.append({
            "file": rel.as_posix(),          # relative to data/curated/images
            "label_status": label_status,
            "source_folder": folder,
            "stem": stem,
            "group": group_key(stem),
            "md5": h,
            "boxes": kept_boxes,
            "n_boxes": len(kept_boxes),
            "size": [cropped.shape[1], H1],
            "orig_size": [W0, H0],
            "banner_rows_removed": H0 - H1,
        })

    # unlabelled images are kept in curated.json for provenance but excluded
    # from every split, so they can never act as false negatives
    usable = [r for r in records if r["label_status"] != "unlabelled"]

    groups = defaultdict(list)
    for r in usable:
        groups[r["group"]].append(r)

    cls_counter = Counter(b[0] for r in usable for b in r["boxes"])
    names = RAW_NAMES if keep_3class else MERGED_NAMES

    meta = {
        "class_names": names,
        "n_raw_files": len(images),
        "n_curated": len(usable),
        "n_all_records": len(records),
        "n_groups": len(groups),
        "n_groups_with_defects": sum(1 for g in groups.values()
                                     if any(r["n_boxes"] for r in g)),
        "n_boxes": sum(r["n_boxes"] for r in usable),
        "n_annotated_images": sum(1 for r in usable if r["label_status"] == "annotated"),
        "n_true_backgrounds": sum(1 for r in usable if r["label_status"] == "true_background"),
        "n_unlabelled_excluded": sum(1 for r in records if r["label_status"] == "unlabelled"),
        "boxes_per_class": {names[c]: n for c, n in sorted(cls_counter.items())},
        "dropped": dict(dropped),
        "records": usable,
        "excluded_unlabelled": [r["file"] for r in records
                                if r["label_status"] == "unlabelled"],
    }

    meta["banner"] = {
        "removed": True,
        "detector": "data/sem_bar.py (mid-tone collapse)",
        "median_rows_removed": int(np.median([r["banner_rows_removed"] for r in records]))
        if records else 0,
        "boxes_dropped_in_banner": dropped.get("box_in_banner", 0),
        "note": "every downstream stage reads data/curated - the banner does not exist there",
    }

    CURATED.mkdir(parents=True, exist_ok=True)
    (CURATED / "curated.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"[curate] {meta['n_raw_files']} raw -> {meta['n_curated']} curated "
          f"({dropped['augmented_copy']} augmented, {dropped['byte_duplicate']} duplicate dropped)")
    print(f"[curate] {meta['n_groups']} source groups, "
          f"{meta['n_groups_with_defects']} carry defects, {meta['n_boxes']} boxes")
    print(f"[curate] boxes per class: {meta['boxes_per_class']}")
    print(f"[curate] EXCLUDED {meta['n_unlabelled_excluded']} unlabelled images from "
          f"defect-class folders (empty labels != defect-free)")
    print(f"[curate] usable: {meta['n_annotated_images']} annotated + "
          f"{meta['n_true_backgrounds']} true backgrounds")
    b = meta["banner"]
    print(f"[curate] SEM banner stripped: median {b['median_rows_removed']} rows removed, "
          f"{b['boxes_dropped_in_banner']} boxes dropped as banner-only")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-3class", action="store_true",
                    help="keep raw 3-class box scheme (ablation A7)")
    curate(**vars(ap.parse_args()))
