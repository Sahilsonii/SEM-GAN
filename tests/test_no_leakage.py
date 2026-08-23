"""The gate. Nothing downstream is measurable if these fail."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
SPLITS = ROOT / "data" / "splits"

NAMES = ("train", "val", "test")
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def _load():
    return {n: json.loads((SPLITS / f"{n}.json").read_text(encoding="utf-8")) for n in NAMES}


def test_groups_are_disjoint():
    s = _load()
    for a, b in PAIRS:
        overlap = set(s[a]["groups"]) & set(s[b]["groups"])
        assert not overlap, f"{a}/{b} share {len(overlap)} source groups: {sorted(overlap)[:5]}"


def test_pixels_are_disjoint():
    s = _load()
    for a, b in PAIRS:
        ma = {r["md5"] for r in s[a]["records"]}
        mb = {r["md5"] for r in s[b]["records"]}
        assert not (ma & mb), f"{a}/{b} share {len(ma & mb)} byte-identical images"


def test_no_augmented_files_survive():
    s = _load()
    for n in NAMES:
        bad = [r["stem"] for r in s[n]["records"] if "_aug" in r["stem"]]
        assert not bad, f"{n} contains {len(bad)} augmented copies: {bad[:5]}"


def test_every_image_appears_once():
    s = _load()
    seen = [r["file"] for n in NAMES for r in s[n]["records"]]
    assert len(seen) == len(set(seen)), "an image appears in more than one split"


def test_test_split_has_enough_defects():
    """Guards against a split that is technically legal but useless."""
    s = _load()
    with_def = sum(1 for r in s["test"]["records"] if r["n_boxes"])
    boxes = sum(r["n_boxes"] for r in s["test"]["records"])
    assert with_def >= 15, f"only {with_def} test images carry defects"
    assert boxes >= 400, f"only {boxes} test boxes"


def test_original_splits_were_actually_leaky():
    """The audit underpinning contribution N0 - if this ever passes cleanly,
    the group_key logic has silently stopped working."""
    m = json.loads((SPLITS / "splits_manifest.json").read_text(encoding="utf-8"))
    audit = m["original_split_audit"]
    if not audit.get("available"):
        return
    assert audit["total_leaked_groups"] > 0, "expected the published splits to leak"
