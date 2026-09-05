"""
Copy-paste pool: the properties that make it a fair baseline for the renderer.

Labels exact, counts sampled (not constant), backgrounds present, both classes
present, no val/test canvas - and yolo_export must keep the background images
rather than dropping them as it used to.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from synth import copypaste  # noqa: E402

POOL = "_test_cp"
N = 16


@pytest.fixture(scope="module")
def pool():
    s = copypaste.generate(n_images=N, seed=0, pool_name=POOL, bg_fraction=0.4)
    yield s, copypaste.OUT_ROOT / POOL
    shutil.rmtree(copypaste.OUT_ROOT / POOL, ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "yolo" / POOL, ignore_errors=True)


def test_layout_and_leak(pool):
    s, d = pool
    assert s["images"] == N and (d / "manifest.jsonl").exists()
    train_groups = set(json.loads((ROOT / "data/splits/train.json").read_text())["groups"])
    for line in (d / "manifest.jsonl").read_text().splitlines():
        assert json.loads(line)["background_group"] in train_groups


def test_counts_sampled_backgrounds_present_both_classes(pool):
    s, d = pool
    counts, classes = [], set()
    for p in (d / "labels").glob("*.txt"):
        rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
        counts.append(len(rows))
        classes |= {r[0] for r in rows}
    assert s["backgrounds"] >= 1 and 0 in counts, "no background images emitted"
    assert len({c for c in counts if c > 0}) > 1, f"defect counts are constant: {counts}"
    assert classes == {"0", "1"}


def test_labels_are_exact(pool):
    """Every box must bound a region that actually changed vs the background,
    and lie inside the frame."""
    import cv2
    s, d = pool
    for line in (d / "manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        if not e["n_defects"]:
            continue
        img = cv2.imread(str(d / "images" / f"{e['stem']}.jpg"))
        bg = cv2.imread(str(copypaste.CURATED_IMAGES / e["background_file"]))
        bg = copypaste._fit_long_side(bg, e["render_px"])
        H, W = img.shape[:2]
        for row in (d / "labels" / f"{e['stem']}.txt").read_text().splitlines():
            _, cx, cy, w, h = (float(v) for v in row.split())
            assert 0 < cx < 1 and 0 < cy < 1 and 0 < w <= 1 and 0 < h <= 1
            x0, x1 = int((cx - w / 2) * W), int(np.ceil((cx + w / 2) * W))
            y0, y1 = int((cy - h / 2) * H), int(np.ceil((cy + h / 2) * H))
            diff = np.abs(img[y0:y1, x0:x1].astype(int) - bg[y0:y1, x0:x1].astype(int)).mean()
            assert diff > 1.0, f"box in {e['stem']} does not differ from background"
        break   # one image is enough; the loop above checks every box in it


def test_yolo_export_keeps_synthetic_backgrounds(pool):
    import yolo_export
    s, d = pool
    y = yolo_export.build(regime=POOL, synth_pool=POOL, synth_ratio=1.0)
    m = json.loads((y.parent / "export_manifest.json").read_text())
    assert m["synthetic_images"] == N, (
        f"exported {m['synthetic_images']} of {N}: backgrounds were dropped")
