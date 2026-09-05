"""
include_real=False must give a synthetic-only TRAIN split and leave val alone.

Val staying real is the load-bearing property: the checkpoint is selected on
val, and a synthetic val would measure how well the model fits its own
generator rather than how well it transfers.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

import yolo_export  # noqa: E402

POOL = ROOT / "data" / "synthetic" / "refined"
REGIME = "_test_synth_only"


@pytest.fixture
def cleanup():
    yield
    shutil.rmtree(yolo_export.EXPORTS / REGIME, ignore_errors=True)


@pytest.mark.skipif(not (POOL / "manifest.jsonl").exists(),
                    reason="refined pool not generated")
def test_synth_only_drops_real_train_keeps_real_val(cleanup):
    yaml = yolo_export.build(regime=REGIME, synth_pool="refined",
                             synth_ratio=0.01, include_real=False)
    m = json.loads((yaml.parent / "export_manifest.json").read_text())

    n_pool = sum(1 for l in (POOL / "manifest.jsonl").read_text().splitlines() if l.strip())
    assert m["include_real"] is False
    assert m["synthetic_images"] == m["counts"]["train"], "train must be synthetic only"
    assert m["counts"]["train"] <= int(n_pool * 0.01)
    # val is untouched: same count as a normal export of the val split
    n_val = len(json.loads((ROOT / "data/splits/val.json").read_text())["records"])
    assert m["counts"]["val"] == n_val

    # and no real image leaked into the train tree
    train_imgs = {p.name for p in (yaml.parent / "images" / "train").iterdir()}
    assert all(n.startswith("syn_") for n in train_imgs)


def test_synth_only_without_pool_is_refused():
    with pytest.raises(ValueError):
        yolo_export.build(regime=REGIME, synth_pool=None, include_real=False)
