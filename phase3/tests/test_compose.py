"""
The properties that make the two composition pools a fair test:
layout_transfer reproduces a real layout (same class, centre within jitter);
augment_defective keeps the real labels verbatim and adds the promised extras;
neither uses a val/test canvas; yolo_export keeps both classes and the
backgrounds.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "phase3"))

import synth_compose as sc   # noqa: E402

N = 12


def _rows(p):
    return [[float(v) for v in l.split()] for l in p.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def lt():
    s = sc.generate_layout_transfer(n_images=N, seed=0, pool_name="_test_lt", bg_fraction=0.34)
    yield s, sc.OUT_ROOT / "_test_lt"
    shutil.rmtree(sc.OUT_ROOT / "_test_lt", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "yolo" / "_test_lt", ignore_errors=True)


@pytest.fixture(scope="module")
def ad():
    s = sc.generate_augment_defective(n_images=N, seed=0, pool_name="_test_ad")
    yield s, sc.OUT_ROOT / "_test_ad"
    shutil.rmtree(sc.OUT_ROOT / "_test_ad", ignore_errors=True)


def test_layout_transfer_reproduces_real_layouts(lt):
    s, d = lt
    lays = {l["file"]: l for l in sc.layouts("train")}
    n_with = 0
    for line in (d / "manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        if not e["n_defects"]:
            continue
        n_with += 1
        lay = lays[e["layout_file"]]
        got = _rows(d / "labels" / f"{e['stem']}.txt")
        assert len(got) == len(lay["boxes"]), "every layout box must be reproduced"
        tol = 4 * sc.JITTER + 0.02      # 4 sd of jitter plus edge clamping slack
        for g, (c, cx, cy, w, h) in zip(got, lay["boxes"]):
            if e["flipped"]:
                cx = 1.0 - cx
            assert int(g[0]) == int(c)
            assert abs(g[1] - cx) < tol and abs(g[2] - cy) < tol, "centre drifted from the layout"
    assert n_with >= 1 and s["backgrounds"] >= 1


def test_augment_defective_keeps_real_labels(ad):
    s, d = ad
    for line in (d / "manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        real = [list(map(float, b)) for b in
                next(r for r in sc._load_split("train")["records"]
                     if r["file"] == e["background_file"])["boxes"]]
        got = _rows(d / "labels" / f"{e['stem']}.txt")
        import numpy as np
        assert np.allclose(got[:len(real)], real, atol=1e-6), "real labels must be verbatim"
        assert len(got) == len(real) + e["n_extra"]
        assert e["n_extra"] == max(1, round(sc.EXTRA_FRACTION * len(real)))
    assert s["backgrounds"] == 0


def test_no_holdout_canvas(lt, ad):
    train = set(json.loads((ROOT / "data/splits/train.json").read_text())["groups"])
    for _, d in (lt, ad):
        for line in (d / "manifest.jsonl").read_text().splitlines():
            assert json.loads(line)["background_group"] in train


def test_yolo_export_keeps_both_classes_and_backgrounds(lt):
    import yolo_export
    s, d = lt
    y = yolo_export.build(regime="_test_lt", synth_pool="_test_lt", synth_ratio=1.0)
    m = json.loads((y.parent / "export_manifest.json").read_text())
    assert m["synthetic_images"] == N
    assert all(v > 0 for v in m["synthetic_boxes_per_class"].values())
