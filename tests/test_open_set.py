"""
Open-set scoring: the two things that would silently produce a wrong number.

1. Background images must not count as known positives. They score 0.0 by
   construction, and there are 51 of them in test - labelling them positive
   put half the positives at the bottom of the ranking and crushed AUROC for a
   reason unrelated to open-set ability.
2. The box-level rates must be per ground-truth box and class-separated, so a
   detector that fires on every PbI2 particle gets false_alarm=1.0 regardless
   of how it labels them.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import open_set  # noqa: E402


def test_known_images_exclude_backgrounds(monkeypatch):
    recs = [
        {"file": "a.jpg", "n_boxes": 2, "boxes": [[1, .5, .5, .1, .1], [1, .2, .2, .1, .1]]},
        {"file": "bg.jpg", "n_boxes": 0, "boxes": []},                       # background
        {"file": "pb.jpg", "n_boxes": 1, "boxes": [[0, .5, .5, .1, .1]]},   # pbi2-only
        {"file": "mix.jpg", "n_boxes": 2, "boxes": [[0, .5, .5, .1, .1], [1, .3, .3, .1, .1]]},
    ]
    monkeypatch.setattr(open_set, "_records", lambda split: recs)
    monkeypatch.setattr(open_set, "_ensure_export",
                        lambda split: {"held_out_images": {"val": ["pb.jpg"]}})
    known, unknown = open_set.split_images("val")
    assert [p.name for p in known] == ["a.jpg", "mix.jpg"]
    assert [p.name for p in unknown] == ["pb.jpg"]


class _FakeBoxes:
    def __init__(self, xyxy):
        self._x = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.xyxy = self
    def __len__(self):
        return len(self._x)
    def cpu(self):
        return self
    def numpy(self):
        return self._x


class _FakeNet:
    """Fires exactly one box on every PbI2 GT, nothing on pinholes."""
    def __init__(self, recs, W, H):
        self.recs, self.W, self.H = {r["file"]: r for r in recs}, W, H
        self.calls = 0
    def predict(self, img, conf, verbose):
        r = list(self.recs.values())[self.calls]
        self.calls += 1
        preds = [open_set.xywhn_to_xyxy(b, self.W, self.H) for b in r["boxes"] if b[0] == open_set.PBI2]
        res = type("R", (), {})()
        res.boxes = _FakeBoxes(preds if preds else np.zeros((0, 4)))
        return [res]


def test_box_level_is_class_separated(monkeypatch):
    W = H = 100
    recs = [
        {"file": "x.jpg", "n_boxes": 3, "boxes": [[0, .2, .2, .1, .1], [0, .7, .7, .1, .1], [1, .5, .5, .1, .1]]},
        {"file": "y.jpg", "n_boxes": 2, "boxes": [[1, .3, .3, .1, .1], [0, .8, .2, .1, .1]]},
    ]
    monkeypatch.setattr(open_set, "_records", lambda split: recs)
    monkeypatch.setattr(open_set.cv2, "imread", lambda p: np.zeros((H, W, 3), np.uint8))
    net = _FakeNet(recs, W, H)

    out = open_set.box_level(net, "val", conf=0.25)
    assert out["n_pbi2_boxes"] == 3 and out["n_pinhole_boxes"] == 2
    assert out["unknown_box_false_alarm_rate"] == 1.0   # fired on every PbI2 box
    assert out["known_box_recall"] == 0.0                # never fired on a pinhole
