"""
master_results.csv must hold exactly one row per experiment.

It was append-only, so re-running an experiment - after a crash, a resume, or a
protocol change - added a second row for the same deterministic exp_id. The file
reached 14 rows for 10 experiments, and real_only_seed42 appeared twice with
DIFFERENT values (0.0583 from a 300-epoch run, 0.0513 from a 100-epoch run).
Grouping by regime then reported n=4 over "seeds 1,2,42,42" and averaged a stale
300-epoch result into a 100-epoch seed mean. Silent, and the kind of thing that
survives into a published table.
"""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_detector as td


def _row(exp_id, mAP50="0.1", git="T", epochs="100", seed="0"):
    return {"exp_id": exp_id, "regime": exp_id.split("_yolo")[0], "seed": seed,
            "model": "yolo11s", "p2_head": "False", "synth_pool": "", "synth_ratio": "1.0",
            "epochs": epochs, "imgsz": "640", "batch": "8", "mAP50": mAP50,
            "mAP50_95": "0.05", "precision": "0.1", "recall": "0.1",
            "params_M": "9.41", "train_seconds": "1", "git_sha": git}


@pytest.fixture
def csv_path(tmp_path, monkeypatch):
    p = tmp_path / "master_results.csv"
    monkeypatch.setattr(td, "MASTER_CSV", p)
    return p


def _read(p):
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_rerunning_an_experiment_replaces_its_row(csv_path):
    td._append_master(_row("real_only_yolo11s_seed42", mAP50="0.0583", epochs="300"))
    td._append_master(_row("real_only_yolo11s_seed42", mAP50="0.0513", epochs="100"))
    rows = _read(csv_path)
    assert len(rows) == 1, "re-run appended instead of replacing - the original bug"
    assert rows[0]["mAP50"] == "0.0513", "kept the stale value instead of the newest"
    assert rows[0]["epochs"] == "100"


def test_distinct_experiments_accumulate(csv_path):
    for s in ("1", "2", "42"):
        td._append_master(_row(f"real_only_yolo11s_seed{s}", seed=s))
    assert len(_read(csv_path)) == 3


def test_no_duplicate_exp_ids_ever(csv_path):
    for _ in range(4):
        td._append_master(_row("scale_005_yolo11s_seed1"))
        td._append_master(_row("scale_005_yolo11s_seed2"))
    ids = [r["exp_id"] for r in _read(csv_path)]
    assert len(ids) == len(set(ids)), f"duplicates present: {ids}"


def test_header_written_once(csv_path):
    td._append_master(_row("a_yolo11s_seed1"))
    td._append_master(_row("b_yolo11s_seed1"))
    text = csv_path.read_text(encoding="utf-8")
    assert text.count("exp_id,regime") == 1


def test_report_dedups_on_read_too(tmp_path, monkeypatch):
    """A hand-edited or legacy file with duplicates must not skew the aggregate."""
    import make_report
    out = tmp_path
    monkeypatch.setattr(make_report, "OUT", out)
    p = out / "master_results.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=td.MASTER_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow(_row("real_only_yolo11s_seed42", mAP50="0.0583", epochs="300"))
        w.writerow(_row("real_only_yolo11s_seed42", mAP50="0.0513", epochs="100"))
    rows = make_report._rows()
    assert len(rows) == 1, "make_report averaged a duplicated experiment"
    assert rows[0]["mAP50"] == "0.0513"


def test_live_file_has_no_duplicates():
    """Guards the real results file in the repo."""
    p = ROOT / "outputs" / "master_results.csv"
    if not p.exists():
        pytest.skip("no results yet")
    ids = [r["exp_id"] for r in _read(p) if r.get("exp_id")]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate exp_ids in the live results file: {dupes}"
