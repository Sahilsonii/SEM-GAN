"""Re-hash the vendored corpus. Must pass before anything is deleted."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import snapshot


def test_snapshot_matches_manifest():
    m = snapshot.verify()
    assert m["n_files"] == snapshot.EXPECTED_FILES
    assert m["n_unique_md5"] == snapshot.EXPECTED_UNIQUE
    assert m["n_boxes"] == 9357
