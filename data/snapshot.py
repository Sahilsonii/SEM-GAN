"""
Stage 0 - vendor an immutable copy of the source FESEM corpus.

The dataset lives outside the repo and every legacy script reached it by
hard-coded absolute path. That is unreproducible. We copy it once, hash every
file, and never write to it again; `verify()` is what the test suite calls.

The original train/val/test.txt are copied in deliberately: they are the
evidence for the leakage audit (they are NOT used for training).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

SOURCE = Path(r"C:/Users/Sahil/Downloads/SEM-Annotation/balanced_dataset")
DEST = Path(__file__).resolve().parent / "raw_snapshot"
MANIFEST = DEST / "SNAPSHOT.json"

EXPECTED_FILES = 440
EXPECTED_UNIQUE = 415


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _image_files(root: Path) -> list[Path]:
    return sorted(p for p in (root / "images").rglob("*") if p.is_file())


def create(force: bool = False) -> dict:
    """Copy SOURCE -> DEST and write the checksum manifest."""
    if MANIFEST.exists() and not force:
        print(f"[snapshot] manifest exists, skipping copy ({MANIFEST})")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    if not SOURCE.exists():
        raise FileNotFoundError(f"source corpus not found: {SOURCE}")

    DEST.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        src, dst = SOURCE / sub, DEST / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for extra in ("data.yaml", "train.txt", "val.txt", "test.txt"):
        if (SOURCE / extra).exists():
            shutil.copy2(SOURCE / extra, DEST / extra)

    files = _image_files(DEST)
    entries = {}
    for p in files:
        rel = p.relative_to(DEST).as_posix()
        lbl = DEST / "labels" / Path(rel).relative_to("images").with_suffix(".txt")
        n_boxes = 0
        if lbl.exists():
            n_boxes = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        entries[rel] = {"md5": md5(p), "bytes": p.stat().st_size, "boxes": n_boxes}

    manifest = {
        "source": str(SOURCE),
        "n_files": len(entries),
        "n_unique_md5": len({e["md5"] for e in entries.values()}),
        "n_boxes": sum(e["boxes"] for e in entries.values()),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[snapshot] copied {manifest['n_files']} files "
          f"({manifest['n_unique_md5']} unique) -> {DEST}")
    return manifest


def verify() -> dict:
    """Re-hash the snapshot against its manifest. Raises on any drift."""
    if not MANIFEST.exists():
        raise FileNotFoundError("no snapshot manifest; run create() first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    on_disk = {p.relative_to(DEST).as_posix() for p in _image_files(DEST)}
    recorded = set(manifest["files"])
    if on_disk != recorded:
        missing, extra = recorded - on_disk, on_disk - recorded
        raise AssertionError(f"snapshot drift: {len(missing)} missing, {len(extra)} extra")

    bad = [rel for rel, e in manifest["files"].items() if md5(DEST / rel) != e["md5"]]
    if bad:
        raise AssertionError(f"checksum mismatch on {len(bad)} file(s): {bad[:5]}")

    assert manifest["n_files"] == EXPECTED_FILES, \
        f"expected {EXPECTED_FILES} files, manifest has {manifest['n_files']}"
    assert manifest["n_unique_md5"] == EXPECTED_UNIQUE, \
        f"expected {EXPECTED_UNIQUE} unique, manifest has {manifest['n_unique_md5']}"

    print(f"[snapshot] verified {manifest['n_files']} files, "
          f"{manifest['n_unique_md5']} unique, {manifest['n_boxes']} boxes")
    return manifest


if __name__ == "__main__":
    create()
    verify()
