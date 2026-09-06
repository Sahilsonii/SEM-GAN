"""Eight frames per pool with their labels drawn, for the eye. Run after generating."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(3)
for pool in ("layout_transfer", "augment_defective"):
    d = ROOT / "data/synthetic" / pool
    ents = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]
    ents = [e for e in ents if e["n_defects"] > 0]
    pick = [ents[i] for i in rng.choice(len(ents), 8, replace=False)]
    tiles = []
    for e in pick:
        img = cv2.imread(str(d / "images" / f"{e['stem']}.jpg"))
        H, W = img.shape[:2]
        for l in (d / "labels" / f"{e['stem']}.txt").read_text().splitlines():
            c, cx, cy, w, h = map(float, l.split())
            x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0) if c == 1 else (0, 128, 255), 1)
        img = cv2.resize(img, (512, 352))
        cv2.putText(img, f"{e['n_defects']} boxes", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(HERE / f"spot_{pool}.png"), grid)
    print("wrote", f"spot_{pool}.png")
