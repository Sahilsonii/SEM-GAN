"""
Diagnostic: where do the extra predictions of the augmented arms land?

Every synthetic/augmented arm in this project predicts MORE boxes on test than
real-only and has lower precision. Two candidate reasons: (a) the pasted
content teaches the detector to fire on things that are not defects, so false
alarms rise on defect-free films; (b) the arm just fires more everywhere.
This counts predictions per image separately on the 51 defect-free test
images and the 22 defect-bearing ones, for the four 8,300-step arms, three
seeds each. Read-only on already-scored checkpoints; no selection happens here.

Writes outputs/phase3_background_fp.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.detection import load_net   # noqa: E402

EXPORT = ROOT / "data/yolo/openset_probe"      # built with include_test; backgrounds present
ARMS = {
    "real_only": "real_only_steps8300_yolo11s_seed{s}",
    "balanced_rendered": "scale_005_bal_steps8300_yolo11s_seed{s}",
    "layout_transfer": "layout_transfer_steps8300_yolo11s_seed{s}",
    "augment_defective": "augment_defective_steps8300_yolo11s_seed{s}",
}
SEEDS = (1, 2, 42)
CONFS = (0.05, 0.25)


def main():
    imgs = sorted((EXPORT / "images/test").glob("*.jpg"))
    lbl = EXPORT / "labels/test"
    # the open-set export drops PbI2-only images, so read background-ness from the
    # curated split instead of the export's labels
    test = json.loads((ROOT / "data/splits/test.json").read_text())["records"]
    is_bg = {Path(r["file"]).name: r["n_boxes"] == 0 for r in test}
    bg = [p for p in imgs if is_bg.get(p.name)]
    fg = [p for p in imgs if p.name in is_bg and not is_bg[p.name]]
    print(f"{len(bg)} defect-free and {len(fg)} defect-bearing test images")

    out = {"n_background_images": len(bg), "n_defect_images": len(fg), "arms": {}}
    for arm, pat in ARMS.items():
        rows = []
        for s in SEEDS:
            ck = ROOT / "experiments" / pat.format(s=s) / "run/weights/best.pt"
            net = load_net(str(ck))
            rec = {}
            for name, paths in (("background", bg), ("defect", fg)):
                res = net.predict([str(p) for p in paths], conf=0.001, imgsz=640,
                                  verbose=False, batch=8)
                confs = [r.boxes.conf.cpu().numpy() for r in res]
                for c in CONFS:
                    per_img = [int((x >= c).sum()) for x in confs]
                    rec[f"{name}_preds_per_image@{c}"] = float(np.mean(per_img))
                    rec[f"{name}_images_with_any@{c}"] = float(np.mean([n > 0 for n in per_img]))
            rows.append(rec)
        out["arms"][arm] = {k: (float(np.mean([r[k] for r in rows])),
                                float(np.std([r[k] for r in rows], ddof=1))) for k in rows[0]}
        r = out["arms"][arm]
        print(f"{arm:18s} bg preds/img @0.05 {r['background_preds_per_image@0.05'][0]:6.1f}  "
              f"@0.25 {r['background_preds_per_image@0.25'][0]:5.2f}   "
              f"defect preds/img @0.05 {r['defect_preds_per_image@0.05'][0]:6.1f}  "
              f"bg imgs with any @0.25 {100 * r['background_images_with_any@0.25'][0]:4.0f}%")
    (ROOT / "outputs/phase3_background_fp.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
