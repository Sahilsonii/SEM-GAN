"""
Physics interpretation layer (plan section 13) - INTERPRETIVE, not a contribution.

Runs the three image-analysis modules over DETECTED defects and reports
distributions. Position in the argument is deliberate: Computer Vision produces
the detections, and this reads morphology off them. Nothing in the headline
result depends on this file.

EVERY QUANTITY HERE IS AN IMAGE-DERIVED INDEX, NOT A MEASUREMENT (plan Rule 7).
The renaming is not cosmetic - the original code reported these as physical
facts, and they are not:

  "Vertical Shunt Severity"  -> Relative Depression Index (RDI)
      Shape-from-shading integrates an intensity gradient and then min-max
      normalises to a HARD-CODED 500 nm. The 500 is an assumption in the
      constructor, not a calibration, so RDI is dimensionless and comparable
      only within an image.
  "Fatal shunts: N"          -> "N regions flagged by the RDI>=0.75 criterion"
      Nothing electrical was measured. RDI>=0.75 is a threshold on image
      intensity, and no electrical validation exists for this corpus.
  "GBDI = 0.9198"            -> Boundary Proximity Index (BPI)
      A mean Gaussian distance weight from PbI2 centroids to Canny ridges. It
      is a proximity statistic, NOT a passivation fraction, and 0.92 does not
      mean 92% of PbI2 is beneficial.
  "Voc loss = 4.75 mV"       -> Intensity-proxy dVoc index (arbitrary units)
      Applies dVoc = -(kT/q)ln(PL/PL0) to a PL SURROGATE built from image
      intensity. There is no photoluminescence measurement anywhere in this
      corpus, so the millivolts are not volts.

Additionally: the FESEM pixel-size headers did not survive JPEG re-encoding
(0 of 440 images retain them, no TIFs exist), so there is no nm-per-pixel
calibration available even in principle. Any depth in nm would be invented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interpret.boundary_index import GBDIAnalyzer
from interpret.depth_sfs import DepthEstimatorSfS
from interpret.pl_proxy import VirtualPLMapper

CURATED = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"

RDI_FLAG_THRESHOLD = 0.75          # threshold on an IMAGE index, not on a volt


def run(checkpoint: str | None = None, split: str = "val", conf: float = 0.10,
        device: str = "0", use_ground_truth: bool = False) -> dict:
    """Interpret either detected boxes (default) or the expert boxes.

    use_ground_truth=True is the cleaner scientific read - it removes detector
    error from the morphology statistics - but detections are what a deployed
    tool would actually feed this layer, so both are supported.
    """
    recs = [r for r in json.loads((SPLITS / f"{split}.json").read_text(encoding="utf-8"))["records"]
            if r["n_boxes"] > 0]

    net = None
    if not use_ground_truth:
        if checkpoint is None:
            raise ValueError("need --checkpoint unless --use-ground-truth")
        from ultralytics import YOLO
        net = YOLO(checkpoint)

    depth = DepthEstimatorSfS()
    gbdi = GBDIAnalyzer()
    pl = VirtualPLMapper()

    rdi, flagged, bpi, dvoc = [], 0, [], []
    n_boxes = 0

    for r in recs:
        im = cv2.imread(str(CURATED / r["file"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        if use_ground_truth:
            boxes = [list(b) for b in r["boxes"]]
        else:
            res = net.predict(im, conf=conf, verbose=False, device=device)[0]
            boxes = []
            if len(res.boxes):
                for (cx, cy, bw, bh), c in zip(res.boxes.xywhn.cpu().numpy(),
                                               res.boxes.cls.cpu().numpy().astype(int)):
                    boxes.append([int(c), float(cx), float(cy), float(bw), float(bh)])
        if not boxes:
            continue
        n_boxes += len(boxes)

        pinholes = [b for b in boxes if b[0] == 1]
        pbi2 = [b for b in boxes if b[0] == 0]

        for b in pinholes[:40]:                      # cap: SfS is not cheap
            try:
                d = depth.analyze_pinhole_shunt_risk(gray, np.array(b[1:5]))
            except Exception:
                continue
            v = float(d.get("vssi", 0.0))
            rdi.append(v)
            if v >= RDI_FLAG_THRESHOLD:
                flagged += 1

        if pbi2:
            try:
                g = gbdi.compute_gbdi(gray, np.array([b for b in pbi2]))
                bpi.append(float(g.get("gbdi_score", 0.0)))
            except Exception:
                pass

        try:
            p = pl.generate_virtual_pl_map(gray, np.array(boxes))
            dvoc.append(float(p.get("mean_voc_drop_mv", 0.0)))
        except Exception:
            pass

    def stats(v, name):
        if not v:
            return {"n": 0}
        a = np.array(v, float)
        return {"n": len(a), "mean": round(float(a.mean()), 4),
                "median": round(float(np.median(a)), 4),
                "p10": round(float(np.percentile(a, 10)), 4),
                "p90": round(float(np.percentile(a, 90)), 4)}

    out = {
        "source": "expert_boxes" if use_ground_truth else f"detections@conf{conf}",
        "checkpoint": checkpoint, "split": split,
        "images": len(recs), "boxes_interpreted": n_boxes,
        "relative_depression_index_RDI": stats(rdi, "RDI"),
        "regions_flagged_by_RDI_criterion": flagged,
        "RDI_flag_threshold": RDI_FLAG_THRESHOLD,
        "boundary_proximity_index_BPI": stats(bpi, "BPI"),
        "intensity_proxy_dVoc_index_au": stats(dvoc, "dVoc"),
        "UNITS_WARNING": (
            "All quantities are IMAGE-DERIVED INDICES, not measurements. RDI is "
            "dimensionless (shape-from-shading min-max normalised to a hard-coded "
            "500 nm assumption). Flagged regions are threshold crossings on that "
            "index, NOT electrically confirmed shunts. BPI is a mean Gaussian "
            "distance weight to Canny ridges, NOT a passivation fraction. The "
            "dVoc index is derived from an intensity-based PL surrogate - no "
            "photoluminescence was measured - so it is in arbitrary units, not "
            "millivolts. No nm-per-pixel calibration exists for this corpus: the "
            "FESEM pixel-size headers were destroyed by JPEG re-encoding in all "
            "440 source images and no TIF originals survive."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "gt" if use_ground_truth else "pred"
    (OUT / f"interpretation_{split}_{tag}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")

    print(f"[interpret] {out['source']}  {len(recs)} images, {n_boxes} boxes")
    print(f"  Relative Depression Index (RDI)   {out['relative_depression_index_RDI']}")
    print(f"  regions flagged at RDI>={RDI_FLAG_THRESHOLD}      {flagged}"
          f"   (image-index threshold crossings, NOT confirmed shunts)")
    print(f"  Boundary Proximity Index (BPI)    {out['boundary_proximity_index_BPI']}")
    print(f"  intensity-proxy dVoc index (a.u.) {out['intensity_proxy_dVoc_index_au']}")
    print("  NOTE: indices, not measurements - see UNITS_WARNING in the JSON")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--use-ground-truth", action="store_true")
    a = ap.parse_args()
    run(a.checkpoint, split=a.split, conf=a.conf,
        use_ground_truth=a.use_ground_truth)
