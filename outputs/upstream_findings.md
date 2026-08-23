# Upstream findings — MicroDefectCV 0.1.1

Found while benchmarking the package for this study. Both are worth fixing in
the package itself, since it is ours.

## 1. `sensitivity` is a dead parameter

`detect_defects(image, mode, sensitivity=1.5, min_area, return_intermediate)`
accepts `sensitivity` and documents it as *"Sensitivity multiplier (higher =
more detections)"*, but the name appears exactly twice in the source — once in
the signature, once in the docstring — and **never in the function body**.

Verified empirically on a curated FESEM image, `mode="pinhole"`, `min_area=30`:

| sensitivity | defects | area ratio |
|---|---|---|
| 0.1 | 2 | 0.000404 |
| 1.0 | 2 | 0.000404 |
| 2.5 | 2 | 0.000404 |
| 20.0 | 2 | 0.000404 |

Identical across two orders of magnitude. A caller tuning it gets silent no-ops.
Either wire it into the percentile thresholds in `_ModeProfile`, or remove it
from the public signature.

## 2. No quantitative benchmark shipped

The docs state no evaluation numbers. This project supplies the first: scored
against 4,578 expert boxes on a leakage-safe split, through the same COCO mAP
implementation used for the deep detectors (`eval/detection.py`). Results land
in `outputs/microdefectcv_baseline_val.json` and are worth folding back into the
package README.

## 3. Confirmed working as documented

`detect_defects` strips the FESEM metadata bar internally. On images where the
banner has already been removed (our `data/curated/`), that internal crop is
correctly a no-op — input rows equal output mask rows — so returned coordinates
need no compensation.
