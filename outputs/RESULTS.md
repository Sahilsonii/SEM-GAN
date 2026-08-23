# Results

Generated 2026-08-24 04:21 from `outputs/master_results.csv` (0 runs).

All numbers are on the **validation** split. The test split is locked until the final evaluation stage and has not been read.

## Detection by regime

| regime | seeds | mAP50 | mAP50-95 |
|---|---|---|---|

## Per class

PbI2 is reported with the training support behind it. With single-digit training images its AP measures the annotation budget, not the method.

| experiment | class | AP50 | train images | interpretable |
|---|---|---|---|---|
| `real_only_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `real_only_yolo11s_seed42` | pinhole | 0.1008 | 45 | yes |
| `real_plus_synth_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `real_plus_synth_yolo11s_seed42` | pinhole | 0.1341 | 45 | yes |

## R1 - MicroDefectCV classical baseline

- mAP50 0.0261, mAP50-95 0.0105, P 0.063, R 0.160
- 5.11 s/image on CPU, 0 trainable parameters

| scale bin | n gt | recall | AP |
|---|---|---|---|
| T1_sub_stride | 26 | 0.115 | 0.1188 |
| T2_tiny | 301 | 0.113 | 0.0127 |
| T3_small | 213 | 0.211 | 0.0439 |
| T4_medium_up | 123 | 0.195 | 0.0680 |

## Caveats that travel with these numbers

- Validation carries 12 defect-bearing images, so run-to-run spread is wide; read the seed standard deviation before any difference.
- Defect sizes are in pixels. JPEG re-encoding stripped the FESEM pixel-size headers from all 440 source images and no TIFs survive, so no nanometre calibration exists.
- Renderer `severity` is a normalised simulation control, not a depth.
- The test split has not been read.
