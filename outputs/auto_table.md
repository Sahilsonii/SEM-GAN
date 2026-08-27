# Results

Generated 2026-08-27 12:47 from `outputs/master_results.csv` (9 runs).

All numbers are on the **validation** split. The test split is locked until the final evaluation stage and has not been read.

## Detection by regime

| regime | seeds | mAP50 | mAP50-95 |
|---|---|---|---|
| `real_only` | 3 | 0.0630 ± 0.0107 | 0.0263 ± 0.0054 |
| `scale_002` | 1 | 0.1020 ± 0.0000 | 0.0427 ± 0.0000 |
| `scale_005` | 3 | 0.1099 ± 0.0130 | 0.0461 ± 0.0057 |
| `scale_010` | 1 | 0.1010 ± 0.0000 | 0.0442 ± 0.0000 |
| `scale_025` | 1 | 0.1002 ± 0.0000 | 0.0437 ± 0.0000 |

## Per class

PbI2 is reported with the training support behind it. With single-digit training images its AP measures the annotation budget, not the method.

| experiment | class | AP50 | train images | interpretable |
|---|---|---|---|---|
| `real_only_yolo11s_seed1` | pbi2 | 0.0000 | 5 | **no** |
| `real_only_yolo11s_seed1` | pinhole | 0.1448 | 45 | yes |
| `real_only_yolo11s_seed2` | pbi2 | 0.0000 | 5 | **no** |
| `real_only_yolo11s_seed2` | pinhole | 0.1303 | 45 | yes |
| `real_only_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `real_only_yolo11s_seed42` | pinhole | 0.1025 | 45 | yes |
| `real_plus_controlled_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `real_plus_controlled_yolo11s_seed42` | pinhole | 0.0155 | 45 | yes |
| `scale_002_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `scale_002_yolo11s_seed42` | pinhole | 0.2040 | 45 | yes |
| `scale_005_yolo11s_seed1` | pbi2 | 0.0000 | 5 | **no** |
| `scale_005_yolo11s_seed1` | pinhole | 0.2030 | 45 | yes |
| `scale_005_yolo11s_seed2` | pbi2 | 0.0000 | 5 | **no** |
| `scale_005_yolo11s_seed2` | pinhole | 0.2067 | 45 | yes |
| `scale_005_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `scale_005_yolo11s_seed42` | pinhole | 0.2499 | 45 | yes |
| `scale_010_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `scale_010_yolo11s_seed42` | pinhole | 0.2020 | 45 | yes |
| `scale_025_yolo11s_seed42` | pbi2 | 0.0000 | 5 | **no** |
| `scale_025_yolo11s_seed42` | pinhole | 0.2003 | 45 | yes |

## Synthetic scaling

| synthetic ratio | mAP50 | mAP50-95 |
|---|---|---|
| 2% | 0.1020 | 0.0427 |
| 5% | 0.1249 | 0.0525 |
| 10% | 0.1010 | 0.0442 |
| 25% | 0.1002 | 0.0437 |

## R1 - MicroDefectCV classical baseline

- mAP50 0.0270, mAP50-95 0.0112, P 0.058, R 0.114
- 3.73 s/image on CPU, 0 trainable parameters

| scale bin | n gt | recall | AP |
|---|---|---|---|
| T1_sub_stride | 183 | 0.000 | 0.0000 |
| T2_tiny | 233 | 0.129 | 0.0457 |
| T3_small | 150 | 0.247 | 0.0670 |
| T4_medium_up | 74 | 0.081 | 0.0303 |

## Caveats that travel with these numbers

- Validation carries 12 defect-bearing images, so run-to-run spread is wide; read the seed standard deviation before any difference.
- Defect sizes are in pixels. JPEG re-encoding stripped the FESEM pixel-size headers from all 440 source images and no TIFs survive, so no nanometre calibration exists.
- Renderer `severity` is a normalised simulation control, not a depth.
- The test split has not been read.
