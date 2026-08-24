# When Does Synthetic Microscopy Data Help?

Controllable generation and scale-resolved tiny-defect detection in low-data
perovskite FESEM imagery.

This is **Paper 2** of an existing research line. Paper 1 established the corpus
and taxonomy:

> Ansari, Z. A., **Soni, S.**, Fatima, S., Siddiqui, S., & Prasad, P. V. H. (2025).
> *A multi-model deep learning framework for SEM-based defect detection in FAPbI₃
> perovskite thin films.* **Scientific Reports 15**, 41909.
> DOI [10.1038/s41598-025-25848-x](https://doi.org/10.1038/s41598-025-25848-x)

and its Future Work section — dataset expansion, self-supervised methods, hybrid
models, defects below 100 nm — is what this repository sets out to deliver. The
second predecessor artifact is [**MicroDefectCV**](https://pypi.org/project/microdefectcv/)
(PyPI, MIT), used here as an independent evaluator and classical baseline.

---

## Quick start

```bash
py -3.10 run_full_pipeline.py --check          # environment + what is runnable
py -3.10 run_full_pipeline.py --stage 0-8      # everything implemented today
py -3.10 -m pytest tests/ -q                   # the guarantees
```

Requires **Python 3.10** (`py -3.10`), which is where CUDA torch lives on this
machine. The default `python` on PATH is 3.14 with CPU-only torch.

`run_full_pipeline.py` is the single entry point for the whole study — there is
no separate overnight/bulk runner. Every stage is invokable individually
(`--stage 5`), as a range (`--stage 0-4`), or all at once (`--stage all`); a
failed stage's dependency is reported by `--check` rather than crashing
mid-run.

## Pipeline stages

| # | Stage | Status | What it does |
|---|---|---|---|
| 0 | `snapshot` | ready | Vendors the external corpus into `data/raw_snapshot/`, write-once, MD5-manifested |
| 1 | `dataset` | ready | Curates + strips the FESEM banner + builds leakage-safe grouped splits |
| 2 | `bins` | ready | Profiles defects against pre-registered scale bins |
| 3 | `synth` | ready | Renders the label-exact synthetic pool + counterfactual severity ladder. `--per-class N` switches to balanced N-per-class bulk generation |
| 4 | `classical` | ready | MicroDefectCV zero-training baseline, tuned on val |
| 5 | `refiner` | ready | Conditional GAN texture refiner, FFT-on and FFT-off checkpoints (ablation A1 / H2) |
| 6 | `quality` | ready | Repaints the renderer pool's geometry with both refiner checkpoints → `data/synthetic/refined{,_nofft}` |
| 7 | `detector` | ready | E-A/E-D matrix, or `--ratios` for the scaling ladder, or `--refined` to train on the GAN-refined pool. `--target-steps` caps wall-clock at bulk scale |
| 8 | `uncertain` | ready | Open-set (PbI₂ held out) AUROC/AUPR/FPR@95TPR + calibration (ECE, Brier, risk-coverage) |
| 9 | `final` | **manual only** | `eval/final_eval.py`, gated behind `--i-am-sure --confirm "I am done tuning"` — reads the locked test split exactly once |

### Bulk scale (5000 images/class)

```bash
py -3.10 run_full_pipeline.py --stage 3 --per-class 5000        # ~100 min, measured 0.61s/image
py -3.10 run_full_pipeline.py --stage 5                          # refiner, ~30 min/checkpoint
py -3.10 run_full_pipeline.py --stage 6                          # apply refiner to the bulk pool
py -3.10 run_full_pipeline.py --stage 7 --refined --target-steps 6000
```

`--target-steps` matters here: epoch time scales with training-set size, so a
fixed epoch count that takes 11 minutes on 160 real images takes **hours** on
10,160 (5000/class + real). `--target-steps` derives epochs from a step budget
instead, so a run's cost stays bounded as the synthetic pool grows rather than
scaling linearly with it (measured: ~0.229 s/step on this GPU).

## What the corpus actually is

The headline "440 images" does not survive contact with the files:

| | |
|---|---|
| Files in the source corpus | 440 |
| Unique by MD5 | 415 |
| After dropping `_aug###` geometric copies | **338** |
| Unique **source groups** | **267** |
| Groups carrying any defect box | **84** |
| Expert boxes (banner-stripped) | **4,578** |

Split at source-group level: **train** 160 groups / 197 images / 3,176 boxes ·
**val** 39 / 59 / 663 · **test** 68 / 82 / 739.

### Why the published splits could not be used

Grouping files by the specimen they came from, the original `train/val/test.txt`
share **22 groups between train and val, 18 between train and test, 8 between val
and test** — 36 distinct source groups leaked. Two mechanisms: `_aug###` flips of
one image landing on both sides, and the same specimen filed under two class
folders. `data/splits.py` audits this and the result is recorded in
`splits_manifest.json`.

### Why the metadata banner is stripped

All 338 images carry a burned-in FESEM instrument banner over the bottom ~8.2%
of their height. It is removed once, corpus-wide, at curation, and every
downstream stage reads `data/curated/` where it does not exist. Left in, it is a
context cue present in every image, a magnet for saliency maps, and exactly the
kind of high-contrast blob a morphological filter is built to find. 27
annotations that lived entirely inside the banner were dropped with it.

### Why defect scale is the framing

Against a 640 px detector input, **~66% of test boxes are under 16 px** and the
median defect is ~11 px. Bins are anchored to detector strides (P3 = 8, P4 = 16,
P5 = 32), not to quantiles of the data, and are committed in
`configs/tiny_defect_bins.yaml` **before** any detector exists.

> No nanometre-scale claim is possible here. JPEG re-encoding stripped the FESEM
> `Image Pixel Size` headers from all 440 images and no `.tif` originals survive,
> so there is no pixel-size calibration. Every size is reported in pixels, and
> renderer `severity` is a normalised simulation control — never a depth.

## Guarantees (enforced by tests, not by convention)

```
tests/test_snapshot_integrity.py   corpus matches its manifest, 440/415/9357
tests/test_no_leakage.py           groups, pixels and augmentations are disjoint
tests/test_renderer.py             boxes bound their masks; classes are exact
```

- **The test split is written once by stage 1 and not read again until stage 9.**
- **Renderer priors come from expert boxes on train only** — MicroDefectCV never
  enters a training loss, so it remains an independent evaluator (the firewall).
- **Synthetic canvases come only from train backgrounds**, asserted at generation.

## Layout

```
data/     snapshot.py build_dataset.py splits.py sem_bar.py
          raw_snapshot/ (write-once)  curated/  splits/  synthetic/
synth/    renderer.py  generate.py
eval/     detection.py  tiny_defect.py  spectrum.py
          microdefectcv_baseline.py  foundation_features.py
interpret/ depth_sfs.py  boundary_index.py  pl_proxy.py
tests/    the guarantees above
outputs/  metrics, figures, master results
docs/     dissertation documents
```

`interpret/` modules are downstream scientific reading of CV output, not
contributions. Their quantities are image-derived indices with no electrical
validation, and are named accordingly.
