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

Full numbers, caveats and corrections: **[`outputs/RESULTS.md`](outputs/RESULTS.md)**.

---

## Headline result

Adding **5% GAN-refined synthetic data** to the real training set, evaluated on a
locked test split never used for tuning:

| metric | real only | + 5% synthetic | ratio | Welch *p* |
|---|---|---|---|---|
| mAP50 | 0.0688 ± 0.0105 | **0.1333 ± 0.0128** | **1.94×** | 0.0028 |
| mAP50-95 | 0.0261 ± 0.0053 | **0.0521 ± 0.0065** | **2.00×** | 0.0065 |

n = 3 seeds (1, 2, 42), YOLO11s @ 640 px. The two seed ranges are fully
disjoint — the worst synthetic seed beats the best real-only seed.

The gain **decreases monotonically with defect size**, which is the pattern the
protocol pre-registered as its strongest available finding:

| bin | n gt | AP50 real | AP50 +5% | Δ |
|---|---|---|---|---|
| T1 sub-stride (<8 px) | 249 | 0.0096 | 0.0205 | **+114%** |
| T2 tiny (8–16 px) | 561 | 0.0561 | 0.1410 | **+151%** |
| T3 small (16–32 px) | 340 | 0.1084 | 0.1971 | **+82%** |
| T4 medium+ (≥32 px) | 181 | 0.1187 | 0.1523 | **+28%** |

> ### ⚠ One caveat travels with this table
>
> Both arms ran 100 epochs, but the synthetic arm has 660 training images
> against 160, so it took **8,300 gradient steps against 2,000**. Adding data to
> a fixed epoch budget silently adds compute, so the claim that stands without
> qualification is *at a matched **epoch** budget*. The step-matched control is
> specified in `outputs/RESULTS.md` §13 and **has not been run yet**. It is the
> one outstanding experiment that could require rewriting this section.

Three further findings, none of them flattering, all reported:

- **The Fourier discriminator branch does not help detection.** It cut the
  frequency-domain gap to real by 62% and downstream mAP50 did not move
  (0.1015 FFT-on vs 0.1065 FFT-off, well inside seed noise). A distribution
  distance that failed to predict utility.
- **The classical baseline never detects a sub-stride defect.** MicroDefectCV
  gets T1 recall 0.000 with zero training, contradicting the prior expectation
  that stride-free morphological filtering would win there.
- **10 of 21 perturbations degrade the detector without it noticing.** 2%
  Gaussian noise costs 96% of mAP50 while mean confidence falls 13.5%.

---

## Quick start

```bash
py -3.10 run_full_pipeline.py --check          # environment + stage status
py -3.10 run_full_pipeline.py --stage 0-8      # data -> pools -> refiner -> detector
py -3.10 run_full_pipeline.py --stage 10-16    # all analyses
py -3.10 -m pytest tests/ -q                   # 31 guarantees
```

Requires **Python 3.10** (`py -3.10`), which is where CUDA torch lives on this
machine. The default `python` on PATH is 3.14 with CPU-only torch.

`run_full_pipeline.py` is the single entry point — there is no separate
overnight or bulk runner. Every stage is invokable individually (`--stage 5`),
as a range (`--stage 0-4`), or all at once. A missing dependency is reported by
`--check` rather than crashing mid-run.

Hardware this was built and measured on: one **RTX 3050 Ti laptop GPU, 4 GB**.
That constraint shapes several choices (YOLO11s over 11m, 192 px refiner
patches, batch 8, AMP non-optional) and is stated wherever it does.

## Pipeline stages

| # | Stage | What it does |
|---|---|---|
| 0 | `snapshot` | Vendors the external corpus into `data/raw_snapshot/`, write-once, MD5-manifested |
| 1 | `dataset` | Curates, strips the FESEM banner, builds leakage-safe grouped splits |
| 2 | `bins` | Profiles defects against pre-registered scale bins |
| 3 | `synth` | Renders the label-exact synthetic pool + counterfactual severity ladder. `--per-class N` for balanced bulk generation |
| 4 | `classical` | MicroDefectCV zero-training baseline, tuned on val |
| 5 | `refiner` | Conditional GAN texture refiner; FFT-on and FFT-off checkpoints (H2 / ablation A1) |
| 6 | `quality` | Repaints the renderer pool with both refiner checkpoints → `data/synthetic/refined{,_nofft}` |
| 7 | `detector` | Detector matrix; `--ratios` for the scaling ladder, `--refined` for the GAN pool, `--model rtdetr-l` for the transformer arm |
| 8 | `uncertain` | Open-set (PbI₂ held out) AUROC/AUPR/FPR@95TPR + calibration |
| 9 | `final` | **manual only** — `eval/final_eval.py`, gated behind `--i-am-sure --confirm "I am done tuning"`; reads the locked test split exactly once |
| 10 | `domaingap` | Domain gap real vs synthetic at 4 levels: pixel, frequency, morphology, feature |
| 11 | `counterfact` | Counterfactual severity monotonicity on a fixed background |
| 12 | `robustness` | Perturbation sweep — does confidence track degradation? |
| 13 | `failures` | Pre-declared failure taxonomy + saved crops |
| 14 | `explain` | Occlusion attribution faithfulness |
| 15 | `interpret` | Image-derived morphology indices (interpretive, not a contribution) |
| 16 | `report` | Assembles `outputs/auto_table.md` |

Stage 9 sits deliberately outside the `--stage all` set so no automated run can
touch the locked test split.

### Bulk scale (5000 images/class)

```bash
py -3.10 run_full_pipeline.py --stage 3 --per-class 5000   # ~100 min, 0.61 s/image
py -3.10 run_full_pipeline.py --stage 5                    # refiner, ~30 min/checkpoint
py -3.10 run_full_pipeline.py --stage 6                    # apply refiner to the bulk pool
py -3.10 run_full_pipeline.py --stage 7 --refined --ratios 0.05
```

**Prefer a ratio to a raw step budget.** Epoch time scales with training-set
size, so a fixed epoch count that takes 11 minutes on 160 real images takes
hours on 10,160. `--target-steps` bounds that, but the measured optimum here is
a **5% synthetic ratio** — 25% used 4× the steps of 5% and scored *worse*, so
throwing the whole bulk pool at the detector is the failure mode this project
already made once.

If you do use `--target-steps`, raise `--patience` above the derived epoch count.
At 160 images an 8,300-step budget is 415 epochs and the default patience of 30
will early-stop it, silently voiding whatever comparison it was matching.

## What the corpus actually is

The headline "440 images" does not survive contact with the files:

| | |
|---|---|
| Files in the source corpus | 440 |
| Unique by MD5 | 415 |
| After dropping `_aug###` geometric copies | **338** |
| Usable after excluding unlabelled | **273** |
| Unique **source groups** | **211** |
| Expert boxes (banner-stripped) | **4,578** |

Split at source-group level:

| split | groups | images | boxes | defect-bearing images |
|---|---|---|---|---|
| train | 126 | 160 | 2,607 | 50 |
| val | 31 | 40 | 640 | **12** |
| test | 54 | 73 | 1,331 | 22 |

Validation carries only 12 defect images, so run-to-run spread is wide — the
real-only baseline alone spanned 36% across three seeds. Read the seed standard
deviation before any difference.

**65 of 73 `class0_pbI2` images carry no annotation** and are excluded rather
than treated as negatives; including them would have injected 65 confident false
negatives. The cost is that PbI₂ has 5 training images, its AP is 0.0000 in
every run, and it is flagged not-interpretable everywhere it appears. No PbI₂
claim is made.

### Why the published splits could not be used

Grouping files by the specimen they came from, the original `train/val/test.txt`
share **22 groups between train and val, 18 between train and test, 8 between
val and test** — 36 distinct source groups leaked. Two mechanisms: `_aug###`
flips of one image landing on both sides, and the same specimen filed under two
class folders. `data/splits.py` audits this and records the result in
`splits_manifest.json`.

### Why the metadata banner is stripped

All 338 images carry a burned-in FESEM instrument banner over the bottom ~8.2%
of their height. It is removed once, corpus-wide, at curation, and every
downstream stage reads `data/curated/`. Left in, it is a context cue present in
every image, a magnet for saliency maps, and exactly the kind of high-contrast
blob a morphological filter is built to find. 27 annotations that lived entirely
inside the banner were dropped with it.

### Why defect scale is the framing

Against a 640 px detector input, **60.9% of test boxes are under 16 px** and the
median test defect is **13.3 px**. Bins are anchored to detector strides
(P3 = 8, P4 = 16, P5 = 32), not to quantiles of the data, and are committed in
`configs/tiny_defect_bins.yaml` **before** any detector exists.

> No nanometre-scale claim is possible here. JPEG re-encoding stripped the FESEM
> `Image Pixel Size` headers from all 440 images and no `.tif` originals survive,
> so there is no pixel-size calibration. Every size is in pixels, and renderer
> `severity` is a normalised simulation control — never a depth.

## Guarantees (enforced by tests, not by convention)

```
tests/test_snapshot_integrity.py   corpus matches its manifest: 440 / 415 / 9357
tests/test_no_leakage.py           groups, pixels and augmentations are disjoint
tests/test_renderer.py             boxes bound their masks; classes are exact
tests/test_refiner_gradients.py    the reconstruction loss is not identically zero
tests/test_master_results.py       one row per experiment; no cross-architecture
                                   or cross-regime averaging in the report
```

- **The test split is written once by stage 1 and not read again until stage 9.**
- **Renderer priors come from expert boxes on train only** — MicroDefectCV never
  enters a training loss, so it stays an independent evaluator (the firewall).
- **Synthetic canvases come only from train backgrounds**, asserted at generation.
- **The refiner never moves a label.** It rewrites texture inside the mask; box,
  mask and class are ground truth by construction, and all three pools carry
  byte-identical labels.

## Figures

```bash
py -3.10 figures/gen_figures.py        # 8  generation figures
py -3.10 figures/train_figures.py      # 8  training / results figures
py -3.10 figures/eval_figures.py       # 6  diagnostics figures
py -3.10 figures/regen_val_plots.py    # per-run confusion matrix + PR/P/R/F1
```

`figures/style.py` holds the shared palette and rcParams so the families read as
one document. No seaborn — matplotlib covers it, and a new dependency on a
working 4 GB CUDA environment is a bad trade.

`train_figures.py` prints a **parsed-vs-documented sanity check** against the
values quoted in `RESULTS.md`. Read that output, not just the PNGs: it is what
caught a per-bin table reporting recall under an `AP50` header, and a stale
baseline left behind by an append-only CSV.

## Layout

```
data/       snapshot.py build_dataset.py splits.py sem_bar.py yolo_export.py
            raw_snapshot/ (write-once)  curated/  splits/  synthetic/  yolo/
synth/      renderer.py  generate.py  refiner.py
eval/       detection.py  tiny_defect.py  domain_gap.py  open_set.py
            calibration.py  robustness.py  failure_analysis.py  explain.py
            counterfactual.py  microdefectcv_baseline.py  final_eval.py
interpret/  depth_sfs.py  boundary_index.py  pl_proxy.py  run_interpretation.py
figures/    style.py  gen_figures.py  train_figures.py  eval_figures.py
            regen_val_plots.py
tests/      the guarantees above
configs/    tiny_defect_bins.yaml (pre-registered)
outputs/    RESULTS.md  master_results.csv  figures/  failures/
docs/       dissertation documents
run_full_pipeline.py    train_detector.py  train_refiner.py  make_report.py
```

`interpret/` modules are downstream scientific reading of CV output, not
contributions. Their quantities are image-derived indices with no electrical
validation, and are named accordingly — a "Relative Depression Index", not a
shunt depth in nanometres.

## Outstanding

| priority | work | cost | why |
|---|---|---|---|
| **1** | Step-matched real-only baseline | ~2.2 h | The only experiment that could overturn the headline. See `RESULTS.md` §13 |
| 2 | Scaling curve at n=3 for 2 / 10 / 25% | ~4 h | Curve *shape* is currently single-seed |
| 3 | RT-DETR cross-paradigm arm | ~1.5 h | Wired and ready: `--model rtdetr-l --batch 4` |
| — | PbI₂ detection and open-set | blocked | Needs annotation, not compute. The 65 unlabelled `class0_pbI2` images are the unlock |

Not attempted: micro-sam (EM-pretrained encoder, needs an install; generic SAM
is not a substitute) and PerovSegNet (segmentation, needs a clone and a width
reduction to fit 4 GB).
