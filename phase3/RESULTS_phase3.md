# Phase 3 — composition-first synthesis

Phase 2 froze with the synthetic line neutral: three pools (rendered PbI₂-only,
rendered balanced, copy-paste with real crops), two budgets, never above
real-only (`outputs/RESULTS.md` §17). Appearance was ruled out by the
copy-paste control; class mix by the balanced pool; budget dilution by the
8,300-step cell. What every pool shared was **composition**: defects scattered
by a prior over backgrounds a human had labelled clean, at counts drawn
independently of where they land. A clean film with 26 pasted pits is not a
defective film.

The 50 annotated train images carry the real thing — counts from 1 to 266,
clustering, class co-occurrence, boundary affinity — so rather than model that
distribution, these two pools **reuse it** (`phase3/synth_compose.py`).

## The two pools (seed 42, 500 images each, render 512 px)

| pool | construction | boxes | backgrounds | class mix (pbi2 / pinhole) | count per image |
|---|---|---|---|---|---|
| `layout_transfer` | the exact box layout of one real annotated train image (flipped with p = 0.5, jitter σ = 1.5% of canvas) reproduced on a clean train background with class- and size-matched real crops | 17,582 | 189 (38%) | 1,429 / 16,153 | median 4, mean 35, max 266 |
| `augment_defective` | a real annotated train image as canvas, its labels kept verbatim, plus 20% extra class-matched crops placed 1–2 nearest-neighbour distances from a real defect | 31,360 (5,290 pasted) | 0 | 3,970 / 27,390 | median 31, mean 63, max 319 |

Both: leak assertion on every canvas group, labels exact by construction,
`phase3/tests/test_compose.py` green (layout reproduced box-for-box; real
labels verbatim; both classes survive export; backgrounds kept).

What each isolates:

- `layout_transfer` changes **only the layout** relative to copy-paste. Same
  crops, same backgrounds, same blend. If it moves, composition was the lever.
- `augment_defective` adds **real defective-film context** as well. It is
  Ghiasi et al. 2021 as actually practised: augment the images that carry the
  signal, not the ones that do not.

## Protocol

Six runs, `py -3.10 phase3/run_phase3.py`: each pool × seeds 1, 2, 42,
YOLO11s, 160 real + 500 pool images, **8,300 steps** (101 epochs, patience
500), checkpoint on real val, scored once on the locked test split by the
same `final_eval` merge as Phase 2. Comparators already on disk at the same
budget: real-only 0.1480 ± 0.0367 (seeds 0.1509 / 0.1100 / 0.1832), balanced
rendered 0.1347 ± 0.0052.

## Pre-registered reading (written 2026-09-06, before any run)

Against real-only at 8,300 steps, mean of 3 seeds, Welch p:

| if | conclusion |
|---|---|
| `augment_defective` ≥ 1.15×, p < 0.05, **and** `layout_transfer` within 0.85–1.15× | composition **and** defective-film context matter; clean-background synthesis was the flaw, not the generator. Reopens the renderer line with real defective canvases. |
| **both** ≥ 1.15×, p < 0.05 | layout alone is enough; the renderer should sample layouts from real images instead of a prior. |
| both within 0.85–1.15× (or p ≥ 0.05) | composition was not the lever either. The corpus is at its 160-image floor; only grouped CV (to see a small effect) or new annotation moves it. Synthetic line closes. |
| either < 0.85× | that pool is harmful; report and stop that pool. |

Secondary, read regardless: per-bin AP50 at 8,300 (does T3/T4, where every
earlier pool lost, recover?), val-vs-test per seed (does the val gain finally
transfer?), and `eval/domain_gap.py --pools layout_transfer,augment_defective`
for two more rows in the "gap does not predict utility" table.

A caution on `augment_defective`: its canvases are the 50 real defect images
the real-only arm already trains on, so at 8,300 steps it is partly "more
passes over the same defective images plus 20% extra boxes". If it wins and
`layout_transfer` does not, the honest statement is that *re-exposure to real
defective context* helped, which is an augmentation result, not a synthesis
one — and still useful.

## Results — the fourth row: both pools are harmful

All six runs completed at 101 epochs (verified from `results.csv`). Scored
once on the locked test split, three seeds each, 8,300 steps throughout:

| arm, 8,300 steps | mAP50 | mAP50-95 | vs real-only | Welch *p* | precision | recall | seeds |
|---|---|---|---|---|---|---|---|
| real only | 0.1480 ± 0.0367 | 0.0582 ± 0.0168 | — | — | 0.049 | 0.349 | 0.1509 / 0.1100 / 0.1832 |
| + balanced rendered (Phase 2) | 0.1347 ± 0.0052 | 0.0529 ± 0.0008 | 0.91× | 0.60 | 0.036 | 0.323 | 0.1314 / 0.1320 / 0.1407 |
| **+ layout_transfer** | **0.0852 ± 0.0174** | 0.0295 ± 0.0078 | **0.58×** | 0.079 | 0.035 | 0.304 | 0.0950 / 0.0651 / 0.0954 |
| **+ augment_defective** | **0.1048 ± 0.0043** | 0.0364 ± 0.0038 | **0.71×** | 0.18 | 0.034 | 0.313 | 0.1011 / 0.1039 / 0.1095 |

Layout transfer is also below the rendered pool it was meant to improve on
(0.63×, p = 0.030). Augment-defective against layout transfer: 1.23×,
p = 0.18.

| per-bin AP50, test, 8,300 steps | T1 <8 px | T2 8–16 | T3 16–32 | T4 ≥32 |
|---|---|---|---|---|
| real only | 0.0125 | 0.1493 | 0.2218 | 0.1980 |
| + balanced rendered | 0.0170 | 0.1409 | 0.2032 | 0.1699 |
| + layout_transfer | 0.0067 | 0.0914 | 0.1342 | 0.1139 |
| + augment_defective | 0.0050 | 0.1274 | 0.1434 | 0.1495 |

Every band, both pools, below real-only. The loss is largest in T3/T4, as it
was for every earlier pool.

| val vs test, per seed | val mAP50 (selection) | test mAP50 (locked) | best epoch / 101 |
|---|---|---|---|
| layout_transfer s1 / s2 / s42 | 0.088 / 0.113 / 0.093 | 0.095 / 0.065 / 0.095 | 42 / 90 / 82 |
| augment_defective s1 / s2 / s42 | 0.122 / 0.111 / 0.120 | 0.101 / 0.104 / 0.110 | 64 / 68 / 47 |

Same pattern as Phase 2: on val both pools sit at or above the real-only val
mean (0.108), on test both lose. Layout seed 2 is the extreme case, val 0.113
against test 0.065.

### Two secondary checks, both negative for the obvious explanations

**Domain gap does not predict this either** (`eval/domain_gap.py --skip-l4`,
60 real vs 60 pool images, lower = closer to real train):

| pool | L1 pixel | L2 frequency | L3 morphology | test mAP50 at its budget |
|---|---|---|---|---|
| copy-paste (Phase 2, 2,000 steps) | **0.19** | **0.10** | 1.16 | 0.0525 (0.76× real@2,000) |
| layout_transfer | 0.57 | 0.11 | 1.18 | 0.0852 (0.58× real@8,300) |
| rendered, refined | 5.13 | 0.20 | 1.21 | 0.1347 (0.91× real@8,300) |
| rendered, no FFT branch | 4.55 | 0.53 | 1.18 | — |
| augment_defective | 10.34 | 0.48 | 1.29 | 0.1048 (0.71× real@8,300) |

The two pools closest to real on every level are the two that hurt most. The
rendered pool, ten to twenty times further away on L1, is the least harmful.
This is the third time in the project that a smaller measured gap has failed
to predict utility, now in the wrong direction.

**It is not false alarms on clean films** (`phase3/diag_background_fp.py`,
predictions per image on the 51 defect-free and 19 defect-bearing test images
the open-set export carries, mean of 3 seeds):

| arm, 8,300 steps | preds / defect-free image @0.05 | @0.25 | preds / defect image @0.05 | defect-free images with any pred @0.25 |
|---|---|---|---|---|
| real only | 25.0 | 2.07 | 75.9 | 21% |
| + balanced rendered | 24.7 | 1.61 | 74.4 | 25% |
| + layout_transfer | 21.0 | 1.10 | 68.0 | 28% |
| + augment_defective | 16.6 | 1.17 | 76.3 | 37% |

The hypothesis written before this check was that pasted content teaches the
detector to fire on things that are not defects, so the augmented arms would
put more boxes on defect-free films. They put *fewer*. The extra predictions
in the headline `n_pred` are on the defect-bearing images, and the AP loss is a
ranking loss there: the augmented detectors fire about as often on real
defects but order their confidences worse, so precision falls at every recall.

### Reading, against the pre-registration

The fourth row: **either pool < 0.85× → harmful, report and stop**. Both are.

What this rules out, on top of Phase 2:

- **Composition was not the lever.** Real layouts, real counts, real
  clustering, real class co-occurrence, on clean backgrounds: 0.58×. The best
  composition this corpus can offer made the worst pool.
- **Real defective context was not the lever.** Augmenting the actual
  defective images with 20% extra real crops beside real defects, Ghiasi's
  recipe as practised: 0.71×. And that arm gave the detector ten times more
  passes over the 50 defect images than real-only had; more exposure to the
  real defective images, with pastes in them, was worse than fewer passes
  without.
- **It is not background false alarms**, and it is **not the domain gap**.

What is left is the one thing every pasted or rendered pool shares and
real-only does not: **the pixels inside and around the box were not produced
by the same imaging event as the film around them.** A Poisson-blended real
crop is a real defect with a wrong seam, a wrong local texture gradient, and
a margin from another specimen. The detector, which has 8,300 steps and a
660-image set, fits those cues; they rank real defects worse at test time.
The rendered pool hurts least precisely because its defects are smooth
synthetic shapes with no foreign margin. Copy-paste has the smallest measured
gap and the worst utility because the gap metrics look at whole-image
statistics and the damage is at the box boundary.

This is consistent with everything measured since §13 of the Phase 2 results
and it closes the augmentation line for this corpus: **six pools, four
budgets counted across both phases, 27 augmented seeds, none above
real-only.** The honest statement for the thesis is that on a 160-image,
tiny-defect FESEM corpus, every form of label-exact augmentation tried —
rendered, GAN-refined, real-crop copy-paste, real-layout transfer, and
in-place augmentation of real defective images — was neutral or harmful at a
matched optimisation budget, while the same budget spent on the real images
alone doubled the baseline. The generator does what it claims; the detector
does not need it.

### What would still be worth doing, in order

1. **Real-only, longer.** Best epochs at 8,300 steps were 287–365 of 415;
   the baseline is still rising. A 16,600-step real-only arm is the cheapest
   experiment left and it raises the bar every claim is measured against.
2. **Grouped 5-fold CV** over the 84 defect groups for selection. Every
   pool won on the 12-image val and lost on test; until selection is stable,
   no small effect in either direction is measurable.
3. **Seam-free augmentation only if 1–2 leave room**: geometric and
   photometric augmentation of the real images (already in the YOLO
   pipeline), or crops pasted with *no* margin at all and the box grown to
   the paste, so the seam is inside the label. Not another pool of the kind
   tried here.

Phase 3 is frozen at these six runs. 2026-09-06. 42 detector runs on the CSV.
