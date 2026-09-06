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

## Results

_(appended after the runs; every number pasted from `outputs/final_test_results.json`)_
