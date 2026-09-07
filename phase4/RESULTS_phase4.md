# Phase 4 — whole-image diffusion with exact labels

## Why this arm exists

Phases 2 and 3 closed every *composite* route: rendered, GAN-refined,
real-crop copy-paste, real-layout transfer, in-place augmentation of real
defective images. Six pools, 27 augmented seeds, none above real-only at a
matched budget, and the two pools closest to real on every gap measure hurt
the most (`phase3/RESULTS_phase3.md`). What every one of them shares is a
composite boundary: pixels inside and around the box that did not come from
the same imaging event as the film around them.

This arm is the one construction with **no composite boundary**: the
renderer draws the layout (labels exact by construction), and a diffusion
model re-denoises the **whole image** in one pass, so every pixel in the
output went through the same process. It is also the private route the
project needs: Stable Diffusion 1.5, a Canny ControlNet and an IP-Adapter
are free weights downloaded once to this machine; no real image is uploaded
anywhere; nothing is fine-tuned.

## Feasibility (2026-09-07, `phase4/smoke.py`, `phase4/assets/smoke.json`)

| configuration | peak VRAM | s / image | outcome |
|---|---|---|---|
| txt2img, model CPU offload, 20 steps | 2.47 GiB | 7.2 | runs; but defects were **not** reliably painted where the renderer put them (Canny of a dense grain field does not tell the model which edges are pits) |
| img2img from the rendered composite, model offload | 3.80 GiB, spills | 75 | defects preserved; unusable speed |
| **img2img, sequential CPU offload** | **0.44 GiB** | **15** | defects preserved; the configuration used |

`enable_attention_slicing()` must not be used: it replaces the IP-Adapter
attention processors and the pipeline crashes on the (text, image) tuple.

## Model audit (2026-09-07, `phase4/compare_models.py`)

Five free open-weight models, identical inputs: 20 renderer layouts (5 per
class: background 3D, background 3D-2D, pinhole-only, PbI₂-only), same
Canny condition, same IP-Adapter reference where the family has one, same
strength 0.5 / 30 steps / guidance 6. "Kept" = share of label boxes whose
inside-vs-surround contrast kept its sign and ≥ 50% of its rendered
magnitude after the diffusion pass.

| model | s / img | pinhole kept | PbI₂ kept | verdict |
|---|---|---|---|---|
| **SD 1.5** + Canny CN + IP-Adapter | 20 | 55% | 60% | **kept** — closest to the real films, grain structure preserved |
| Realistic Vision 5.1 | 19 | 42% | 56% | deleted — marginally sharper, erases more pinholes |
| DreamShaper 8 | 19 | 49% | 51% | deleted — invents honeycomb patterns inside grains |
| SDXL 1.0 + SDXL CN + SDXL IP-Adapter | 120 | 61% | 73% | deleted — best retention but waxy, embossed, pits become pebbles |
| SD 2.1 base + SD2.1 Canny CN | — | — | — | download blocked mid-run; dropped (no IP-Adapter exists for it) |

Contact sheets: `phase4/assets/compare_*.png`. Cache after deletion: SD 1.5,
its Canny ControlNet, the SD 1.5 IP-Adapter, DINO — 6.4 GB.

## Strength sweep (SD 1.5, the 10 defect audit layouts, `phase4/strength_sweep.py`)

| strength | pinhole kept | PbI₂ kept | T1 <8 px | T2 8–16 | T3 16–32 | T4 ≥32 | median out/init contrast |
|---|---|---|---|---|---|---|---|
| 0.15 | 82% | 82% | | | | | 0.85 |
| 0.20 | 79% | 80% | | | | | 0.84 |
| **0.25** | 75% | 75% | 53% | 74% | 91% | 90% | 0.83 |
| 0.35 | 65% | 71% | 53% | 62% | 82% | 85% | 0.77 |
| 0.50 | 55% | 60% | 44% | 57% | 70% | 55% | 0.57 |

Two things tried and rejected: drawing the renderer's defect outlines into
the ControlNet condition (`EDGE_BOOST`) *lowered* pinhole retention (63% vs
75% at 0.25 — the model paints a ridge where it is shown an outline); and a
ghost band with whole-image rejection threw away 6–7 of 10 images at every
strength.

**Decision.** No strength reaches the 85–90% target on the tiny bands, and
the loss is concentrated in T1/T2. So: **strength 0.25** (the pre-agreed
fallback: labels before prettier texture), and labels are made exact by
**verification** rather than only by construction: every rendered box is
checked against the output, kept if its defect retains ≥ 30% of the rendered
contrast (87% of boxes at 0.25 on the audit set), dropped otherwise — a
dropped box is a defect the image no longer contains. `n_rendered`,
`n_dropped` and the per-box ratios are written to the manifest, so the
size bias of the drops is auditable. The pool therefore under-represents
sub-16-px defects relative to the renderer's prior; that is stated, not
hidden.

## The pool: `diffusion_cn` (`phase4/generate_diffusion.py`)

Per image: a clean TRAIN background at 512×352 → the parametric renderer
draws a layout (counts from the real prior, class mix = real train box mix,
grain-boundary placement, exact boxes) → SD 1.5 img2img at strength 0.25,
30 steps, guidance 6, with Canny(60,140) of the composite as the ControlNet
condition (scale 0.8) and a random real train image as the IP-Adapter
reference (scale 0.6). 40% of images are backgrounds. Output greyscale.
500 images, seed 42, resumable, leak-asserted. The written labels are the
renderer's boxes that survived verification (≥ 30% contrast retained).

The style reference and the background edges are derived from real train
images, so the output is not independent of the real data; the memorisation
check below quantifies how close it is before anything is called releasable.

## Protocol

`py -3.10 phase4/run_phase4.py`, six runs at the 8,300-step budget,
patience 500, seeds 1, 2, 42, YOLO11s, locked test split scored once:

| regime | train images | steps / epochs |
|---|---|---|
| `diffusion_cn_only_steps8300` (**releasability**: synthetic alone, real val for selection) | 500 diffusion | 8,300 / ~132 |
| `diffusion_cn_steps8300` (augmentation) | 160 real + 500 diffusion | 8,300 / 101 |

Comparators on disk at 8,300 steps: real-only 0.1480 ± 0.0367; balanced
rendered 0.1347 ± 0.0052; layout transfer 0.0852; augment-defective 0.1048.
Synthetic-only rendered at its budget: 0.0029.

## Pre-registered reading (written before generation finished)

**Releasability** (`diffusion_cn_only`, scored on the locked real test split):

| if | conclusion |
|---|---|
| ≥ 0.10 (two-thirds of real-only) | a diffusion set trained on nothing but the renderer's layouts and real style references transfers; the set is releasable as training data, subject to the memorisation check |
| 0.03 – 0.10 | partial transfer; releasable as a *pre-training* set with a real fine-tune, not as a replacement |
| < 0.03 (the rendered synthetic-only floor) | no transfer; the set is a showcase, not training data, and the thesis says so |

**Augmentation** (`diffusion_cn` mixed vs real-only at 8,300):

| if | conclusion |
|---|---|
| ≥ 1.15×, p < 0.05 | the composite boundary was the lever all along; whole-image generation helps where every paste did not |
| 0.85 – 1.15× | neutral, like the rendered pool: removing the seam did not matter; the corpus is at its floor |
| < 0.85× | harmful, like the real-pixel pools; the line closes for good |

**Memorisation** (`phase4/memorisation_check.py`): DINO ViT-B/16 cosine
similarity of every generated image to every real train image. Reported:
similarity to its own background source, to its IP-Adapter reference, and to
the nearest *other* real image, against the real-to-real nearest-neighbour
distribution. Any generated image whose nearest real neighbour is closer
than the 99th percentile of real-to-real similarity is flagged and would be
dropped from a release. Pixel-level check: normalised cross-correlation with
its own background at 256 px; > 0.95 flags a near-copy.

## Results

_(appended after the runs; every number pasted from `outputs/final_test_results.json`)_
