# Phase 4 — whole-image diffusion with verified labels

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
fallback: labels before prettier texture). The labels of this pool are
**not** "exact" in the sense that every rendered label survives; they are
*verified* labels: generated exactly by construction and subsequently
filtered using an output-image contrast-preservation check, which removes
the defects that did not survive refinement. Every rendered box is checked
against the output, kept if its defect retains ≥ 30% of the rendered
contrast (87% of boxes at 0.25 on the audit set), dropped otherwise — a
dropped box is a defect the image no longer contains, so the detector is
never trained to find something that is not there. `n_rendered`,
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
renderer's boxes that survived the contrast-preservation check (≥ 30%
retained); the size-dependent retention is reported as a result, not hidden.

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

### The pool as built

500 images, 205 backgrounds, strength 0.25, 11.2 s/image. **8,805 boxes
rendered, 7,674 kept, 1,131 dropped** by the contrast-preservation check
(87% kept overall; no image was rejected, because the ghost band was
replaced by a single threshold before generation). Class mix of the kept
labels: 6,686 pinhole / 988 PbI₂. Median 2 boxes per image, mean 15.3,
max 59.

**Retention is size-dependent, as the sweep predicted** — this is the
methodological result of the phase, and it is a limitation of the method,
not a defect of the reporting:

| band (side at 640 px) | rendered | kept |
|---|---|---|
| T1 < 8 px | 2,074 | **81%** |
| T2 8–16 px | 3,764 | 88% |
| T3 16–32 px | 2,086 | 92% |
| T4 ≥ 32 px | 798 | **95%** |

By class: pinhole 88% (n = 7,632), PbI₂ 90% (n = 1,090). So the pool
under-represents sub-8-px defects relative to the renderer's prior by about
14 points against the largest band. Any detector trained on it sees a
size distribution shifted upward from the one the renderer sampled.

### Detector results, locked test split, 3 seeds, 8,300 steps

| arm | mAP50 | mAP50-95 | recall | precision | seeds |
|---|---|---|---|---|---|
| real only | 0.1480 ± 0.0367 | 0.0582 | 0.349 | 0.049 | 0.1509 / 0.1100 / 0.1832 |
| + rendered + GAN synthetic (Phase 2) | 0.1347 ± 0.0052 | 0.0529 | 0.323 | 0.036 | 0.1314 / 0.1320 / 0.1407 |
| **+ diffusion synthetic** | **0.1264 ± 0.0271** | 0.0475 | 0.327 | 0.035 | 0.1185 / 0.1566 / 0.1042 |
| rendered synthetic **only** (Phase 2) | 0.0029 ± 0.0023 | 0.0009 | 0.054 | 0.004 | — |
| **diffusion synthetic only** | **0.0050 ± 0.0027** | 0.0013 | 0.060 | 0.007 | 0.0052 / 0.0023 / 0.0076 |

| comparison | ratio | Welch *p* |
|---|---|---|
| diffusion mixed vs real only | **0.85×** | 0.46 |
| diffusion mixed vs rendered mixed | 0.94× | 0.65 |
| diffusion only vs rendered synthetic only | 1.72× | 0.36 |
| diffusion only vs real only | **0.03×** | 0.021 |

| per-bin AP50 at 8,300 steps | T1 <8 px | T2 8–16 | T3 16–32 | T4 ≥32 |
|---|---|---|---|---|
| real only | 0.0125 | 0.1493 | 0.2218 | 0.1980 |
| + rendered synthetic | 0.0170 | 0.1409 | 0.2032 | 0.1699 |
| + diffusion synthetic | **0.0215** | 0.1454 | 0.1845 | 0.1660 |

### Reading, against the pre-registration

**Releasability: the third row — no transfer.** `diffusion_cn_only` scored
**0.0050**, below the 0.03 threshold. It is 1.72× the rendered synthetic-only
floor (0.0029) but that difference is not significant at n = 3 (p = 0.36) and
both are two orders of magnitude below real-only. The training curves say the
same thing as Phase 2's synthetic-only arm: **best epoch 1, 7 and 7 of 134**,
so the selected checkpoint is essentially the COCO initialisation with a few
epochs of drift, and fitting the diffusion pool was monotonically harmful to
real-image performance. **The set is a showcase, not training data**, and the
thesis will say so.

**Augmentation: the middle row — neutral, trending down.** Mixed with the 160
real images at a matched budget, the diffusion pool gives **0.85×** real-only
(p = 0.46), statistically indistinguishable from the rendered pool it was
meant to improve on (0.94×, p = 0.65). Removing the composite seam did not
help. This is the fourth construction to land in the same place.

**One signal worth recording, honestly weak.** T1 (sub-8-px) AP is the only
bin where the diffusion arm leads: 0.0215 vs 0.0125 real-only and 0.0170
rendered. Absolute values are near zero and the seed spread is wide, so this
is a hypothesis for future work, not a finding. It is also the bin the pool
*under-represents* most (81% retention), which makes it harder, not easier,
to explain as a data effect.

**Memorisation: clean.** DINO ViT-B/16 cosine similarity, 500 generated
against 160 real train images. Real-to-real nearest-neighbour median 0.867,
99th percentile 0.973. Generated images: median 0.765 to their own background
source (max 0.963), 0.393 to their IP-Adapter reference, 0.712 to the nearest
other real image. **Zero images exceeded the p99 real-to-real threshold**, and
pixel-level cross-correlation to the own background had median 0.55, max 0.90,
**none above 0.95**. So the pool does not reproduce any real specimen and
would be safe to publish in place of the real corpus — it simply is not useful
as training data, which is the more important finding.

### What Phase 4 adds to the thesis

The seam hypothesis from Phase 3 is now tested and rejected. Whole-image
diffusion, with no composite boundary anywhere and verified labels, performs
like every paste-based pool: neutral to slightly harmful as augmentation, and
useless alone. Across Phases 2–4 that is **seven pools, 33 augmented seeds,
none above real-only at a matched budget**. The privacy question is answered
in the negative for training use: a releasable synthetic replacement for this
corpus does not exist by any method tried, though the diffusion pool is at
least memorisation-clean enough to publish as illustration.

**Phase 4 is frozen here.** 2026-09-08, 48 detector runs on the CSV.
