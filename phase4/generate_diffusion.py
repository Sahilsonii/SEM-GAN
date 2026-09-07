"""
Whole-image diffusion pool with exact labels: renderer layout -> ControlNet.

    py -3.10 phase4/generate_diffusion.py [--n 500] [--seed 42]

Per image:
  1. a clean TRAIN background, scaled to 512x352 (the real aspect);
  2. the parametric renderer (synth/renderer.py) draws a sampled defect layout
     on it - counts from the real prior, class mix from the real train boxes,
     grain-boundary placement - and returns exact boxes;
  3. SD 1.5 re-denoises the WHOLE composite (img2img, strength STRENGTH) with
     Canny edges of the composite as the ControlNet condition and a random
     real train image as the IP-Adapter style reference. The drawn defect
     seeds the pixels inside its box, the grain texture is regenerated around
     it, and there is no paste boundary anywhere: every pixel in the output
     went through the same denoising.
  4. The labels written are the renderer's. Whether the diffusion model kept a
     defect where the renderer put it is exactly what the detector benchmark
     tests; nothing is inferred after the fact.

40% of images are backgrounds (condition = the clean background's edges, no
defects, empty label), as in the copy-paste and layout pools.

Output layout is the one every other pool uses, so yolo_export/train_detector
need no change: data/synthetic/diffusion_cn/{images,labels,masks,cond}/,
manifest.jsonl (with background_group), summary.json. Resumable.
All weights are local Hugging Face downloads; no image leaves this machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.generate import (CURATED_IMAGES, OUT_ROOT, _assert_no_holdout_leak,  # noqa: E402
                            _fit_long_side, _load_split, background_pool)
from synth.renderer import fit_priors, render, sample_params           # noqa: E402

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN = "lllyasviel/control_v11p_sd15_canny"
IP = ("h94/IP-Adapter", "models", "ip-adapter_sd15.bin")
W, H = 512, 352
PROMPT = ("scanning electron micrograph of a polycrystalline perovskite thin film, "
          "grains and grain boundaries, pinholes, bright lead iodide platelets, "
          "greyscale, sharp, high magnification")
NEG = "color, blurry, painting, cartoon, text, watermark"
STEPS, GUIDANCE, CN_SCALE, IP_SCALE, STRENGTH = 30, 6.0, 0.8, 0.6, 0.25
CANNY = (60, 140)
BG_FRACTION = 0.4
EDGE_BOOST = False     # tested: drawing outlines into the condition LOWERED pinhole retention (63% vs 75% at 0.25)


def condition(composite, mask):
    """ControlNet condition: Canny of the composite, plus - when EDGE_BOOST - the
    outline of every rendered defect. Canny alone does not tell the model which
    edges are pits; at strength 0.5 it repainted about half of them away."""
    edges = cv2.Canny(cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY), *CANNY)
    if EDGE_BOOST and mask is not None and mask.any():
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = np.zeros_like(edges)
        cv2.drawContours(outline, cnts, -1, 255, 1)
        edges = np.maximum(edges, cv2.dilate(outline, np.ones((2, 2), np.uint8)))
    return edges


# Sweep on the 10 defect audit layouts (phase4/assets/strength_sweep*.json):
# retention falls with strength (0.15: 82%, 0.25: 75%, 0.5: 55%) and the loss
# is on T1/T2 boxes; T3/T4 keep >= 90% at 0.25. A ghost band with image
# rejection threw away 6-7 of 10 images at every strength, so the rule is a
# single threshold: a box whose defect keeps >= 30% of its rendered contrast
# is a real, if faint, defect and stays labelled; below that it is gone and
# the label goes with it. Strength 0.25 = the user's fallback: labels first.
KEEP_RATIO, GHOST_RATIO, MAX_TRIES = 0.3, 0.3, 1


def box_contrast(gray, box):
    """Mean inside the box minus mean of a 4-px ring around it. Sign says
    pit (negative) or platelet (positive)."""
    H, W = gray.shape
    _, cx, cy, w, h = box
    x0, x1 = int((cx - w / 2) * W), int(np.ceil((cx + w / 2) * W))
    y0, y1 = int((cy - h / 2) * H), int(np.ceil((cy + h / 2) * H))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = 4
    ring = gray[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m].astype(float)
    inner = gray[y0:y1, x0:x1].astype(float)
    return inner.mean() - (ring.sum() - inner.sum()) / max(ring.size - inner.size, 1)


def verify_labels(composite, output, boxes):
    """Labels are exact by VERIFICATION, not only by construction. For every
    rendered box: ratio = output contrast / rendered contrast.
      ratio >= KEEP_RATIO            -> defect survived, label kept
      ratio <  GHOST_RATIO (or gone) -> defect erased, label dropped (the image
                                        has no defect there, so no label is right)
      in between                     -> a ghost: visible residue that would be an
                                        unlabelled defect. The image is rejected.
    Returns (kept_boxes, n_dropped, has_ghost, ratios)."""
    gi = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    go = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    kept, dropped, ghost, ratios = [], 0, False, []
    for b in boxes:
        ci, co = box_contrast(gi, b), box_contrast(go, b)
        if ci is None or abs(ci) < 3:
            dropped += 1
            continue
        r = (co / ci) if np.sign(co) == np.sign(ci) else 0.0
        ratios.append(r)
        if r >= KEEP_RATIO:
            kept.append(b)
        elif r < GHOST_RATIO:
            dropped += 1
        else:
            ghost = True
    return kept, dropped, ghost, ratios


def load_pipe():
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    cn = ControlNetModel.from_pretrained(CN, dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        SD, controlnet=cn, dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False)
    pipe.load_ip_adapter(IP[0], subfolder=IP[1], weight_name=IP[2])
    pipe.set_ip_adapter_scale(IP_SCALE)
    # sequential offload, measured on the 3050 Ti at 512x352: 0.44 GiB peak and
    # 15 s/img. Model-level offload peaked at 3.8 GiB, spilled into shared
    # memory and took 75 s/img. Not attention slicing either: it replaces the
    # IP-Adapter attention processors and the pipeline crashes.
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(n_images=500, seed=42, pool_name="diffusion_cn", split="train"):
    rng = np.random.default_rng(seed)
    backgrounds = background_pool(split)
    recs = _load_split(split)["records"]
    all_real = [r for r in recs if r["n_boxes"] >= 0]           # any train image as style ref
    priors = fit_priors(split)
    n_pb = sum(1 for r in recs for b in r["boxes"] if b[0] == 0)
    n_all = sum(r["n_boxes"] for r in recs)
    pbi2_fraction = n_pb / max(1, n_all)                         # the real train box mix

    out_dir = OUT_ROOT / pool_name
    for sub in ("images", "labels", "masks", "cond"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = {json.loads(l)["stem"] for l in manifest.read_text().splitlines() if l.strip()} \
        if manifest.exists() else set()

    pipe = load_pipe()
    used, n_boxes, n_bg, t_gen, n_rejected = set(), 0, 0, [], 0
    with manifest.open("a", encoding="utf-8") as mf:
        for i in tqdm(range(n_images), desc=f"diffusion[{pool_name}]", unit="img", dynamic_ncols=True):
            stem = f"dc_{pool_name}_{i:05d}"
            bg = backgrounds[i % len(backgrounds)]
            used.add(bg["group"])
            # draw the RNG in the same order whether or not we skip, so a resumed
            # run reproduces the layouts it would have made
            is_bg = rng.random() < BG_FRACTION
            k = int(min(60, max(1, rng.choice(priors["counts"]))))
            ref = all_real[int(rng.integers(len(all_real)))]
            seed_i = int(rng.integers(0, 2**31 - 1))
            if stem in done:
                continue

            canvas = _fit_long_side(cv2.imread(str(CURATED_IMAGES / bg["file"])), 512)
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
            if is_bg:
                composite, boxes, mask, params = canvas, [], np.zeros((H, W), np.uint8), []
                n_bg += 1
            else:
                gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                params = sample_params(priors, k, rng, render_px=512,
                                       pbi2_fraction=pbi2_fraction, bg_gray=gray)
                res = render(canvas, params, seed=seed_i)
                composite, boxes, mask, params = res["image"], res["boxes"], res["mask"], res["params"]
            boxes_rendered, dropped, attempt, ratios = boxes, 0, 0, []

            edges = condition(composite, mask)
            cond = Image.fromarray(np.stack([edges] * 3, -1))
            ref_img = cv2.resize(cv2.imread(str(CURATED_IMAGES / ref["file"])), (W, H))
            ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))

            init = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
            for attempt in range(MAX_TRIES):
                t0 = time.time()
                img = pipe(PROMPT, negative_prompt=NEG, image=init, control_image=cond,
                           ip_adapter_image=ref_pil, strength=STRENGTH,
                           num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                           controlnet_conditioning_scale=CN_SCALE, width=W, height=H,
                           generator=torch.Generator("cpu").manual_seed(seed_i + attempt)).images[0]
                t_gen.append(time.time() - t0)
                out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)      # the corpus is greyscale
                kept, dropped, ghost, ratios = verify_labels(composite, out, boxes)
                if not ghost:
                    break
            if ghost:
                n_rejected += 1
                continue                                          # no image, no label
            boxes_rendered, boxes = boxes, kept
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), mask)
            cv2.imwrite(str(out_dir / "cond" / f"{stem}.png"), edges)
            cv2.imwrite(str(out_dir / "cond" / f"{stem}_init.jpg"), composite,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            (out_dir / "labels" / f"{stem}.txt").write_text(
                "".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes), encoding="utf-8")
            mf.write(json.dumps({
                "stem": stem, "background_file": bg["file"], "background_group": bg["group"],
                "ip_ref_file": ref["file"], "source_split": split, "render_px": 512,
                "n_defects": len(boxes), "n_rendered": len(boxes_rendered), "n_dropped": dropped,
                "attempt": attempt, "retention_ratios": [round(r, 3) for r in ratios],
                "seed": seed_i, "params": params,
            }) + "\n")
            mf.flush()
            n_boxes += len(boxes)

    _assert_no_holdout_leak(used)
    # totals from the manifest, so a resumed run reports the whole pool
    ents = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    summary = {"pool": pool_name, "method": "renderer layout -> Canny ControlNet + IP-Adapter, SD1.5",
               "images": len(ents), "backgrounds": sum(1 for e in ents if not e["n_defects"]),
               "boxes": sum(e["n_defects"] for e in ents), "backgrounds_used": len(used),
               "seed": seed, "size": [W, H], "steps": STEPS, "guidance": GUIDANCE,
               "controlnet_scale": CN_SCALE, "ip_adapter_scale": IP_SCALE, "canny": CANNY,
               "pbi2_fraction": round(pbi2_fraction, 4),
               "img2img_strength": STRENGTH, "edge_boost": EDGE_BOOST,
               "boxes_rendered": sum(e.get("n_rendered", e["n_defects"]) for e in ents),
               "boxes_dropped_erased": sum(e.get("n_dropped", 0) for e in ents),
               "images_rejected_ghost": n_rejected,
               "label_rule": f"keep if contrast ratio >= {KEEP_RATIO}; drop if < {GHOST_RATIO}; "
                             f"reject image if any box in between (up to {MAX_TRIES} seeds)",
               "seconds_per_image": round(float(np.mean(t_gen)), 2) if t_gen else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[diffusion] '{pool_name}': {summary['images']} images ({summary['backgrounds']} backgrounds), "
          f"{summary['boxes']} boxes kept of {summary['boxes_rendered']} rendered "
          f"({summary['boxes_dropped_erased']} erased, {n_rejected} images rejected for ghosts), "
          f"{summary['seconds_per_image']} s/img, leak check passed")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", default="diffusion_cn")
    a = ap.parse_args()
    generate(a.n, a.seed, a.pool)
