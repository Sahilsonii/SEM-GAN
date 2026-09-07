"""
Same layouts, five free image models: generate side by side, then choose.

    py -3.10 phase4/compare_models.py                 # all five, 20 audit images each
    py -3.10 phase4/compare_models.py --models sdxl   # one model
    py -3.10 phase4/compare_models.py --sheet         # only rebuild the contact sheets

Every model gets IDENTICAL inputs: the same 20 renderer layouts - 5 per
class over the corpus's four classes: two background morphologies (3D and
3D-2D films, clean) and two defect classes (pinhole-only, PbI2-only layouts,
drawn on canvases alternating between the two morphologies), seed 7 - the same
Canny condition, the same IP-Adapter reference where the family supports
one, the same img2img strength, steps and guidance. Only the weights change,
so what differs in the output is the model. This is an AUDIT set: enough to
judge each model by eye per class, not a training pool.

Models (all free, all downloaded once to the local Hugging Face cache, all
run on the 4 GB card with sequential CPU offload):

  sd15          Stable Diffusion 1.5 + Canny ControlNet + IP-Adapter        (the one already running)
  realvis       Realistic Vision 5.1 - SD1.5 fine-tune for photoreal texture, same ControlNet/IP-Adapter
  dreamshaper   DreamShaper 8 - SD1.5 fine-tune, same ControlNet/IP-Adapter
  sd21          Stable Diffusion 2.1 base + SD2.1 Canny ControlNet (no IP-Adapter exists for 2.1)
  sdxl          SDXL 1.0 base + SDXL Canny ControlNet + SDXL IP-Adapter, rendered at 768x528 then
                downscaled to 512x352 for parity. Largest: ~13 GB of weights, minutes per image.

Outputs:
  data/synthetic/_cmp_<model>/          images/ labels/ cond/ manifest.jsonl  (same pool layout)
  phase4/assets/compare_<model>.png     that model's 20 outputs, one row per class, labels drawn
  phase4/assets/compare_grid.png        rows = layouts, cols = rendered init | each model
  phase4/assets/compare_timing.json     s/img and peak VRAM per model

Nothing here decides anything. Look at the sheets, keep the models worth a
500-image pool, and the detector benchmark decides after that.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from synth.generate import CURATED_IMAGES, OUT_ROOT, _fit_long_side, _load_split, background_pool  # noqa: E402
from synth.renderer import fit_priors, render, sample_params   # noqa: E402
import generate_diffusion as G                                  # noqa: E402  PROMPT/NEG/STRENGTH/etc.

W, H = G.W, G.H
PER_CLASS, LAYOUT_SEED = 5, 7
# The corpus has two film morphologies (3D and 3D-2D) and two defect classes.
# 5 layouts each: two background classes on their own morphology's canvases,
# two defect classes drawn on canvases alternating between the morphologies.
CLASSES = ("background_3D", "background_3D-2D", "pinhole", "pbi2")
BG_FOLDER = {"background_3D": "class3_3D_background", "background_3D-2D": "class4_3D-2D_background"}
IPA_SD15 = ("h94/IP-Adapter", "models", "ip-adapter_sd15.bin")
IPA_SDXL = ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin")

MODELS = {
    "sd15": dict(family="sd15", base="stable-diffusion-v1-5/stable-diffusion-v1-5",
                 variant="fp16", cn="lllyasviel/control_v11p_sd15_canny", ipa=IPA_SD15),
    "realvis": dict(family="sd15", base="SG161222/Realistic_Vision_V5.1_noVAE", variant=None,
                    vae_from="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    cn="lllyasviel/control_v11p_sd15_canny", ipa=IPA_SD15),
    "dreamshaper": dict(family="sd15", base="Lykon/dreamshaper-8", variant=None,
                        cn="lllyasviel/control_v11p_sd15_canny", ipa=IPA_SD15),
    "sd21": dict(family="sd21", base="stabilityai/stable-diffusion-2-1-base", variant=None,
                 cn="thibaud/controlnet-sd21-canny-diffusers", ipa=None),
    "sdxl": dict(family="sdxl", base="stabilityai/stable-diffusion-xl-base-1.0", variant="fp16",
                 cn="diffusers/controlnet-canny-sdxl-1.0", ipa=IPA_SDXL,
                 vae_from="madebyollin/sdxl-vae-fp16-fix", size=(768, 528)),
}


# ----------------------------------------------------------------- layouts --

def make_layouts():
    """The same 16 (init composite, canny, boxes, reference) for every model."""
    rng = np.random.default_rng(LAYOUT_SEED)
    bgs = background_pool("train")
    recs = _load_split("train")["records"]
    priors = fit_priors("train")
    n_pb = sum(1 for r in recs for b in r["boxes"] if b[0] == 0)
    pbi2_fraction = n_pb / max(1, sum(r["n_boxes"] for r in recs))
    by_morph = {f: [b for b in bgs if b["source_folder"] == f] for f in BG_FOLDER.values()}
    out = []
    for i in range(PER_CLASS * len(CLASSES)):
        cls = CLASSES[i // PER_CLASS]
        if cls in BG_FOLDER:
            pool = by_morph[BG_FOLDER[cls]]
        else:                                    # defects: alternate the two morphologies
            pool = by_morph[list(BG_FOLDER.values())[i % 2]]
        bg = pool[int(rng.integers(len(pool)))]
        ref = recs[int(rng.integers(len(recs)))]
        canvas = cv2.resize(_fit_long_side(cv2.imread(str(CURATED_IMAGES / bg["file"])), 512),
                            (W, H), interpolation=cv2.INTER_AREA)
        seed_i = int(rng.integers(0, 2**31 - 1))
        mask = np.zeros((H, W), np.uint8)
        if cls in BG_FOLDER:
            comp, boxes = canvas, []
        else:
            k = int(min(60, max(1, rng.choice(priors["counts"]))))
            params = sample_params(priors, k, rng, render_px=512,
                                   pbi2_fraction=1.0 if cls == "pbi2" else 0.0,
                                   bg_gray=cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
            res = render(canvas, params, seed=seed_i)
            comp, boxes, mask = res["image"], res["boxes"], res["mask"]
        edges = G.condition(comp, mask)
        ref_img = cv2.resize(cv2.imread(str(CURATED_IMAGES / ref["file"])), (W, H))
        out.append({"i": i, "cls": cls, "morphology": bg["source_folder"], "init": comp, "mask": mask, "edges": edges, "boxes": boxes, "ref": ref_img,
                    "seed": seed_i, "background_file": bg["file"], "background_group": bg["group"],
                    "ip_ref_file": ref["file"]})
    return out


# ----------------------------------------------------------------- pipelines --

def build(name):
    from diffusers import AutoencoderKL, ControlNetModel
    m = MODELS[name]
    kw = dict(dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    if m["variant"]:
        kw["variant"] = m["variant"]
    if m.get("vae_from"):
        sub = None if m["family"] == "sdxl" else "vae"
        kw["vae"] = AutoencoderKL.from_pretrained(m["vae_from"], subfolder=sub, dtype=torch.float16)
    cn = ControlNetModel.from_pretrained(m["cn"], dtype=torch.float16,
                                         **({"variant": "fp16"} if m["family"] == "sdxl" else {}))
    if m["family"] == "sdxl":
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline as P
        kw.pop("safety_checker"); kw.pop("requires_safety_checker")
    else:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline as P
    pipe = P.from_pretrained(m["base"], controlnet=cn, **kw)
    if m["ipa"]:
        pipe.load_ip_adapter(m["ipa"][0], subfolder=m["ipa"][1], weight_name=m["ipa"][2])
        pipe.set_ip_adapter_scale(G.IP_SCALE)
    pipe.enable_sequential_cpu_offload()     # 4 GB card; never attention slicing with IP-Adapter
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_one(pipe, m, lay):
    size = m.get("size", (W, H))
    init = Image.fromarray(cv2.cvtColor(cv2.resize(lay["init"], size), cv2.COLOR_BGR2RGB))
    cond = Image.fromarray(np.stack([cv2.resize(lay["edges"], size, interpolation=cv2.INTER_NEAREST)] * 3, -1))
    kw = dict(prompt=G.PROMPT, negative_prompt=G.NEG, image=init, control_image=cond,
              strength=G.STRENGTH, num_inference_steps=G.STEPS, guidance_scale=G.GUIDANCE,
              controlnet_conditioning_scale=G.CN_SCALE, width=size[0], height=size[1],
              generator=torch.Generator("cpu").manual_seed(lay["seed"]))
    if m["ipa"]:
        kw["ip_adapter_image"] = Image.fromarray(cv2.cvtColor(cv2.resize(lay["ref"], size), cv2.COLOR_BGR2RGB))
    img = pipe(**kw).images[0]
    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    if size != (W, H):
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------- driver --

def run_model(name, layouts, timing):
    m = MODELS[name]
    d = OUT_ROOT / f"_cmp_{name}"
    for sub in ("images", "labels", "cond"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    todo = [l for l in layouts if not (d / "images" / f"cmp_{name}_{l['i']:02d}.jpg").exists()]
    if not todo:
        print(f"[{name}] already complete")
        return
    print(f"[{name}] loading {m['base']} ...")
    t0 = time.time()
    pipe = build(name)
    print(f"[{name}] loaded in {time.time() - t0:.0f}s; generating {len(todo)} images")
    times = []
    with (d / "manifest.jsonl").open("a", encoding="utf-8") as mf:
        for lay in todo:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = generate_one(pipe, m, lay)
            times.append(time.time() - t0)
            stem = f"cmp_{name}_{lay['i']:02d}"
            cv2.imwrite(str(d / "images" / f"{stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(d / "cond" / f"{stem}.png"), lay["edges"])
            (d / "labels" / f"{stem}.txt").write_text(
                "".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in lay["boxes"]), encoding="utf-8")
            mf.write(json.dumps({"stem": stem, "background_file": lay["background_file"],
                                 "background_group": lay["background_group"],
                                 "ip_ref_file": lay["ip_ref_file"], "n_defects": len(lay["boxes"]),
                                 "seed": lay["seed"]}) + "\n")
            print(f"[{name}] {stem} ({lay['cls']}): {times[-1]:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
    timing[name] = {"seconds_per_image": round(float(np.mean(times)), 1),
                    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "base": m["base"], "ip_adapter": bool(m["ipa"]), "size": list(m.get("size", (W, H)))}
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def _draw(img, boxes):
    img = img.copy()
    for c, cx, cy, w, h in boxes:
        p0 = (int((cx - w / 2) * W), int((cy - h / 2) * H))
        p1 = (int((cx + w / 2) * W), int((cy + h / 2) * H))
        cv2.rectangle(img, p0, p1, (0, 255, 0) if c == 1 else (0, 128, 255), 1)
    return img


def sheets(layouts):
    names = [n for n in MODELS if (OUT_ROOT / f"_cmp_{n}" / "images").exists()]
    thumb = (W // 2, H // 2)
    rows = []
    for lay in layouts:
        tiles = [_draw(lay["init"], lay["boxes"])]
        for n in names:
            p = OUT_ROOT / f"_cmp_{n}" / "images" / f"cmp_{n}_{lay['i']:02d}.jpg"
            tiles.append(_draw(cv2.imread(str(p)), lay["boxes"]) if p.exists()
                         else np.zeros((H, W, 3), np.uint8))
        rows.append(np.hstack([cv2.resize(t, thumb, interpolation=cv2.INTER_AREA) for t in tiles]))
    header = np.full((28, thumb[0] * (len(names) + 1), 3), 255, np.uint8)
    for j, t in enumerate(["rendered init"] + names):
        cv2.putText(header, t, (j * thumb[0] + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(HERE / "assets" / "compare_grid.png"), np.vstack([header] + rows))
    for n in names:
        tiles = []
        for lay in layouts:
            p = OUT_ROOT / f"_cmp_{n}" / "images" / f"cmp_{n}_{lay['i']:02d}.jpg"
            if p.exists():
                tiles.append(_draw(cv2.imread(str(p)), lay["boxes"]))
        if len(tiles) == PER_CLASS * len(CLASSES):
            rows_ = []
            for r, cls in enumerate(CLASSES):
                row = np.hstack(tiles[r * PER_CLASS:(r + 1) * PER_CLASS])
                cv2.putText(row, cls, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                rows_.append(row)
            cv2.imwrite(str(HERE / "assets" / f"compare_{n}.png"), np.vstack(rows_))
    print(f"sheets written for {names} -> phase4/assets/compare_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS), help="comma list; default all five")
    ap.add_argument("--sheet", action="store_true", help="only rebuild the contact sheets")
    a = ap.parse_args()
    (HERE / "assets").mkdir(exist_ok=True)
    layouts = make_layouts()
    tpath = HERE / "assets" / "compare_timing.json"
    timing = json.loads(tpath.read_text()) if tpath.exists() else {}
    if not a.sheet:
        for name in a.models.split(","):
            name = name.strip()
            if name not in MODELS:
                sys.exit(f"unknown model '{name}'; choose from {list(MODELS)}")
            try:
                run_model(name, layouts, timing)
            except Exception as e:           # one model failing must not lose the others
                print(f"[{name}] FAILED: {type(e).__name__}: {e}")
                timing[name] = {"error": f"{type(e).__name__}: {e}"}
            tpath.write_text(json.dumps(timing, indent=1))
    sheets(layouts)
    print(json.dumps(timing, indent=1))
