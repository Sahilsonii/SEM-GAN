"""
Feasibility check, nothing else: does SD 1.5 + ControlNet (canny) + IP-Adapter
run on the RTX 3050 Ti (4 GB) at 512 px, FP16, with model CPU offload?

    py -3.10 phase4/smoke.py

Prints peak VRAM and seconds per image, writes phase4/assets/smoke_*.png.
Weights are downloaded once to the Hugging Face cache (~5 GB) and never
leave this machine. No real image is uploaded anywhere.
"""
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
from synth.generate import CURATED_IMAGES, _load_split   # noqa: E402

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN = "lllyasviel/control_v11p_sd15_canny"
IP = ("h94/IP-Adapter", "models", "ip-adapter_sd15.bin")
W, H = 512, 352          # the real 1024x705 aspect, both multiples of 8
PROMPT = ("scanning electron micrograph of a polycrystalline perovskite thin film, "
          "grains and grain boundaries, pinholes, bright lead iodide platelets, "
          "greyscale, sharp, high magnification")
NEG = "color, blurry, painting, cartoon, text, watermark"


def main():
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    recs = [r for r in _load_split("train")["records"] if r["n_boxes"] > 20]
    src = cv2.imread(str(CURATED_IMAGES / recs[0]["file"]), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(CURATED_IMAGES / recs[1]["file"]), cv2.IMREAD_COLOR)
    src = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA)
    canny = cv2.Canny(src, 60, 140)
    canny_img = Image.fromarray(np.stack([canny] * 3, -1))
    ref_img = Image.fromarray(cv2.cvtColor(cv2.resize(ref, (W, H)), cv2.COLOR_BGR2RGB))

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(CN, dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD, controlnet=controlnet, dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False)
    pipe.load_ip_adapter(IP[0], subfolder=IP[1], weight_name=IP[2])
    pipe.set_ip_adapter_scale(0.6)
    pipe.enable_model_cpu_offload()
    # NOT enable_attention_slicing(): it swaps every attention processor for a
    # sliced one and throws away the IP-Adapter processors, which then receive
    # the (text, image) tuple and crash. Torch 2 SDPA is memory-light already.
    pipe.set_progress_bar_config(disable=True)
    print(f"[smoke] weights loaded in {time.time() - t0:.0f}s")

    HERE.joinpath("assets").mkdir(exist_ok=True)
    canny_img.save(HERE / "assets/smoke_canny.png")
    ref_img.save(HERE / "assets/smoke_ref.png")
    out = {}
    for tag, steps in (("warmup", 4), ("run", 20)):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        img = pipe(PROMPT, negative_prompt=NEG, image=canny_img, ip_adapter_image=ref_img,
                   num_inference_steps=steps, guidance_scale=6.0,
                   controlnet_conditioning_scale=0.8, width=W, height=H,
                   generator=torch.Generator("cpu").manual_seed(42)).images[0]
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        out[tag] = {"steps": steps, "seconds": round(dt, 1), "peak_vram_gib": round(peak, 2)}
        print(f"[smoke] {tag}: {steps} steps in {dt:.1f}s, peak VRAM {peak:.2f} GiB")
        img.save(HERE / f"assets/smoke_{tag}.png")
    (HERE / "assets/smoke.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
