"""
Did the diffusion pool memorise real images? Run before calling anything releasable.

    py -3.10 phase4/memorisation_check.py [--pool diffusion_cn]

For every generated image: DINO ViT-B/16 cosine similarity to (a) its own
background source, (b) its IP-Adapter reference, (c) the nearest OTHER real
train image; all against the real-to-real nearest-neighbour distribution. A
generated image whose nearest real neighbour is closer than the 99th
percentile of real-to-real similarity is flagged. Pixel level: normalised
cross-correlation with its own background at 256 px, > 0.95 flags a
near-copy. Writes outputs/<pool>_memorisation.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.generate import CURATED_IMAGES, OUT_ROOT, _load_split   # noqa: E402


def dino():
    import timm
    m = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev), dev


def embed(model, dev, paths):
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            batch = []
            for p in paths[i:i + 16]:
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                batch.append(((img - mean) / std).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch)).to(dev)
            f = model(x)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def ncc(a, b):
    a = cv2.resize(cv2.imread(str(a), 0), (256, 176)).astype(np.float32)
    b = cv2.resize(cv2.imread(str(b), 0), (256, 176)).astype(np.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))


def main(pool):
    d = OUT_ROOT / pool
    ents = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
    real = _load_split("train")["records"]
    real_paths = [CURATED_IMAGES / r["file"] for r in real]
    idx = {r["file"]: i for i, r in enumerate(real)}
    model, dev = dino()
    R = embed(model, dev, real_paths)
    G = embed(model, dev, [d / "images" / f"{e['stem']}.jpg" for e in ents])

    rr = R @ R.T
    np.fill_diagonal(rr, -1)
    real_nn = rr.max(1)                       # real-to-real nearest neighbour
    thr = float(np.percentile(real_nn, 99))

    S = G @ R.T
    own, ref, other, flags, pix = [], [], [], [], []
    for k, e in enumerate(ents):
        i_bg, i_ref = idx[e["background_file"]], idx[e["ip_ref_file"]]
        own.append(float(S[k, i_bg]))
        ref.append(float(S[k, i_ref]))
        m = S[k].copy()
        m[[i_bg, i_ref]] = -1
        other.append(float(m.max()))
        flags.append(bool(S[k].max() > thr))
        pix.append(ncc(d / "images" / f"{e['stem']}.jpg", CURATED_IMAGES / e["background_file"]))
    out = {
        "pool": pool, "n": len(ents),
        "real_to_real_nn": {"median": float(np.median(real_nn)), "p99": thr},
        "sim_to_own_background": {"median": float(np.median(own)), "max": float(np.max(own))},
        "sim_to_ip_reference": {"median": float(np.median(ref)), "max": float(np.max(ref))},
        "sim_to_nearest_other_real": {"median": float(np.median(other)), "max": float(np.max(other))},
        "flagged_above_p99": int(sum(flags)),
        "pixel_ncc_to_own_background": {"median": float(np.median(pix)), "max": float(np.max(pix)),
                                        "n_above_0.95": int(sum(p > 0.95 for p in pix))},
    }
    (ROOT / "outputs" / f"{pool}_memorisation.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="diffusion_cn")
    main(ap.parse_args().pool)
