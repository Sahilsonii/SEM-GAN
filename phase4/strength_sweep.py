"""How much of the rendered defect survives img2img at each strength? sd15, the 10 defect audit layouts."""
import sys, json, time
import cv2, numpy as np, torch
from PIL import Image
sys.path.insert(0, 'phase4'); sys.path.insert(0, '.')
import compare_models as c
import generate_diffusion as G

def contrast(img, box):
    H, W = img.shape
    cl, cx, cy, w, h = box
    x0, x1 = int((cx-w/2)*W), int(np.ceil((cx+w/2)*W)); y0, y1 = int((cy-h/2)*H), int(np.ceil((cy+h/2)*H))
    if x1-x0 < 2 or y1-y0 < 2: return None
    m = 4; ring = img[max(0,y0-m):y1+m, max(0,x0-m):x1+m].astype(float); inner = img[y0:y1, x0:x1].astype(float)
    return inner.mean() - (ring.sum()-inner.sum())/max(ring.size-inner.size, 1)

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--boost", action="store_true"); ap.add_argument("--cn", type=float, default=0.8)
ap.add_argument("--tag", default="")
ap.add_argument("--strengths", default="0.25,0.35,0.5")
a = ap.parse_args()
STRENGTHS = [float(x) for x in a.strengths.split(",")]
G.EDGE_BOOST = a.boost; G.CN_SCALE = a.cn
L = [l for l in c.make_layouts() if l["boxes"]]
pipe = c.build("sd15"); m = c.MODELS["sd15"]
res = {"boost": a.boost, "cn_scale": a.cn}
for strength in STRENGTHS:
    G.STRENGTH = strength
    kept = {0: [], 1: []}; t0 = time.time()
    for lay in L:
        out = cv2.cvtColor(c.generate_one(pipe, m, lay), cv2.COLOR_BGR2GRAY)
        init = cv2.cvtColor(lay["init"], cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"phase4/assets/sweep{a.tag}_s{int(strength*100)}_{lay['i']:02d}.jpg", out)
        for b in lay["boxes"]:
            ci, co = contrast(init, b), contrast(out, b)
            if ci is None or abs(ci) < 3: continue
            kept[int(b[0])].append(bool(np.sign(co) == np.sign(ci) and abs(co) >= 0.5*abs(ci)))
    res[strength] = {"pinhole_kept": float(np.mean(kept[1])), "pbi2_kept": float(np.mean(kept[0])), "s_per_img": (time.time()-t0)/len(L)}
    print(f"strength {strength}: pinhole kept {100*res[strength]['pinhole_kept']:.0f}%  pbi2 kept {100*res[strength]['pbi2_kept']:.0f}%  {res[strength]['s_per_img']:.1f} s/img", flush=True)
json.dump(res, open(f"phase4/assets/strength_sweep{a.tag}.json", "w"), indent=1)
