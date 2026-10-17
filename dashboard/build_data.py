"""Pack results, split stats and a sample of images into dashboard/data.js + dashboard/img/.
Run from anywhere:  python build_data.py   (reads the repo one level up; writes next to itself)."""
import csv
import json
import os
import re
import shutil
import statistics as st

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(HERE, "img")
FIG = os.path.join(HERE, "fig")
MAXW = 800

# (group key, label, family) - family drives bar color in the UI
GROUPS = [
    ("real_only_s", "Real only · 100 epochs (2,000 steps)", "real"),
    ("real_only_steps8300", "Real only · 415 epochs (8,300 steps)", "real"),
    ("real_only_yolov8s", "YOLOv8s · real only", "detector"),
    ("real_only_rtdetr", "RT-DETR-L · real only", "detector"),
    ("real_only_p2", "YOLOv8s + P2 head · real only", "detector"),
    ("scale005_s", "Renderer + GAN 5% · 100 epochs", "gan"),
    ("scale005_steps2000", "Renderer + GAN 5% · 2,000 steps", "gan"),
    ("scale_005_bal_steps8300", "Renderer + GAN balanced · 8,300 steps", "gan"),
    ("diffusion_cn_steps8300", "Stable Diffusion + real · 8,300 steps", "diffusion"),
    ("diffusion_cn_only", "Stable Diffusion only", "diffusion"),
    ("layout_transfer", "Re-used real layouts · 8,300 steps", "reuse"),
    ("augment_defective", "Re-used defective pictures · 8,300 steps", "reuse"),
    ("copypaste", "Pasted real defects · 2,000 steps", "reuse"),
    ("synth_only_005_bal", "Fake pictures only", "gan"),
]


def group_of(name):
    # longest-prefix match so scale005_steps2000 is not swallowed by scale005_s
    hits = [g for g in GROUPS if name.startswith(g[0])]
    return max(hits, key=lambda g: len(g[0]))[0] if hits else None


def ms(xs):
    return [round(st.mean(xs), 4), round(st.stdev(xs), 4) if len(xs) > 1 else 0.0]


def thumb(src, dst):
    im = Image.open(src).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=80)


def boxes(label_path):
    if not os.path.exists(label_path):
        return []
    out = []
    for ln in open(label_path):
        p = ln.split()
        if len(p) >= 5:
            out.append([int(p[0])] + [round(float(v), 4) for v in p[1:5]])
    return out


def pick(files, n):
    step = max(1, len(files) // n)
    return files[::step][:n]


def main():
    shutil.rmtree(IMG, ignore_errors=True)
    os.makedirs(IMG)
    os.makedirs(FIG, exist_ok=True)

    final = json.load(open(os.path.join(ROOT, "outputs/final_test_results.json")))
    master = {r["exp_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "outputs/master_results.csv")))}
    labels = dict((g[0], (g[1], g[2])) for g in GROUPS)

    by = {}
    for name, r in final.items():
        g = group_of(name)
        if not g:
            continue
        exp = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(r["checkpoint_path"].replace("\\", "/")))))
        m = master.get(exp, {})
        by.setdefault(g, []).append({
            "seed": re.search(r"(?:_s|seed)(\d+)$", name).group(1) if re.search(r"(?:_s|seed)(\d+)$", name) else "42",
            "map50": r["mAP50"], "map5095": r["mAP50_95"], "p": r["precision"], "r": r["recall"],
            "bins": {k: v["ap"] for k, v in r["per_bin_at50"].items()},
            "recbins": {k: v["recall"] for k, v in r["per_bin_at50"].items()},
            "val": float(m["mAP50"]) if m.get("mAP50") else None,
            "epochs": int(m["epochs"]) if m.get("epochs") else None,
            "params": float(m["params_M"]) if m.get("params_M") else None,
            "secs": float(m["train_seconds"]) if m.get("train_seconds") else None,
            "model": m.get("model"), "npred": r["n_pred"],
        })
    runs = []
    for key, label, fam in GROUPS:
        rs = by.get(key)
        if not rs:
            continue
        pick_ = lambda f: [x[f] for x in rs if x[f] is not None]
        bins = {b: ms([x["bins"][b] for x in rs]) for b in rs[0]["bins"]}
        runs.append({
            "key": key, "label": label, "family": fam, "n": len(rs),
            "map50": ms(pick_("map50")), "map5095": ms(pick_("map5095")),
            "p": ms(pick_("p")), "r": ms(pick_("r")), "bins": bins,
            "val": ms(pick_("val")) if pick_("val") else None,
            "epochs": round(st.mean(pick_("epochs"))) if pick_("epochs") else None,
            "params": pick_("params")[0] if pick_("params") else None,
            "mins": round(st.mean(pick_("secs")) / 60, 1) if pick_("secs") else None,
            "npred": round(st.mean(x["npred"] for x in rs)), "model": rs[0]["model"], "seeds": [x["map50"] for x in rs],
        })
    mdc = json.load(open(os.path.join(ROOT, "outputs/microdefectcv_baseline_test.json")))
    runs.append({"key": "microdefectcv", "label": "Classical MicroDefectCV (no training)", "family": "classical",
                 "n": 1, "map50": [mdc["mAP50"], 0], "map5095": [mdc["mAP50_95"], 0], "p": [mdc["precision"], 0],
                 "r": [mdc["recall"], 0], "bins": {k: [v["ap"], 0] for k, v in mdc["per_bin_at50"].items()},
                 "val": None, "epochs": 0, "params": 0, "mins": 0, "npred": mdc["n_pred"], "model": "classical", "seeds": [mdc["mAP50"]]})

    split = json.load(open(os.path.join(ROOT, "data/splits/splits_manifest.json")))
    ngt = {k: v["n_gt"] for k, v in final["real_only_s1"]["per_bin_at50"].items()}

    # ---- gallery ----
    gallery = []

    def add(set_, title, src_dir, stem_filter=None, n=8, lab_dir=None, sub=None):
        d = os.path.join(ROOT, src_dir, "images")
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
        if stem_filter:
            files = [f for f in files if stem_filter in f]
        for f in pick(files, n):
            stem = os.path.splitext(f)[0]
            dst = f"{set_}_{stem}.jpg"
            thumb(os.path.join(d, f), os.path.join(IMG, dst))
            gallery.append({"set": set_, "title": title, "file": dst, "stem": stem,
                            "boxes": boxes(os.path.join(ROOT, src_dir, "labels", stem + ".txt"))})

    # real FESEM pictures that carry labels
    for cls in ("class0_pbI2", "class1_3D_pinholes"):
        d = os.path.join(ROOT, "data/curated/images", cls)
        files = [f for f in sorted(os.listdir(d)) if boxes(os.path.join(ROOT, "data/curated/labels", cls, os.path.splitext(f)[0] + ".txt"))]
        for f in pick(files, 6):
            stem = os.path.splitext(f)[0]
            dst = f"real_{cls}_{stem}.jpg"
            thumb(os.path.join(d, f), os.path.join(IMG, dst))
            gallery.append({"set": "real", "title": "Real FESEM", "file": dst, "stem": f"{cls}/{stem}",
                            "boxes": boxes(os.path.join(ROOT, "data/curated/labels", cls, stem + ".txt"))})
    # same layouts: rendered vs GAN-refined (identical stems)
    for kind in ("pbi2", "pinhole"):
        add("rendered", "Rendered", "data/synthetic/controlled_" + kind, n=8)
        add("refined", "Rendered + GAN", "data/synthetic/refined", stem_filter="controlled_" + kind, n=8)
    add("diffusion", "Stable Diffusion", "data/synthetic/diffusion_cn", n=8)
    add("copypaste", "Pasted real defects", "data/synthetic/copypaste", n=6)
    add("augment", "Re-used defective pictures", "data/synthetic/augment_defective", n=6)
    add("layout", "Re-used real layouts", "data/synthetic/layout_transfer", n=6)

    # ---- figures reused from the report (already white-flattened) ----
    keep = ["problem_closeups", "funnel", "leakage_diagram", "split_chart", "methods_gallery", "pipeline_overall", "renderer",
            "refiner", "diffusion_pipeline", "diffusion_triplet", "diffusion_samples", "models_compare", "arch_chart",
            "compute_matching", "fair_chart", "all_chart", "sizes_chart", "found_chart", "benchmark_chart", "scores_chart",
            "seeds_chart", "confusion_models", "confusion_seeds", "gap_chart", "prisma", "lit_to_gap"]
    for k in keep:
        s = os.path.join(ROOT, "dissertation_phase1/figures", k + ".png")
        if os.path.exists(s):
            shutil.copy(s, os.path.join(FIG, k + ".png"))


    # ---- Stable Diffusion audit: same drawn layout through each model, plus pool pairs ----
    sd = {"layouts": [], "pool": []}
    models = ["sd15", "realvis", "dreamshaper", "sdxl"]
    for i in (10, 11, 15, 16):
        stem = f"cmp_sd15_{i:02d}"
        entry = {"idx": i, "boxes": boxes(os.path.join(ROOT, "data/synthetic/_cmp_sd15/labels", stem + ".txt")), "out": {}}
        thumb(os.path.join(ROOT, "data/synthetic/_cmp_sd15/cond", stem + ".png"), os.path.join(IMG, f"sdcond_{i}.jpg"))
        entry["cond"] = f"sdcond_{i}.jpg"
        for m in models:
            thumb(os.path.join(ROOT, f"data/synthetic/_cmp_{m}/images/cmp_{m}_{i:02d}.jpg"), os.path.join(IMG, f"sd_{m}_{i}.jpg"))
            entry["out"][m] = f"sd_{m}_{i}.jpg"
        sd["layouts"].append(entry)
    pdir = os.path.join(ROOT, "data/synthetic/diffusion_cn")
    for stem in [f"dc_diffusion_cn_{n:05d}" for n in (0, 40, 80, 120, 160, 200)]:
        thumb(os.path.join(pdir, "cond", stem + "_init.jpg"), os.path.join(IMG, f"pool_init_{stem}.jpg"))
        thumb(os.path.join(pdir, "images", stem + ".jpg"), os.path.join(IMG, f"pool_out_{stem}.jpg"))
        sd["pool"].append({"stem": stem, "init": f"pool_init_{stem}.jpg", "out": f"pool_out_{stem}.jpg",
                           "boxes": boxes(os.path.join(pdir, "labels", stem + ".txt"))})
    for src, dst in (("compare_grid.png", "sd_compare_grid.png"), ("sweep_s25_panel.png", "sd_sweep_panel.png"), ("smoke_pool_panel.png", "sd_pool_panel.png")):
        p = os.path.join(ROOT, "phase4/assets", src)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB"); im.save(os.path.join(FIG, dst))
    timing = json.load(open(os.path.join(ROOT, "phase4/assets/compare_timing.json")))

    data = {"sd": sd, "timing": timing, "runs": runs, "split": {k: split["summary"][k] for k in ("train", "val", "test")},
            "audit": {k: split["original_split_audit"][k] for k in ("groups", "train_val_overlap", "train_test_overlap", "val_test_overlap")},
            "ngt": ngt, "gallery": gallery}
    with open(os.path.join(HERE, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.DATA=" + json.dumps(data, ensure_ascii=False) + ";")
    print(len(runs), "run groups |", len(gallery), "images |", round(sum(os.path.getsize(os.path.join(IMG, x)) for x in os.listdir(IMG)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
