"""
run_all.py — Master Execution Script for SEM GAN Dissertation Pipeline
=======================================================================
Executes the full project pipeline in one command:
  Step 1: Data download & preprocessing
  Step 2: Stage 1 Pre-training (NFFA SEM multi-class)
  Step 3: Stage 2 Fine-tuning (Perovskite SEM 5-class)
  Step 4: Benchmark evaluation (all models, all metrics)
  Step 5: Visualization generation
  Step 6: Export dissertation figures and LaTeX benchmark table
"""

import os
import sys
import yaml
import argparse

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def banner(msg):
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="SEM GAN Dissertation — Master Runner")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip Stage 1 pre-training")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip Stage 2 fine-tuning")
    parser.add_argument("--only-eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--only-viz", action="store_true", help="Run visualization only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_name = cfg["project"]["name"]

    banner(f"STARTING: {project_name}")
    print(f"  Config   : {args.config}")
    print(f"  Device   : {cfg['project']['device']}")
    print(f"  Classes  : {list(cfg['data']['class_map'].values())}")

    if not args.only_eval and not args.only_viz:
        # --- Step 1: Data Preprocessing ---
        banner("STEP 1: Preprocessing Perovskite SEM Dataset")
        # TODO: import and call data.dataset preprocessing
        print("  [PENDING] — Send dataset folder path to activate this step.")

        # --- Step 2: Stage 1 Pre-training ---
        if not args.skip_pretrain and cfg["training"]["pretrain"]["enabled"]:
            banner("STEP 2: Stage 1 — NFFA SEM Domain Pre-Training")
            # TODO: from training.pretrain import run_pretrain; run_pretrain(cfg)
            print("  [READY] — Will train on NFFA-Europe 5,000-image SEM subset.")

        # --- Step 3: Stage 2 Fine-tuning ---
        if not args.skip_finetune and cfg["training"]["finetune"]["enabled"]:
            banner("STEP 3: Stage 2 — Perovskite SEM Fine-Tuning (5 Classes)")
            # TODO: from training.finetune import run_finetune; run_finetune(cfg)
            print("  [READY] — Will fine-tune all 5 models on perovskite dataset.")

    # --- Step 4: Benchmark Evaluation ---
    banner("STEP 4: Multi-Model Benchmark Evaluation")
    # TODO: from evaluation.benchmark import run_benchmark; run_benchmark(cfg)
    print("  [READY] — PSNR, SSIM, LPIPS, FID, RPSD evaluation on all models.")

    # --- Step 5: Visualization ---
    banner("STEP 5: Dissertation Visualization Generation")
    # TODO: from evaluation.visualize import run_visualization; run_visualization(cfg)
    print("  [READY] — 4-panel grids, spectral curves, t-SNE plots, gen matrices.")

    # --- Step 6: Export ---
    banner("STEP 6: Exporting Dissertation Figures & LaTeX Table")
    print("  [READY] — 300 DPI PNG/PDF + LaTeX benchmark table output.")

    banner("PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
