"""
run_pipeline.py — Master One-Click Execution Pipeline for Perovskite SEM GAN & Live Detection
=============================================================================================
Usage:
    py run_pipeline.py --mode all
    py run_pipeline.py --mode inpaint
    py run_pipeline.py --mode train
    py run_pipeline.py --mode eval
"""

import os
import sys
import argparse
import time

def banner(title):
    print("\n" + "=" * 80)
    print(f"   {title}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "inpaint", "train", "eval"])
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation")
    parser.add_argument("--inpaint_samples", type=int, default=50) # default 50 for quick run
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    start_total = time.time()
    
    banner("PEROVSKITE SEM GENERATIVE & LIVE DETECTION PIPELINE: INITIALIZING")
    print(f"  Root Dataset  : {args.data_dir}")
    print(f"  Project Output: {args.save_dir}")
    print(f"  Execution Mode: {args.mode.upper()}")
    
    # -------------------------------------------------------------
    # Stage 1 & 2: Physics & Generative Inpainting
    # -------------------------------------------------------------
    if args.mode in ["all", "inpaint"]:
        banner("STAGE 1 & 2: GENERATIVE DEFECT INPAINTING & DATASET EXPANSION")
        from pipeline.generate_synthetic_dataset import generate_expanded_dataset
        exp_dir = os.path.join(args.save_dir, "data", "expanded_dataset")
        generate_expanded_dataset(
            raw_dataset_dir=args.data_dir,
            output_dir=exp_dir,
            num_synthetic_samples=args.inpaint_samples
        )

    # -------------------------------------------------------------
    # Stage 3: Live Evidential Detector Training
    # -------------------------------------------------------------
    if args.mode in ["all", "train"]:
        banner("STAGE 3: TRAINING LIVE EVIDENTIAL DETECTOR (EDL)")
        from pipeline.train_live_detector import train_detector
        
        class TrainArgs:
            data_dir = args.data_dir
            save_dir = os.path.join(args.save_dir, "checkpoints")
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.lr
            
        train_detector(TrainArgs)

    # -------------------------------------------------------------
    # Stage 4: Master Research Benchmark & Physical Evaluation
    # -------------------------------------------------------------
    if args.mode in ["all", "eval"]:
        banner("STAGE 4: MASTER RESEARCH BENCHMARK & EVALUATION")
        from evaluation.benchmark_full import run_master_benchmark
        
        class EvalArgs:
            data_dir = args.data_dir
            checkpoint = os.path.join(args.save_dir, "checkpoints", "best_live_detector.pth")
            save_dir = os.path.join(args.save_dir, "outputs", "benchmark_tables")
            
        run_master_benchmark(EvalArgs)

    total_time = time.time() - start_total
    banner(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS!")

if __name__ == "__main__":
    main()
