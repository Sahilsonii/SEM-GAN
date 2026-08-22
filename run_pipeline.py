import os
import sys
import argparse
import time
import torch

def banner(title):
    print("\n" + "=" * 80)
    print(f"   {title}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train_gan", "inpaint", "train_detector", "eval"])
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation")
    parser.add_argument("--gan_epochs", type=int, default=10)
    parser.add_argument("--inpaint_samples", type=int, default=100)
    parser.add_argument("--detector_epochs", type=int, default=10)
    parser.add_argument("--gan_batch_size", type=int, default=2) # 2 for 4GB VRAM GPU
    parser.add_argument("--detector_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    start_total = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    
    banner("PEROVSKITE SEM GENERATIVE & LIVE DETECTION PIPELINE: INITIALIZING")
    print(f"  Root Dataset   : {args.data_dir}")
    print(f"  Project Output : {args.save_dir}")
    print(f"  Execution Mode : {args.mode.upper()}")
    print(f"  Hardware Device: {device.upper()} ({device_name})")
    
    # -------------------------------------------------------------
    # Stage 1: Train Defect GAN Generator & Generate Samples
    # -------------------------------------------------------------
    if args.mode in ["all", "train_gan"]:
        banner("STAGE 1: TRAINING PEROVSKITE DEFECT GAN WITH DUAL-DOMAIN DISCRIMINATOR")
        from pipeline.train_gan import train_gan_generator
        train_gan_generator(
            data_dir=args.data_dir,
            save_dir=os.path.join(args.save_dir, "checkpoints"),
            output_dir=os.path.join(args.save_dir, "outputs", "gan_generated_samples"),
            epochs=args.gan_epochs,
            batch_size=args.gan_batch_size,
            device=device
        )

    # -------------------------------------------------------------
    # Stage 2: Generative Inpainting & Dataset Expansion
    # -------------------------------------------------------------
    if args.mode in ["all", "inpaint"]:
        banner("STAGE 2: GENERATING EXPANDED DEFECT DATASET VIA INPAINTING")
        from pipeline.generate_synthetic_dataset import generate_expanded_dataset
        exp_dir = os.path.join(args.save_dir, "data", "expanded_dataset")
        generate_expanded_dataset(
            raw_dataset_dir=args.data_dir,
            output_dir=exp_dir,
            num_synthetic_samples=args.inpaint_samples,
            device=device
        )

    # -------------------------------------------------------------
    # Stage 3: Live Evidential Detector Training
    # -------------------------------------------------------------
    if args.mode in ["all", "train_detector"]:
        banner("STAGE 3: TRAINING LIVE EVIDENTIAL DETECTOR (EDL)")
        from pipeline.train_live_detector import train_detector
        
        class TrainArgs:
            data_dir = args.data_dir
            save_dir = os.path.join(args.save_dir, "checkpoints")
            epochs = args.detector_epochs
            batch_size = args.detector_batch_size
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
