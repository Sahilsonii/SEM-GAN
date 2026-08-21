import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import SEMPatchDataset
from models.generator import SEMSwinIRGenerator
from metrics.perceptual import evaluate_patch_metrics
from metrics.frequency import compute_spectral_error

def evaluate_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running SEM Super-Resolution Benchmark Evaluation on Device: {device} ---")
    
    dataset = SEMPatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        stride=args.stride * 2, # Sparse test sampling
        scale_factor=args.scale_factor,
        is_train=False
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load trained generator
    netG = SEMSwinIRGenerator(in_channels=1, out_channels=1, scale_factor=args.scale_factor).to(device)
    if os.path.exists(args.checkpoint):
        netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded Generator Checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found! Running evaluation with randomly initialized weights.")
    netG.eval()
    
    bicubic_psnr, bicubic_ssim, bicubic_spec = [], [], []
    gan_psnr, gan_ssim, gan_spec = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            
            # 1. Bicubic Baseline Interpolation
            bicubic_sr = F.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)
            b_m = evaluate_patch_metrics(bicubic_sr, hr)
            b_s = compute_spectral_error(bicubic_sr, hr)
            bicubic_psnr.append(b_m["psnr"])
            bicubic_ssim.append(b_m["ssim"])
            bicubic_spec.append(b_s)
            
            # 2. Proposed Transformer-GAN Super-Resolution
            gan_sr = netG(lr)
            g_m = evaluate_patch_metrics(gan_sr, hr)
            g_s = compute_spectral_error(gan_sr, hr)
            gan_psnr.append(g_m["psnr"])
            gan_ssim.append(g_m["ssim"])
            gan_spec.append(g_s)
            
    print("\n" + "=" * 65)
    print("      SEM SUPER-RESOLUTION COMPUTER VISION BENCHMARK REPORT      ")
    print("=" * 65)
    print(f"{'Method':<25} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'Spectral Error':<20}")
    print("-" * 65)
    print(f"{'Bicubic Interpolation':<25} | {np.mean(bicubic_psnr):<12.2f} | {np.mean(bicubic_ssim):<10.4f} | {np.mean(bicubic_spec):<20.4f}")
    print(f"{'Proposed Transformer-GAN':<25} | {np.mean(gan_psnr):<12.2f} | {np.mean(gan_ssim):<10.4f} | {np.mean(gan_spec):<20.4f}")
    print("=" * 65)
    
    # Save results to a markdown summary
    out_md = os.path.join(os.path.dirname(args.checkpoint), "evaluation_results.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Quantitative Evaluation Benchmark Results\n\n")
        f.write(f"Evaluating scale factor: **{args.scale_factor}x** on **{len(dataset)}** test patches.\n\n")
        f.write("| Method | PSNR (dB) ↑ | SSIM ↑ | Spectral Log Error ↓ |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Bicubic Baseline | {np.mean(bicubic_psnr):.2f} | {np.mean(bicubic_ssim):.4f} | {np.mean(bicubic_spec):.4f} |\n")
        f.write(f"| **Proposed Transformer-GAN** | **{np.mean(gan_psnr):.2f}** | **{np.mean(gan_ssim):.4f}** | **{np.mean(gan_spec):.4f}** |\n")
    print(f"Results successfully saved to {out_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SEM Transformer-GAN Super-Resolution Benchmark")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM\3D pin hole")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Users\Sahil\.gemini\antigravity-ide\scratch\sem_gan_project\checkpoints\best_generator.pth")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--scale_factor", type=int, default=2)
    
    args = parser.parse_args()
    evaluate_models(args)
