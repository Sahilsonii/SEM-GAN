import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data.dataset import SEMPatchDataset
from models.generator import SEMSwinIRGenerator
from metrics.frequency import compute_radial_power_spectrum

def generate_visualizations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Generating Visualization Plots on Device: {device} ---")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    dataset = SEMPatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        stride=args.stride * 2,
        scale_factor=args.scale_factor,
        is_train=False
    )
    
    netG = SEMSwinIRGenerator(in_channels=1, out_channels=1, scale_factor=args.scale_factor).to(device)
    if os.path.exists(args.checkpoint):
        netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded Generator Checkpoint: {args.checkpoint}")
    netG.eval()
    
    indices = [0, min(5, len(dataset)-1), min(10, len(dataset)-1)]
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            lr = sample["lr"].unsqueeze(0).to(device) # [1, 1, LR_H, LR_W]
            hr = sample["hr"].unsqueeze(0).to(device) # [1, 1, HR_H, HR_W]
            
            bicubic = F.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)
            gan_sr = netG(lr)
            
            lr_np = ((lr.squeeze().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            bicubic_np = ((bicubic.squeeze().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            gan_np = ((gan_sr.squeeze().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            hr_np = ((hr.squeeze().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # --- 1. 4-Panel Visual Comparison Plot ---
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(lr_np, cmap='gray')
            axes[0].set_title(f"Low-Res Input ({lr_np.shape[1]}x{lr_np.shape[0]})")
            axes[0].axis('off')
            
            axes[1].imshow(bicubic_np, cmap='gray')
            axes[1].set_title(f"Bicubic ({bicubic_np.shape[1]}x{bicubic_np.shape[0]})")
            axes[1].axis('off')
            
            axes[2].imshow(gan_np, cmap='gray')
            axes[2].set_title(f"Proposed Transformer-GAN")
            axes[2].axis('off')
            
            axes[3].imshow(hr_np, cmap='gray')
            axes[3].set_title(f"Ground Truth HR ({hr_np.shape[1]}x{hr_np.shape[0]})")
            axes[3].axis('off')
            
            plt.tight_layout()
            viz_path = os.path.join(args.out_dir, f"visual_comparison_sample_{i+1}.png")
            plt.savefig(viz_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # --- 2. Fourier Radial Power Spectrum Profile ---
            ps_bicubic = compute_radial_power_spectrum(bicubic_np.astype(float))
            ps_gan = compute_radial_power_spectrum(gan_np.astype(float))
            ps_hr = compute_radial_power_spectrum(hr_np.astype(float))
            
            plt.figure(figsize=(7, 4))
            plt.plot(np.log1p(ps_hr[:100]), label='Ground Truth HR', color='black', linewidth=2)
            plt.plot(np.log1p(ps_gan[:100]), label='Proposed Transformer-GAN', color='crimson', linewidth=1.8)
            plt.plot(np.log1p(ps_bicubic[:100]), label='Bicubic Interpolation', color='blue', linestyle='--', linewidth=1.5)
            plt.xlabel('Spatial Radial Frequency Radius $r$')
            plt.ylabel('Log Power Spectrum Density')
            plt.title('Fourier Radial Frequency Spectrum Alignment')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            spec_path = os.path.join(args.out_dir, f"spectral_profile_sample_{i+1}.png")
            plt.savefig(spec_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Generated visual and spectral plots for Sample {i+1} -> {viz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visualization Plots for SEM GAN")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM\3D pin hole")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Users\Sahil\.gemini\antigravity-ide\scratch\sem_gan_project\checkpoints\best_generator.pth")
    parser.add_argument("--out_dir", type=str, default=r"C:\Users\Sahil\.gemini\antigravity-ide\scratch\sem_gan_project\visualizations")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--scale_factor", type=int, default=2)
    
    args = parser.parse_args()
    generate_visualizations(args)
