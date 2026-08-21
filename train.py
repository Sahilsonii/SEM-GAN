import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SEMPatchDataset
from models.generator import SEMSwinIRGenerator
from models.discriminator import DualDomainDiscriminator
from models.losses import SEMCombinedLoss
from metrics.perceptual import evaluate_patch_metrics
from metrics.frequency import compute_spectral_error

def train_sem_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting SEM Transformer-GAN Training on Device: {device} ---")
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    # 1. Dataset & DataLoader Setup
    full_dataset = SEMPatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        scale_factor=args.scale_factor,
        is_train=True
    )
    
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset Loaded: {len(full_dataset)} total patches ({train_size} Train | {val_size} Val)")
    
    # 2. Model, Loss, & Optimizer Instantiation
    netG = SEMSwinIRGenerator(in_channels=1, out_channels=1, scale_factor=args.scale_factor).to(device)
    netD = DualDomainDiscriminator(in_channels=1).to(device)
    
    loss_fn = SEMCombinedLoss().to(device)
    
    optG = torch.optim.AdamW(netG.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    optD = torch.optim.AdamW(netD.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    
    best_psnr = 0.0
    
    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            lr_imgs = batch["lr"].to(device) # [B, 1, LR_H, LR_W]
            hr_imgs = batch["hr"].to(device) # [B, 1, HR_H, HR_W]
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optD.zero_grad()
            sr_imgs = netG(lr_imgs).detach()
            
            spatial_real, fourier_real = netD(hr_imgs)
            spatial_fake, fourier_fake = netD(sr_imgs)
            
            loss_d = loss_fn.forward_d(spatial_real, spatial_fake, fourier_real, fourier_fake)
            loss_d.backward()
            optD.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optG.zero_grad()
            sr_imgs = netG(lr_imgs)
            spatial_fake, fourier_fake = netD(sr_imgs)
            
            loss_g, loss_dict = loss_fn.forward_g(sr_imgs, hr_imgs, spatial_fake, fourier_fake)
            loss_g.backward()
            optG.step()
            
            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()
            
        avg_g_loss = running_g_loss / len(train_loader)
        avg_d_loss = running_d_loss / len(train_loader)
        elapsed = time.time() - start_time
        
        # 4. Validation Loop
        netG.eval()
        val_psnr, val_ssim, val_spectral_err = 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                lr_val = val_batch["lr"].to(device)
                hr_val = val_batch["hr"].to(device)
                sr_val = netG(lr_val)
                
                metrics = evaluate_patch_metrics(sr_val, hr_val)
                spec_err = compute_spectral_error(sr_val, hr_val)
                
                val_psnr += metrics["psnr"]
                val_ssim += metrics["ssim"]
                val_spectral_err += spec_err
                
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_spectral_err /= len(val_loader)
        
        # TensorBoard Logging
        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Metrics/Val_PSNR", val_psnr, epoch)
        writer.add_scalar("Metrics/Val_SSIM", val_ssim, epoch)
        writer.add_scalar("Metrics/Val_Spectral_Log_Error", val_spectral_err, epoch)
        
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | G_Loss: {avg_g_loss:.4f} | D_Loss: {avg_d_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | Spectral Error: {val_spectral_err:.4f} | Time: {elapsed:.1f}s")
        
        # Save Checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt_path = os.path.join(args.save_dir, "best_generator.pth")
            torch.save(netG.state_dict(), ckpt_path)
            print(f"  >>> Best Checkpoint Saved to {ckpt_path} (PSNR: {best_psnr:.2f} dB)")
            
    writer.close()
    print(f"\n--- Training Complete! Best Validation PSNR: {best_psnr:.2f} dB ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SEM Transformer-GAN Super-Resolution")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM\3D pin hole")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Sahil\.gemini\antigravity-ide\scratch\sem_gan_project\checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    
    args = parser.parse_args()
    train_sem_gan(args)
