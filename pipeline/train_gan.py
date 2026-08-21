import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from data.dataset_yolo import PerovskiteYOLODataset, yolo_collate_fn
from models.inpainter_controlnet import DefectInpainterUNet, generate_random_defect_layout
from models.discriminator import DualDomainDiscriminator
from losses.physics_loss import PhysicsFourierLoss

def train_gan_generator(
    data_dir=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset",
    save_dir=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\checkpoints",
    output_dir=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\outputs\gan_generated_samples",
    epochs=10,
    batch_size=4,
    lr_g=1e-4,
    lr_d=1e-4,
    device="cpu"
):
    print("=" * 75)
    print(f"   TRAINING PEROVSKITE DEFECT GENERATIVE ADVERSARIAL NETWORK (GAN)   ")
    print(f"   Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}   ")
    print("=" * 75)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Dataset
    dataset = PerovskiteYOLODataset(root_dir=data_dir)
    clean_canvases = dataset.get_clean_canvases()
    print(f"Loaded {len(dataset)} dataset samples ({len(clean_canvases)} pristine background canvases)")
    
    # 2. Generator & Dual-Domain Discriminator
    netG = DefectInpainterUNet(in_channels=4, out_channels=3, embed_dim=48).to(device)
    netD = DualDomainDiscriminator(in_channels=3, ndf=32).to(device)
    
    # 3. Losses & Optimizers
    l1_loss = nn.L1Loss()
    fft_loss = PhysicsFourierLoss()
    
    optG = torch.optim.AdamW(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn)
    
    for epoch in range(1, epochs + 1):
        netG.train()
        netD.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        start_time = time.time()
        
        for batch in loader:
            real_imgs = batch["image"].to(device) # [B, 3, H, W]
            B, C, H, W = real_imgs.shape
            
            # Generate defect layout condition masks
            masks = torch.zeros((B, 1, H, W), device=device)
            for b in range(B):
                mask_np, _ = generate_random_defect_layout(H=H, W=W, max_defects=8)
                masks[b, 0] = torch.from_numpy(mask_np).to(device)
                
            # ---------------------
            # Train Discriminator
            # ---------------------
            optD.zero_grad()
            with torch.no_grad():
                fake_imgs = netG(real_imgs, masks)
                
            s_real, f_real = netD(real_imgs)
            s_fake, f_fake = netD(fake_imgs.detach())
            
            loss_d_real = torch.mean(F.relu(1.0 - s_real)) + torch.mean(F.relu(1.0 - f_real))
            loss_d_fake = torch.mean(F.relu(1.0 + s_fake)) + torch.mean(F.relu(1.0 + f_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            
            loss_d.backward()
            optD.step()
            
            # ---------------------
            # Train Generator (G)
            # ---------------------
            optG.zero_grad()
            fake_imgs = netG(real_imgs, masks)
            s_fake, f_fake = netD(fake_imgs)
            
            loss_g_adv = -torch.mean(s_fake) - torch.mean(f_fake)
            loss_g_pixel = l1_loss(fake_imgs * (1.0 - masks), real_imgs * (1.0 - masks))
            loss_g_fft = fft_loss(fake_imgs, real_imgs)
            
            loss_g = 0.01 * loss_g_adv + 1.0 * loss_g_pixel + 0.1 * loss_g_fft
            
            loss_g.backward()
            optG.step()
            
            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()
            
        avg_g = running_g_loss / len(loader)
        avg_d = running_d_loss / len(loader)
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | G_Loss: {avg_g:.4f} | D_Loss: {avg_d:.4f} | Physics FFT Loss: {loss_g_fft.item():.4f} | Time: {elapsed:.1f}s")
        
        # Save sample generated images every 2 epochs
        if epoch % 2 == 0 or epoch == epochs:
            netG.eval()
            with torch.no_grad():
                sample_out = fake_imgs[0].permute(1, 2, 0).cpu().numpy()
                sample_u8 = (sample_out * 255.0).clip(0, 255).astype(np.uint8)
                sample_path = os.path.join(output_dir, f"gan_generated_epoch_{epoch:02d}.jpg")
                Image.fromarray(sample_u8).save(sample_path)
                print(f"  >>> Sample Generated GAN Image Saved: {sample_path}")
                
    # Save Generator Checkpoint
    ckpt_path = os.path.join(save_dir, "best_gan_generator.pth")
    torch.save(netG.state_dict(), ckpt_path)
    print("\n" + "=" * 75)
    print(f"   GAN TRAINING COMPLETE! Checkpoint Saved to: {ckpt_path}   ")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Perovskite Defect GAN")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    train_gan_generator(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
