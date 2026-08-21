import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset_yolo import PerovskiteYOLODataset
from models.live_detector_edl import LiveDetectorEDL
from losses.physics_loss import EvidentialLoss

def train_detector(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"   TRAINING LIVE EVIDENTIAL DETECTOR (EDL) ON DEVICE: {device}   ")
    print("=" * 70)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Dataset & Split
    dataset = PerovskiteYOLODataset(root_dir=args.data_dir)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Loaded {len(dataset)} total samples ({train_size} Train | {val_size} Val)")
    
    # 2. Model & Loss & Optimizer
    model = LiveDetectorEDL(num_classes=5).to(device)
    criterion = EvidentialLoss(num_classes=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_u = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            imgs = batch["image"].to(device) # [B, 3, H, W]
            labels = batch["class_idx"].to(device) # [B]
            
            # One-hot target
            target_one_hot = torch.zeros(len(labels), 5, device=device)
            target_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            
            optimizer.zero_grad()
            out = model(imgs)
            
            loss, u_val = criterion(out["alpha"], target_one_hot, epoch=epoch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_u += u_val
            
        avg_loss = running_loss / len(train_loader)
        avg_u = running_u / len(train_loader)
        elapsed = time.time() - start_time
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_u = 0.0
        
        with torch.no_grad():
            for val_batch in val_loader:
                imgs_v = val_batch["image"].to(device)
                labels_v = val_batch["class_idx"].to(device)
                out_v = model(imgs_v)
                
                preds = torch.argmax(out_v["probs"], dim=1)
                correct += (preds == labels_v).sum().item()
                total += len(labels_v)
                val_u += out_v["uncertainty"].mean().item()
                
        val_acc = (correct / total) * 100.0 if total > 0 else 0.0
        val_avg_u = val_u / len(val_loader)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_loss:.4f} | Train Uncertainty: {avg_u:.4f} | Val Acc: {val_acc:.2f}% | Val Uncertainty: {val_avg_u:.4f} | Time: {elapsed:.1f}s")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, "best_live_detector.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> Checkpoint Saved: {ckpt_path} (Acc: {best_acc:.2f}%)")
            
    print("\n" + "=" * 70)
    print(f"   DETECTOR TRAINING COMPLETE! Best Validation Accuracy: {best_acc:.2f}%   ")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Live Evidential Detector")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    train_detector(args)
