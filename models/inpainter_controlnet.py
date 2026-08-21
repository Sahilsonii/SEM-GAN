import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class DefectInpainterUNet(nn.Module):
    """
    Controllable Multi-Scale Inpainting Generator for Perovskite Defect Synthesis.
    Conditioned on pristine background SEM and defect layout binary mask.
    """
    def __init__(self, in_channels=4, out_channels=3, embed_dim=48):
        super().__init__()
        # Input: 3 channels (clean background) + 1 channel (defect layout mask) = 4 channels
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck Residual Blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim * 4, kernel_size=3, padding=1)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim * 4, kernel_size=3, padding=1)
        )
        
        # Decoder with Skip Connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 4, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Output normalized image [0, 1]
        )

    def forward(self, bg_img, defect_mask):
        # bg_img: [B, 3, H, W]
        # defect_mask: [B, 1, H, W]
        x_in = torch.cat([bg_img, defect_mask], dim=1)
        
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        r = e3 + self.res1(e3)
        r = r + self.res2(r)
        
        d2 = self.dec2(r)
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        
        out = self.out_conv(torch.cat([d1, e1], dim=1))
        
        # Blend: Keep background where mask is 0, synthesize defect where mask is 1
        blended = bg_img * (1.0 - defect_mask) + out * defect_mask
        return blended

def generate_random_defect_layout(H=768, W=1024, max_defects=15):
    """
    Synthesizes a realistic random bounding box defect layout.
    Returns: binary mask [H, W] and list of YOLO boxes [cid, cx, cy, w, h].
    """
    mask = np.zeros((H, W), dtype=np.float32)
    boxes = []
    
    num_defects = np.random.randint(3, max_defects + 1)
    for _ in range(num_defects):
        cid = np.random.choice([0, 1, 2], p=[0.35, 0.35, 0.30]) # PbI2, 3D_pinholes, 3D-2D_pinholes
        
        bw = np.random.uniform(0.015, 0.07) # width
        bh = bw * np.random.uniform(0.8, 1.2) # height
        cx = np.random.uniform(bw/2 + 0.02, 1.0 - bw/2 - 0.02)
        cy = np.random.uniform(bh/2 + 0.02, 1.0 - bh/2 - 0.02)
        
        x1 = int((cx - bw/2) * W)
        x2 = int((cx + bw/2) * W)
        y1 = int((cy - bh/2) * H)
        y2 = int((cy + bh/2) * H)
        
        mask[y1:y2, x1:x2] = 1.0
        boxes.append([cid, cx, cy, bw, bh])
        
    return mask, boxes

if __name__ == "__main__":
    model = DefectInpainterUNet()
    bg = torch.randn(2, 3, 256, 256).clamp(0, 1)
    mask = torch.zeros(2, 1, 256, 256)
    mask[:, :, 50:100, 50:100] = 1.0
    
    out = model(bg, mask)
    print(f"Defect Inpainter Test Passed: Input {bg.shape} -> Output {out.shape}")
