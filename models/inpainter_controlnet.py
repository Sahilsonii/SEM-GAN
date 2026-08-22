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
    Features dynamic spatial interpolation to handle any image dimensions without shape mismatch.
    """
    def __init__(self, in_channels=4, out_channels=3, embed_dim=48):
        super().__init__()
        
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
        
        # Decoder
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
            nn.Sigmoid()
        )

    def forward(self, bg_img, defect_mask):
        # bg_img: [B, 3, H, W]
        # defect_mask: [B, 1, H, W]
        
        # Ensure mask matches background dimensions
        if defect_mask.shape[-2:] != bg_img.shape[-2:]:
            defect_mask = F.interpolate(defect_mask, size=bg_img.shape[-2:], mode='nearest')
            
        x_in = torch.cat([bg_img, defect_mask], dim=1)
        
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        r = e3 + self.res1(e3)
        r = r + self.res2(r)
        
        d2 = self.dec2(r)
        # Dynamic shape matching for skip connection
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        
        # Dynamic shape matching for final output
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        out = self.out_conv(torch.cat([d1, e1], dim=1))
        
        # Ensure output matches input spatial size
        if out.shape[-2:] != bg_img.shape[-2:]:
            out = F.interpolate(out, size=bg_img.shape[-2:], mode='bilinear', align_corners=False)
            
        blended = bg_img * (1.0 - defect_mask) + out * defect_mask
        return blended

def generate_random_defect_layout(H=512, W=512, max_defects=12):
    """
    Synthesizes a realistic random bounding box defect layout.
    Returns: binary mask [H, W] and list of YOLO boxes [cid, cx, cy, w, h].
    """
    mask = np.zeros((H, W), dtype=np.float32)
    boxes = []
    
    num_defects = np.random.randint(3, max_defects + 1)
    for _ in range(num_defects):
        cid = int(np.random.choice([0, 1, 2], p=[0.35, 0.35, 0.30])) # PbI2, 3D_pinholes, 3D-2D_pinholes
        
        bw = np.random.uniform(0.02, 0.08) # width
        bh = bw * np.random.uniform(0.8, 1.2) # height
        cx = np.random.uniform(bw/2 + 0.02, 1.0 - bw/2 - 0.02)
        cy = np.random.uniform(bh/2 + 0.02, 1.0 - bh/2 - 0.02)
        
        x1 = max(0, int((cx - bw/2) * W))
        x2 = min(W, int((cx + bw/2) * W))
        y1 = max(0, int((cy - bh/2) * H))
        y2 = min(H, int((cy + bh/2) * H))
        
        mask[y1:y2, x1:x2] = 1.0
        boxes.append([cid, cx, cy, bw, bh])
        
    return mask, boxes

if __name__ == "__main__":
    model = DefectInpainterUNet()
    bg = torch.randn(2, 3, 513, 513).clamp(0, 1) # Test with odd dimension
    mask = torch.zeros(2, 1, 513, 513)
    mask[:, :, 50:100, 50:100] = 1.0
    
    out = model(bg, mask)
    print(f"Defect Inpainter Odd Shape Test Passed: Input {bg.shape} -> Output {out.shape}")
