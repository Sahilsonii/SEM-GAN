import math
import numpy as np
import torch
try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
except ImportError:
    from skimage.metrics import peak_signal_to_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def evaluate_patch_metrics(sr_tensor, hr_tensor):
    """
    Evaluates PSNR and SSIM for a pair of PyTorch image tensors.
    sr_tensor, hr_tensor: [1, 1, H, W] or [1, H, W] in [-1, 1] range.
    """
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor.squeeze(0)
    if hr_tensor.dim() == 4:
        hr_tensor = hr_tensor.squeeze(0)
        
    # Convert from [-1, 1] tensor to [0, 255] uint8 numpy array
    sr_np = ((sr_tensor.squeeze().detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    hr_np = ((hr_tensor.squeeze().detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    psnr_val = compute_psnr(hr_np, sr_np, data_range=255)
    ssim_val = compute_ssim(hr_np, sr_np, data_range=255)
    
    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val)
    }

if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256).clamp(-1, 1)
    y = x + torch.randn_like(x) * 0.05
    res = evaluate_patch_metrics(y, x)
    print("Sample Metrics:", res)
