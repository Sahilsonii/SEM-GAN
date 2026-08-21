import numpy as np
import torch

def compute_radial_power_spectrum(img_np):
    """
    Computes the 1D Radial Power Spectrum Density (RPSD) of a 2D grayscale image array.
    """
    H, W = img_np.shape
    # Compute 2D FFT & Shift zero-frequency component to the center
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) ** 2
    
    # Calculate radial distance coordinates from center
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    r = np.sqrt(x*x + y*y).astype(int)
    
    # Average power values at each radial distance r
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    
    return radial_profile

def compute_spectral_error(sr_tensor, hr_tensor):
    """
    Calculates the relative L1 error between radial power spectrums of SR and HR images.
    """
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor.squeeze(0)
    if hr_tensor.dim() == 4:
        hr_tensor = hr_tensor.squeeze(0)
        
    sr_np = ((sr_tensor.squeeze().detach().cpu().numpy() + 1.0) * 127.5)
    hr_np = ((hr_tensor.squeeze().detach().cpu().numpy() + 1.0) * 127.5)
    
    ps_sr = compute_radial_power_spectrum(sr_np)
    ps_hr = compute_radial_power_spectrum(hr_np)
    
    min_len = min(len(ps_sr), len(ps_hr))
    ps_sr, ps_hr = ps_sr[:min_len], ps_hr[:min_len]
    
    # Log-scale spectral error
    log_err = np.mean(np.abs(np.log1p(ps_sr) - np.log1p(ps_hr)))
    return float(log_err)

if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    y = x + torch.randn_like(x) * 0.1
    err = compute_spectral_error(y, x)
    print("Spectral Log Error:", err)
