"""Radial power spectrum - domain-gap Level 2 (kept from the prior pipeline)."""
import numpy as np


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
