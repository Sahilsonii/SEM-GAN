import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPatchDiscriminator(nn.Module):
    """
    Multi-Scale PatchGAN Discriminator operating in the spatial image domain.
    """
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: [B, 1, H, W] -> [B, ndf, H/2, W/2]
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: -> [B, ndf*2, H/4, W/4]
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: -> [B, ndf*4, H/8, W/8]
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: -> [B, ndf*8, H/8, W/8]
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: Output 1-channel Patch prediction map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.net(x)

class FourierFrequencyDiscriminator(nn.Module):
    """
    Frequency-Domain Discriminator that evaluates the 2D Fourier Power Spectrum of images.
    Prevents hallucinated non-physical periodic grid artifacts in SEM super-resolution.
    """
    def __init__(self, in_channels=1, ndf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x):
        # Compute 2D FFT magnitude spectrum
        # x: [B, 1, H, W]
        fft_img = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft_img, dim=(-2, -1))
        magnitude_spectrum = torch.log(torch.abs(fft_shift) + 1e-8)
        
        # Pass power spectrum through frequency discriminator
        return self.net(magnitude_spectrum)

class DualDomainDiscriminator(nn.Module):
    """
    Combines Spatial PatchGAN Discriminator and Fourier Frequency Discriminator.
    """
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        self.spatial_disc = SpatialPatchDiscriminator(in_channels, ndf)
        self.fourier_disc = FourierFrequencyDiscriminator(in_channels, ndf // 2)
        
    def forward(self, x):
        out_spatial = self.spatial_disc(x)
        out_fourier = self.fourier_disc(x)
        return out_spatial, out_fourier

if __name__ == "__main__":
    netD = DualDomainDiscriminator()
    x = torch.randn(2, 1, 256, 256)
    out_s, out_f = netD(x)
    print(f"Discriminator Spatial Output: {out_s.shape} | Fourier Output: {out_f.shape}")
