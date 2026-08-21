import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 based Perceptual Feature Loss for deep texture evaluation.
    """
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features
            
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).eval()   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).eval()  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)]).eval() # relu3_3
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, sr, hr):
        # Repeat 1-channel grayscale to 3-channels for VGG input
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
            
        # Normalize to VGG mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=sr.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=sr.device).view(1, 3, 1, 1)
        sr_norm = (sr - mean) / std
        hr_norm = (hr - mean) / std
        
        f1_sr, f1_hr = self.slice1(sr_norm), self.slice1(hr_norm)
        f2_sr, f2_hr = self.slice2(f1_sr), self.slice2(f1_hr)
        f3_sr, f3_hr = self.slice3(f2_sr), self.slice3(f2_hr)
        
        loss = F.l1_loss(f1_sr, f1_hr) + F.l1_loss(f2_sr, f2_hr) + F.l1_loss(f3_sr, f3_hr)
        return loss

class FFTFrequencyLoss(nn.Module):
    """
    2D Fast Fourier Transform Spectral Loss to ensure frequency domain alignment.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, sr, hr):
        fft_sr = torch.fft.fft2(sr, dim=(-2, -1))
        fft_hr = torch.fft.fft2(hr, dim=(-2, -1))
        
        mag_sr = torch.abs(fft_sr)
        mag_hr = torch.abs(fft_hr)
        
        loss = F.l1_loss(mag_sr, mag_hr)
        return loss

class GANLoss(nn.Module):
    """
    Adversarial Hinge Loss for Discriminator and Generator training.
    """
    def __init__(self):
        super().__init__()
        
    def loss_discriminator(self, real_pred, fake_pred):
        loss_real = torch.mean(F.relu(1.0 - real_pred))
        loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        return loss_real + loss_fake
        
    def loss_generator(self, fake_pred):
        return -torch.mean(fake_pred)

class SEMCombinedLoss(nn.Module):
    """
    Combined Loss function for SEM Super-Resolution GAN training.
    """
    def __init__(self, lambda_pixel=1.0, lambda_perceptual=0.1, lambda_fft=0.05, lambda_adv=0.01):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_fft = lambda_fft
        self.lambda_adv = lambda_adv
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.fft_loss = FFTFrequencyLoss()
        self.gan_loss = GANLoss()
        
    def forward_g(self, sr, hr, spatial_fake, fourier_fake):
        l_pixel = self.l1_loss(sr, hr)
        l_perc = self.perceptual_loss(sr, hr)
        l_fft = self.fft_loss(sr, hr)
        l_adv = self.gan_loss.loss_generator(spatial_fake) + self.gan_loss.loss_generator(fourier_fake)
        
        total_loss = (
            self.lambda_pixel * l_pixel +
            self.lambda_perceptual * l_perc +
            self.lambda_fft * l_fft +
            self.lambda_adv * l_adv
        )
        
        return total_loss, {
            "loss_pixel": l_pixel.item(),
            "loss_perceptual": l_perc.item(),
            "loss_fft": l_fft.item(),
            "loss_adv": l_adv.item()
        }
        
    def forward_d(self, spatial_real, spatial_fake, fourier_real, fourier_fake):
        l_spatial = self.gan_loss.loss_discriminator(spatial_real, spatial_fake)
        l_fourier = self.gan_loss.loss_discriminator(fourier_real, fourier_fake)
        return l_spatial + l_fourier
