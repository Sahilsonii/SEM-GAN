"""Adversarial loss for the defect-texture refiner (kept from the prior pipeline)."""
import torch
import torch.nn as nn


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
