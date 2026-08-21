import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsFourierLoss(nn.Module):
    """
    2D Fast Fourier Transform Spectral Loss for physical electron scattering conservation.
    Ensures generated microstructures exhibit authentic spatial frequency distribution.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: [B, C, H, W] in [-1, 1] or [0, 1]
        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))
        
        mag_pred = torch.log(torch.abs(fft_pred) + self.eps)
        mag_target = torch.log(torch.abs(fft_target) + self.eps)
        
        loss = F.l1_loss(mag_pred, mag_target)
        return loss

class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning (EDL) Loss based on Dirichlet distribution parameters.
    Quantifies epistemic uncertainty: u = K / S, where S = sum(alpha_k).
    """
    def __init__(self, num_classes=5, lambda_kl=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl

    def forward(self, alpha, target_one_hot, epoch=1):
        # alpha: [B, K] where alpha_k >= 1.0
        # target_one_hot: [B, K]
        S = torch.sum(alpha, dim=1, keepdim=True) # [B, 1]
        
        # 1. Expected Cross Entropy Loss under Dirichlet distribution
        log_S = torch.log(S)
        log_alpha = torch.log(alpha)
        l_ace = torch.sum(target_one_hot * (log_S - log_alpha), dim=1, keepdim=True)
        
        # 2. KL Divergence Regularization to flat Dirichlet uniform prior
        alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        first_term = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float32, device=alpha.device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        )
        second_term = torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=1,
            keepdim=True
        )
        l_kl = first_term + second_term
        
        # Annealing coefficient for KL term
        annealing_coef = min(1.0, epoch / 10.0)
        total_loss = torch.mean(l_ace + annealing_coef * self.lambda_kl * l_kl)
        
        # Calculate uncertainty metric u = K / S
        uncertainty = self.num_classes / S
        
        return total_loss, torch.mean(uncertainty).item()
