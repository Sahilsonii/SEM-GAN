"""
Evidential (Dirichlet) classification head and loss.

Carried over verbatim from the previous pipeline - these two were correct.
The global-average-pool classifier they were attached to was not, and is gone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning Head using Softplus activation to predict Dirichlet parameters alpha_k >= 1.0.
    Outputs: class probabilities and uncertainty metric u = K / S.
    """
    def __init__(self, in_features, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [B, in_features]
        evidence = F.softplus(self.fc(x)) # Non-negative evidence e_k >= 0
        alpha = evidence + 1.0 # Dirichlet parameters alpha_k >= 1.0
        
        S = torch.sum(alpha, dim=-1, keepdim=True) # Total Dirichlet strength
        probs = alpha / S # Expected class probability
        uncertainty = self.num_classes / S # Epistemic uncertainty u in (0, 1]
        
        return {
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty
        }


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
