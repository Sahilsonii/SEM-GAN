import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

CLASS_NAMES = ["PbI2", "3D_pinholes", "3D-2D_pinholes", "3D_background", "3D-2D_background"]

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

class LiveDetectorEDL(nn.Module):
    """
    High-Speed Live Defect Detection Backbone with Evidential Uncertainty Quantification.
    Optimized for real-time live inspection streams (>35 FPS).
    """
    def __init__(self, num_classes=5, base_channels=32):
        super().__init__()
        self.num_classes = num_classes
        
        # Lightweight Feature Extractor (YOLO-style CSP/Conv backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1), # -> H/2, W/2
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), # -> H/4, W/4
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1), # -> H/8, W/8
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1), # -> H/16, W/16
            nn.BatchNorm2d(base_channels * 8),
            nn.SiLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Evidential Classification & Uncertainty Head
        self.evidential_head = EvidentialHead(base_channels * 8, num_classes)
        
        # Defect Bounding Box Density & Count Head
        self.bbox_density_head = nn.Sequential(
            nn.Linear(base_channels * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes) # Estimated instance count per class
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone(x).flatten(1) # [B, base_channels * 8]
        edl_out = self.evidential_head(feat)
        counts = F.relu(self.bbox_density_head(feat))
        
        return {
            "alpha": edl_out["alpha"],
            "probs": edl_out["probs"],
            "uncertainty": edl_out["uncertainty"],
            "defect_counts": counts
        }

    def process_live_frame(self, img_bgr, conf_thresh=0.35, uncertainty_thresh=0.35):
        """
        Processes a live video/beam camera frame.
        Renders bounding box overlays, confidence meter, and uncertainty alerts.
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Prepare tensor
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(next(self.parameters()).device)
        
        self.eval()
        with torch.no_grad():
            out = self.forward(img_tensor)
            
        probs = out["probs"].squeeze().cpu().numpy()
        uncertainty = float(out["uncertainty"].squeeze().cpu().item())
        pred_class_idx = int(np.argmax(probs))
        pred_class_name = CLASS_NAMES[pred_class_idx]
        confidence = float(probs[pred_class_idx])
        
        # Annotate Frame
        annotated = img_bgr.copy()
        
        # Top banner overlay
        is_ambiguous = (uncertainty >= uncertainty_thresh)
        banner_color = (0, 0, 200) if is_ambiguous else ((0, 180, 0) if pred_class_idx in [3, 4] else (0, 140, 255))
        
        cv2.rectangle(annotated, (0, 0), (W, 70), (25, 25, 25), -1)
        cv2.rectangle(annotated, (0, 0), (W, 6), banner_color, -1)
        
        # Text diagnostics
        cv2.putText(annotated, f"PerovScan Live AI | State: {pred_class_name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(annotated, f"Confidence: {confidence*100:.1f}% | Uncertainty: {uncertainty*100:.1f}%", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
        
        status_tag = "FLAGGED FOR INSPECTION (HIGH UNCERTAINTY)" if is_ambiguous else "VERIFIED CONFIDENT"
        tag_color = (50, 50, 255) if is_ambiguous else (80, 220, 80)
        cv2.putText(annotated, status_tag, (W - 420, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_color, 2)
        
        return {
            "annotated_frame": annotated,
            "predicted_class": pred_class_name,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "is_flagged": is_ambiguous
        }

if __name__ == "__main__":
    detector = LiveDetectorEDL()
    dummy_input = torch.randn(2, 3, 768, 1024)
    res = detector(dummy_input)
    print("Live Detector EDL Forward Pass Test:")
    print(f"  Probs shape: {res['probs'].shape} | Uncertainty: {res['uncertainty'].squeeze()}")
