import numpy as np
import torch

class VirtualPLMapper:
    """
    In Silico Virtual Photoluminescence (PL) & Non-Radiative Recombination Loss Mapper.
    Translates localized FESEM microstructural defect topography (pinholes, PbI2, grain boundaries)
    into predicted carrier recombination heatmaps and spatial Voc loss maps.
    """
    def __init__(self, baseline_pl_intensity=100.0, kb_t_q=0.0259):
        # Thermal voltage at 300K: kT/q ≈ 25.9 mV
        self.baseline_pl_intensity = baseline_pl_intensity
        self.vt = kb_t_q

    def generate_virtual_pl_map(self, img_gray, detected_boxes_norm):
        """
        Computes a 2D spatial Virtual-PL carrier intensity map [H, W] and predicted Voc loss heatmap.
        """
        H, W = img_gray.shape[:2]
        
        # Base emission intensity map (normalized)
        if img_gray.max() > 1.0:
            norm_img = img_gray.astype(np.float32) / 255.0
        else:
            norm_img = img_gray.astype(np.float32)

        # Baseline uniform photoluminescence
        pl_map = np.ones((H, W), dtype=np.float32) * self.baseline_pl_intensity
        
        # Microstructure-induced optical quenching
        pl_map *= (0.8 + 0.2 * norm_img)
        
        # Local non-radiative recombination quenching at defect sites
        recombination_quench_map = np.zeros((H, W), dtype=np.float32)
        
        for b in detected_boxes_norm:
            cid, cx, cy, w, h = b[:5]
            x1 = max(0, int((cx - w/2) * W))
            x2 = min(W, int((cx + w/2) * W))
            y1 = max(0, int((cy - h/2) * H))
            y2 = min(H, int((cy + h/2) * H))
            
            # Quenching strength based on defect physics:
            # - Pinholes (cid 1 & 2): 80-95% optical quenching (severe non-radiative shunt)
            # - PbI2 clusters (cid 0): 40-60% quenching (moderate trap recombination)
            quench_factor = 0.90 if cid in [1, 2] else 0.50
            
            # Gaussian spatial profile for defect quenching diffusion radius
            dy, dx = np.ogrid[y1-cy*H:y2-cy*H, x1-cx*W:x2-cx*W]
            sigma_eff = max(w*W, h*H) / 2.0
            gaussian_kernel = np.exp(-(dx**2 + dy**2) / (2 * (sigma_eff**2) + 1e-8))
            
            recombination_quench_map[y1:y2, x1:x2] = np.maximum(
                recombination_quench_map[y1:y2, x1:x2],
                quench_factor * gaussian_kernel
            )

        # Resulting Virtual PL Intensity Map
        virtual_pl = pl_map * (1.0 - recombination_quench_map)
        
        # Optoelectronic Voc Loss Map (ΔVoc = - (kT/q) * ln(PL / PL_baseline))
        pl_ratio = np.clip(virtual_pl / (self.baseline_pl_intensity + 1e-8), 1e-4, 1.0)
        voc_loss_mv = - self.vt * np.log(pl_ratio) * 1000.0 # in milliVolts (mV)
        
        return {
            "virtual_pl_map": virtual_pl,
            "voc_loss_map_mv": voc_loss_mv,
            "mean_voc_drop_mv": float(np.mean(voc_loss_mv)),
            "max_localized_voc_drop_mv": float(np.max(voc_loss_mv))
        }

if __name__ == "__main__":
    mapper = VirtualPLMapper()
    dummy_img = np.random.rand(768, 1024).astype(np.float32)
    dummy_boxes = [
        [1, 0.3, 0.4, 0.05, 0.05], # 3D pinhole
        [0, 0.7, 0.6, 0.04, 0.04]  # PbI2 cluster
    ]
    res = mapper.generate_virtual_pl_map(dummy_img, dummy_boxes)
    print("Virtual-PL Mapping Sample Result:")
    print(f"  Mean Voc Loss: {res['mean_voc_drop_mv']:.2f} mV | Max Localized Drop: {res['max_localized_voc_drop_mv']:.2f} mV")
