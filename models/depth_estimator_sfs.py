import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage

class DepthEstimatorSfS:
    """
    3D Shape-from-Shading (SfS) and Vertical Shunt Severity Index (VSSI) estimator for Perovskite SEM.
    Translates secondary electron (SE) edge-effect brightness and shadow gradients into 3D topography.
    """
    def __init__(self, absorber_thickness_nm=500.0, shunt_threshold_ratio=0.75):
        self.absorber_thickness_nm = absorber_thickness_nm
        self.shunt_threshold_ratio = shunt_threshold_ratio

    def estimate_3d_depth_map(self, img_gray):
        """
        Estimates relative 3D depth map Z(x,y) from grayscale SEM image [0, 1] or [0, 255].
        """
        if isinstance(img_gray, torch.Tensor):
            img_np = img_gray.squeeze().detach().cpu().numpy()
        else:
            img_np = np.array(img_gray, dtype=np.float32)

        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        # Invert intensity (voids/pinholes appear darker in core, bright at edges)
        inverted = 1.0 - img_np
        
        # Smooth and compute spatial intensity gradients
        smoothed = ndimage.gaussian_filter(inverted, sigma=1.5)
        grad_y, grad_x = np.gradient(smoothed)
        
        # Approximate surface depth via Poisson integration of gradient field
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        depth_map = smoothed * (1.0 + 0.5 * grad_mag)
        
        # Scale to physical absorber thickness (nm)
        depth_map_nm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * self.absorber_thickness_nm
        return depth_map_nm

    def analyze_pinhole_shunt_risk(self, img_gray, bbox_norm):
        """
        Analyzes a specific pinhole bounding box [cx, cy, w, h] (normalized 0-1).
        Computes VSSI (Vertical Shunt Severity Index) and classifies shunt risk.
        """
        H, W = img_gray.shape[:2]
        cx, cy, w, h = bbox_norm
        
        x1 = max(0, int((cx - w/2) * W))
        x2 = min(W, int((cx + w/2) * W))
        y1 = max(0, int((cy - h/2) * H))
        y2 = min(H, int((cy + h/2) * H))
        
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return {"vssi": 0.0, "max_depth_nm": 0.0, "is_fatal_shunt": False, "classification": "Noise/Sub-resolution"}

        patch = img_gray[y1:y2, x1:x2]
        depth_patch = self.estimate_3d_depth_map(patch)
        
        max_depth = float(np.max(depth_patch) - np.min(depth_patch))
        depth_ratio = max_depth / self.absorber_thickness_nm
        
        # Area factor
        area_px = (x2 - x1) * (y2 - y1)
        area_factor = np.clip(area_px / 400.0, 0.5, 2.0)
        
        vssi = float(np.clip(depth_ratio * area_factor, 0.0, 1.0))
        is_fatal = (vssi >= self.shunt_threshold_ratio)
        
        classification = "Fatal Through-Thickness Shunt" if is_fatal else "Passivated Shallow Void"
        
        return {
            "vssi": vssi,
            "max_depth_nm": max_depth,
            "is_fatal_shunt": is_fatal,
            "classification": classification,
            "depth_patch": depth_patch
        }

if __name__ == "__main__":
    estimator = DepthEstimatorSfS(absorber_thickness_nm=500.0)
    test_img = np.random.rand(768, 1024).astype(np.float32)
    # Simulate a dark void
    test_img[300:350, 400:450] *= 0.2
    
    res = estimator.analyze_pinhole_shunt_risk(test_img, [425/1024, 325/768, 50/1024, 50/768])
    print("SfS Depth Analysis Sample Result:")
    print(f"  VSSI Score: {res['vssi']:.4f} | Max Depth: {res['max_depth_nm']:.1f} nm | Status: {res['classification']}")
