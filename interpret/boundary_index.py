import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
import cv2

class GBDIAnalyzer:
    """
    Topological Grain Boundary Decoration Index (GBDI) Analyzer for PbI2 phase distribution.
    Differentiates beneficial grain boundary passivation from detrimental recombination clusters.
    """
    def __init__(self, sigma_passivation_px=15.0, passivation_threshold=0.65):
        self.sigma_passivation_px = sigma_passivation_px
        self.passivation_threshold = passivation_threshold

    def extract_grain_boundaries(self, img_gray):
        """
        Extracts grain boundary binary mask from FESEM image using morphological edge filtering.
        """
        if img_gray.max() <= 1.0:
            img_u8 = (img_gray * 255).astype(np.uint8)
        else:
            img_u8 = img_gray.astype(np.uint8)
            
        # Bilateral filter to preserve sharp grain boundaries while smoothing noise
        filtered = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
        # Adaptive thresholding to identify grain boundary networks
        edges = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        
        # Morphological skeletonization/thinning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gb_mask = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        return gb_mask > 0

    def compute_gbdi(self, img_gray, pbi2_boxes_norm):
        """
        Calculates GBDI score for a list of PbI2 bounding boxes [cid, cx, cy, w, h] (normalized).
        Returns GBDI score in [0, 1], distance distributions, and passivation verdict.
        """
        H, W = img_gray.shape[:2]
        gb_mask = self.extract_grain_boundaries(img_gray)
        
        # Coordinates of grain boundary pixels
        gb_points = np.argwhere(gb_mask) # [N, 2] as (y, x)
        
        if len(pbi2_boxes_norm) == 0:
            return {
                "gbdi_score": 1.0,
                "pbi2_count": 0,
                "verdict": "Pristine Stoichiometry (No Residual PbI2)",
                "avg_distance_px": 0.0,
                "is_beneficial": True
            }
            
        if len(gb_points) == 0:
            # Fallback if no distinct GB detected
            return {
                "gbdi_score": 0.5,
                "pbi2_count": len(pbi2_boxes_norm),
                "verdict": "Unresolved Grain Boundaries",
                "avg_distance_px": 999.0,
                "is_beneficial": False
            }

        # Build KDTree for fast spatial nearest-neighbor search
        tree = cKDTree(gb_points)
        
        distances = []
        for box in pbi2_boxes_norm:
            cx, cy = box[1], box[2]
            px_x = cx * W
            px_y = cy * H
            # Query nearest grain boundary point
            dist, _ = tree.query([px_y, px_x])
            distances.append(dist)
            
        distances = np.array(distances)
        
        # Exponential Gaussian decay kernel
        weights = np.exp(- (distances ** 2) / (2 * (self.sigma_passivation_px ** 2)))
        gbdi_score = float(np.mean(weights))
        
        is_beneficial = (gbdi_score >= self.passivation_threshold)
        verdict = "Beneficial Grain Boundary Passivation (Elevates Voc)" if is_beneficial else "Detrimental Intragranular Recombination Clusters"
        
        return {
            "gbdi_score": gbdi_score,
            "pbi2_count": len(pbi2_boxes_norm),
            "verdict": verdict,
            "avg_distance_px": float(np.mean(distances)),
            "is_beneficial": is_beneficial,
            "distances": distances.tolist()
        }

if __name__ == "__main__":
    analyzer = GBDIAnalyzer()
    dummy_img = np.random.rand(768, 1024).astype(np.float32)
    # 5 dummy PbI2 detections
    dummy_boxes = [
        [0, 0.25, 0.30, 0.05, 0.05],
        [0, 0.45, 0.60, 0.04, 0.04],
        [0, 0.70, 0.80, 0.06, 0.06]
    ]
    res = analyzer.compute_gbdi(dummy_img, dummy_boxes)
    print("GBDI Analysis Sample Result:")
    print(f"  GBDI Score: {res['gbdi_score']:.4f} | Avg Dist: {res['avg_distance_px']:.1f} px | Verdict: {res['verdict']}")
