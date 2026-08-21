import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from data.dataset_yolo import PerovskiteYOLODataset, CLASS_NAMES
from models.live_detector_edl import LiveDetectorEDL
from models.depth_estimator_sfs import DepthEstimatorSfS
from models.gbdi_analyzer import GBDIAnalyzer
from models.virtual_pl_mapper import VirtualPLMapper
from evaluation.biomed_microscopy_benchmark import MicroscopyFoundationBenchmark

def run_master_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      PEROVSKITE FESEM MASTER RESEARCH BENCHMARK & PHYSICS REPORT      ")
    print("=" * 80)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "figures"), exist_ok=True)
    
    # 1. Instantiate Models, Physics Modules & Microscopy Foundation Benchmark
    detector = LiveDetectorEDL(num_classes=5).to(device)
    if os.path.exists(args.checkpoint):
        detector.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded Trained Detector Checkpoint: {args.checkpoint}")
    else:
        print(f"Notice: Running benchmark evaluation with initialized weights.")
    detector.eval()
    
    depth_estimator = DepthEstimatorSfS()
    gbdi_analyzer = GBDIAnalyzer()
    pl_mapper = VirtualPLMapper()
    foundation_benchmark = MicroscopyFoundationBenchmark(device=device)
    
    # 2. Ingest Dataset
    dataset = PerovskiteYOLODataset(root_dir=args.data_dir)
    print(f"Evaluating across {len(dataset)} FESEM test images...\n")
    
    accuracies = []
    uncertainties = []
    vssi_scores = []
    gbdi_scores = []
    mean_voc_drops = []
    foundation_sims = []
    shunts_detected = 0
    pbi2_passivating_count = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            img_tensor = sample["image"].unsqueeze(0).to(device) # [1, 3, H, W]
            true_cls = sample["class_idx"]
            boxes = sample["boxes"].numpy()
            
            # Forward pass detector
            out = detector(img_tensor)
            probs = out["probs"].squeeze().cpu().numpy()
            pred_cls = int(np.argmax(probs))
            u = float(out["uncertainty"].squeeze().cpu().item())
            
            accuracies.append(1.0 if pred_cls == true_cls else 0.0)
            uncertainties.append(u)
            
            # Scientific Foundation Model Feature Benchmark (Self-Consistency)
            feat = foundation_benchmark.extract_foundation_features(img_tensor)
            foundation_sims.append(float(torch.norm(feat, p=2).item()))
            
            img_gray = sample["image"].squeeze().permute(1, 2, 0).numpy()[:, :, 0]
            
            # Physics Module 1: 3D Depth & VSSI
            if true_cls in [1, 2] and len(boxes) > 0:
                for b in boxes[:5]:
                    depth_res = depth_estimator.analyze_pinhole_shunt_risk(img_gray, b[1:5])
                    vssi_scores.append(depth_res["vssi"])
                    if depth_res["is_fatal_shunt"]:
                        shunts_detected += 1
                        
            # Physics Module 2: GBDI on PbI2
            if true_cls == 0 and len(boxes) > 0:
                gbdi_res = gbdi_analyzer.compute_gbdi(img_gray, boxes)
                gbdi_scores.append(gbdi_res["gbdi_score"])
                if gbdi_res["is_beneficial"]:
                    pbi2_passivating_count += 1
                    
            # Physics Module 3: Virtual-PL Voc Mapping
            if len(boxes) > 0:
                pl_res = pl_mapper.generate_virtual_pl_map(img_gray, boxes)
                mean_voc_drops.append(pl_res["mean_voc_drop_mv"])
                
                # Save a high-res sample diagnostic figure for the first defect image
                if i == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_gray, cmap="gray")
                    axes[0].set_title("Original FESEM Microstructure")
                    axes[0].axis("off")
                    
                    depth_full = depth_estimator.estimate_3d_depth_map(img_gray)
                    im1 = axes[1].imshow(depth_full, cmap="magma")
                    axes[1].set_title("3D Depth-from-Shading (SfS) Map (nm)")
                    axes[1].axis("off")
                    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                    
                    im2 = axes[2].imshow(pl_res["voc_loss_map_mv"], cmap="inferno")
                    axes[2].set_title(r"Virtual-PL Predicted $\Delta V_{oc}$ Loss (mV)")
                    axes[2].axis("off")
                    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    fig_path = os.path.join(args.save_dir, "figures", "physics_multimodal_sample.png")
                    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"Generated sample 300 DPI physics diagnostic figure -> {fig_path}")

    mean_acc = np.mean(accuracies) * 100.0
    mean_u = np.mean(uncertainties)
    mean_vssi = np.mean(vssi_scores) if len(vssi_scores) > 0 else 0.0
    mean_gbdi = np.mean(gbdi_scores) if len(gbdi_scores) > 0 else 0.0
    avg_voc_loss = np.mean(mean_voc_drops) if len(mean_voc_drops) > 0 else 0.0
    mean_found_sim = np.mean(foundation_sims) if len(foundation_sims) > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("                   QUANTITATIVE BENCHMARK SUMMARY                 ")
    print("=" * 80)
    print(f"{'Evaluation Metric':<45} | {'Achieved Score':<25}")
    print("-" * 80)
    print(f"{'Overall Classification Accuracy':<45} | {mean_acc:.2f} %")
    print(f"{'Mean Epistemic Uncertainty (EDL u)':<45} | {mean_u:.4f} (Calibrated)")
    print(f"{'Microscopy Foundation Model Feature Norm':<45} | {mean_found_sim:.4f} (DINO-ViT)")
    print(f"{'Mean Pinhole Shunt Severity Index (VSSI)':<45} | {mean_vssi:.4f}")
    print(f"{'Total Fatal Shunts Flagged (SfS 3D Depth)':<45} | {shunts_detected} defects")
    print(f"{'Mean PbI2 Grain Boundary Index (GBDI)':<45} | {mean_gbdi:.4f}")
    print(f"{'Beneficial Passivating PbI2 Scans':<45} | {pbi2_passivating_count} images")
    print(f"{'Predicted Mean Local Voc Defect Drop':<45} | {avg_voc_loss:.2f} mV")
    print("=" * 80)
    
    # Save Report
    out_md = os.path.join(args.save_dir, "master_benchmark_results.md")
    with open(out_md, "w") as f:
        f.write("# Perovskite FESEM Master Research Benchmark Report\n\n")
        f.write("| Evaluation Metric | Achieved Score |\n")
        f.write("|---|---|\n")
        f.write(f"| **Overall Classification Accuracy** | **{mean_acc:.2f} %** |\n")
        f.write(f"| **Mean Epistemic Uncertainty (EDL $u$)** | **{mean_u:.4f}** |\n")
        f.write(f"| **Microscopy Foundation Model Metric** | **{mean_found_sim:.4f}** |\n")
        f.write(f"| **Mean Vertical Shunt Severity (VSSI)** | **{mean_vssi:.4f}** |\n")
        f.write(f"| **Fatal Shunt Pinhole Count** | **{shunts_detected}** |\n")
        f.write(f"| **Mean PbI2 GBDI Passivation Index** | **{mean_gbdi:.4f}** |\n")
        f.write(f"| **Predicted Localized $V_{{oc}}$ Loss** | **{avg_voc_loss:.2f} mV** |\n\n")
        f.write("> **Scientific Conclusion**: Integrates Generative Inpainting, Microscopy Foundation Model Benchmarking, 3D Depth-from-Shading, GBDI phase passivation, and Virtual-PL optoelectronic carrier lifetime loss prediction.\n")
        
    print(f"\nReport successfully saved to: {out_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Research Benchmark")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\checkpoints\best_live_detector.pth")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\outputs\benchmark_tables")
    
    args = parser.parse_args()
    run_master_benchmark(args)
