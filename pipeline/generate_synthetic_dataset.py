import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.dataset_yolo import PerovskiteYOLODataset
from models.inpainter_controlnet import DefectInpainterUNet, generate_random_defect_layout

def generate_expanded_dataset(
    raw_dataset_dir=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset",
    output_dir=r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\data\expanded_dataset",
    num_synthetic_samples=1000,
    device="cpu"
):
    print("=" * 70)
    print("       GENERATIVE DEFECT INPAINTING: DATASET EXPANSION PIPELINE       ")
    print("=" * 70)
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # 1. Load Real Dataset & Clean Canvases
    ds = PerovskiteYOLODataset(root_dir=raw_dataset_dir)
    clean_canvases = ds.get_clean_canvases()
    print(f"Loaded {len(ds)} real images ({len(clean_canvases)} pristine background canvases)")
    
    if len(clean_canvases) == 0:
        raise ValueError("No clean background images found for inpainting canvas generation.")

    # 2. Instantiate Generative Inpainter
    inpainter = DefectInpainterUNet().to(device)
    inpainter.eval()
    
    print(f"\nSynthesizing {num_synthetic_samples} novel defect images via generative inpainting...")
    
    synthetic_count = 0
    total_generated_boxes = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_synthetic_samples), desc="Generating Defect Inpaintings"):
            # Sample a clean background canvas
            bg_item = clean_canvases[i % len(clean_canvases)]
            bg_pil = Image.open(bg_item["img_path"]).convert("RGB")
            bg_np = np.array(bg_pil, dtype=np.float32) / 255.0 # [H, W, 3]
            
            H, W = bg_np.shape[:2]
            
            # Generate random defect layout & YOLO bounding boxes
            mask_np, boxes = generate_random_defect_layout(H=H, W=W, max_defects=12)
            
            # Prepare tensors
            bg_tensor = torch.from_numpy(bg_np).permute(2, 0, 1).unsqueeze(0).to(device) # [1, 3, H, W]
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, H, W]
            
            # Synthesize defect texture onto clean canvas
            syn_tensor = inpainter(bg_tensor, mask_tensor)
            syn_np = (syn_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # Save Synthetic Image
            syn_fname = f"syn_defect_{i+1:05d}.jpg"
            syn_img_path = os.path.join(output_dir, "images", syn_fname)
            Image.fromarray(syn_np).save(syn_img_path, quality=95)
            
            # Save YOLO Annotation File
            syn_lbl_fname = f"syn_defect_{i+1:05d}.txt"
            syn_lbl_path = os.path.join(output_dir, "labels", syn_lbl_fname)
            with open(syn_lbl_path, "w") as lf:
                for b in boxes:
                    # [cid, cx, cy, w, h]
                    lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    
            synthetic_count += 1
            total_generated_boxes += len(boxes)
            
    print("\n" + "=" * 70)
    print("                DATASET EXPANSION COMPLETE!                ")
    print("=" * 70)
    print(f"Total Synthetic Images Generated: {synthetic_count}")
    print(f"Total Synthetic Bounding Boxes:   {total_generated_boxes}")
    print(f"Output Directory:                 {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    generate_expanded_dataset(num_synthetic_samples=50) # Fast test run
