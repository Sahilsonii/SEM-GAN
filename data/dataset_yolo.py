import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

CLASS_NAMES = ["PbI2", "3D_pinholes", "3D-2D_pinholes", "3D_background", "3D-2D_background"]
CLASS_FOLDERS = ["class0_pbI2", "class1_3D_pinholes", "class2_3D-2D_pinholes", "class3_3D_background", "class4_3D-2D_background"]

STANDARD_SIZE = (512, 512) # (Width, Height) optimized for RTX 3050 Ti VRAM

def yolo_collate_fn(batch):
    """
    Custom collate function for batches with variable-sized bounding boxes and images.
    Ensures all images in batch are uniformly sized [3, 512, 512].
    """
    processed_images = []
    for item in batch:
        img = item["image"] # [3, H, W]
        if img.shape[1] != STANDARD_SIZE[1] or img.shape[2] != STANDARD_SIZE[0]:
            img = F.interpolate(img.unsqueeze(0), size=(STANDARD_SIZE[1], STANDARD_SIZE[0]), mode='bilinear', align_corners=False).squeeze(0)
        processed_images.append(img)
        
    images = torch.stack(processed_images, dim=0) # [B, 3, 512, 512]
    class_indices = torch.tensor([item["class_idx"] for item in batch], dtype=torch.long)
    filenames = [item["filename"] for item in batch]
    class_names = [item["class_name"] for item in batch]
    is_clean = [item["is_clean"] for item in batch]
    boxes = [item["boxes"] for item in batch]

    return {
        "image": images,
        "class_idx": class_indices,
        "filename": filenames,
        "class_name": class_names,
        "is_clean": is_clean,
        "boxes": boxes
    }

class PerovskiteYOLODataset(Dataset):
    """
    Dataset loader for Perovskite FESEM images with YOLO-format annotations.
    Automatically standardizes image resolution to (512x512) to fit 4GB GPU VRAM.
    """
    def __init__(self, root_dir=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset", split="all", target_size=(512, 512)):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size # (W, H)
        
        self.images_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        
        self.samples = []
        self.clean_backgrounds = []
        self.defect_samples = []
        
        self._load_dataset()

    def _load_dataset(self):
        split_files = None
        if self.split in ["train", "val", "test"]:
            split_txt = os.path.join(self.root_dir, f"{self.split}.txt")
            if os.path.exists(split_txt):
                with open(split_txt, "r") as f:
                    split_files = set([os.path.basename(line.strip()) for line in f if line.strip()])

        for cls_idx, folder in enumerate(CLASS_FOLDERS):
            c_img_dir = os.path.join(self.images_dir, folder)
            c_lbl_dir = os.path.join(self.labels_dir, folder)
            
            if not os.path.exists(c_img_dir):
                continue
                
            img_paths = glob.glob(os.path.join(c_img_dir, "*.*"))
            for img_path in img_paths:
                fname = os.path.basename(img_path)
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    continue
                    
                if split_files is not None and fname not in split_files:
                    continue
                    
                base_name = os.path.splitext(fname)[0]
                lbl_path = os.path.join(c_lbl_dir, f"{base_name}.txt")
                
                boxes = []
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:5])
                                boxes.append([cid, cx, cy, w, h])
                                
                item = {
                    "img_path": img_path,
                    "filename": fname,
                    "class_idx": cls_idx,
                    "class_name": CLASS_NAMES[cls_idx],
                    "boxes": boxes,
                    "is_clean": (cls_idx in [3, 4] or len(boxes) == 0)
                }
                
                self.samples.append(item)
                if item["is_clean"]:
                    self.clean_backgrounds.append(item)
                else:
                    self.defect_samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        
        # Standardize size to 512x512
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.BILINEAR)
            
        img_np = np.array(img, dtype=np.float32) / 255.0 # [H, W, 3] in [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # [3, H, W]
        
        return {
            "image": img_tensor,
            "filename": item["filename"],
            "class_idx": item["class_idx"],
            "class_name": item["class_name"],
            "boxes": torch.tensor(item["boxes"], dtype=torch.float32) if len(item["boxes"]) > 0 else torch.zeros((0, 5)),
            "is_clean": item["is_clean"]
        }

    def get_clean_canvases(self):
        """Returns all pristine background images (Class 3 & Class 4) for generative inpainting."""
        return self.clean_backgrounds

    def get_defect_samples(self):
        """Returns all defect images with bounding box annotations."""
        return self.defect_samples

if __name__ == "__main__":
    ds = PerovskiteYOLODataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=yolo_collate_fn)
    b = next(iter(loader))
    print(f"Batch loaded: image tensor shape={b['image'].shape}, labels={b['class_idx']}")
