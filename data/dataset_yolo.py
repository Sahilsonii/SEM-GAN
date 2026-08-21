import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

CLASS_NAMES = ["PbI2", "3D_pinholes", "3D-2D_pinholes", "3D_background", "3D-2D_background"]
CLASS_FOLDERS = ["class0_pbI2", "class1_3D_pinholes", "class2_3D-2D_pinholes", "class3_3D_background", "class4_3D-2D_background"]

class PerovskiteYOLODataset(Dataset):
    """
    Dataset loader for Perovskite FESEM images with YOLO-format annotations.
    Supports raw image loading, bounding box parsing, and clean vs defect separation.
    """
    def __init__(self, root_dir=r"C:\Users\Sahil\Downloads\SEM-Annotation\balanced_dataset", split="all", img_size=(768, 1024), transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
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
                    "boxes": boxes, # list of [class_id, cx, cy, w, h] normalized
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
        img_np = np.array(img, dtype=np.float32) / 255.0 # [H, W, 3] in [0, 1]
        
        # Convert to tensor [3, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        
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
    print(f"Perovskite Dataset Loaded: {len(ds)} total images")
    print(f"  - Clean Background Canvases: {len(ds.get_clean_canvases())} images")
    print(f"  - Defective Annotated Images: {len(ds.get_defect_samples())} images")
    sample = ds[0]
    print(f"Sample 0: {sample['filename']} | Class: {sample['class_name']} | Image Tensor: {sample['image'].shape} | Boxes: {len(sample['boxes'])}")
