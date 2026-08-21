import os
import glob
import re
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

def parse_sem_metadata(filepath):
    """
    Parses embedded FESEM TIF metadata to extract magnification and physical pixel scale.
    """
    with open(filepath, 'rb') as f:
        content = f.read()
    
    m_pix = re.search(rb'Image Pixel Size = ([0-9\.]+) (nm|um|pm)', content)
    pix_size = float(m_pix.group(1).decode()) if m_pix else 22.33
    unit = m_pix.group(2).decode() if m_pix else "nm"
    
    m_mag = re.search(rb'Mag = +([0-9\.]+) ([KX]+)', content)
    mag_val = float(m_mag.group(1).decode()) if m_mag else 5.0
    
    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "pixel_size_nm": pix_size,
        "magnification_k": mag_val
    }

class SEMPatchDataset(Dataset):
    """
    PyTorch Dataset that ingests 1024x768 SEM TIF images, parses metadata,
    and extracts multi-scale sub-patches for super-resolution training.
    """
    def __init__(self, data_dir, patch_size=256, stride=128, scale_factor=2, is_train=True):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = scale_factor
        self.is_train = is_train
        
        filepaths = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.TIF"))
        if len(filepaths) == 0:
            raise ValueError(f"No .tif images found in directory: {data_dir}")
        
        self.images_meta = [parse_sem_metadata(fp) for fp in filepaths]
        self.patches = []
        self._crop_and_store_patches()
        
    def _crop_and_store_patches(self):
        for meta in self.images_meta:
            fp = meta["filepath"]
            img = Image.open(fp).convert('L')
            img_arr = np.array(img, dtype=np.float32) / 255.0 # Normalize to [0, 1]
            
            H, W = img_arr.shape
            # Extract sliding window patches
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    hr_patch = img_arr[y:y+self.patch_size, x:x+self.patch_size]
                    self.patches.append({
                        "hr_patch": hr_patch,
                        "meta": meta
                    })
                    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        item = self.patches[idx]
        hr_patch = item["hr_patch"] # [H, W] float32 in [0, 1]
        
        # Apply data augmentations if training
        if self.is_train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                hr_patch = np.fliplr(hr_patch).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                hr_patch = np.flipud(hr_patch).copy()
            # Random 90-degree rotations
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                hr_patch = np.rot90(hr_patch, k).copy()
                
        # Convert HR to Tensor [1, H, W] in range [-1, 1]
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0) * 2.0 - 1.0
        
        # Downsample HR to create Low-Res (LR) pair using bicubic interpolation
        lr_size = self.patch_size // self.scale_factor
        lr_tensor = torch.nn.functional.interpolate(
            hr_tensor.unsqueeze(0),
            size=(lr_size, lr_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        return {
            "lr": lr_tensor,      # [1, LR_H, LR_W] in [-1, 1]
            "hr": hr_tensor,      # [1, HR_H, HR_W] in [-1, 1]
            "mag_k": item["meta"]["magnification_k"],
            "filename": item["meta"]["filename"]
        }

if __name__ == "__main__":
    sample_dir = r"C:\Users\Sahil\Downloads\SEM\3D pin hole"
    dataset = SEMPatchDataset(sample_dir, patch_size=256, stride=128, scale_factor=2)
    print(f"Dataset successfully created with {len(dataset)} sub-patches!")
    sample = dataset[0]
    print("LR Tensor shape:", sample["lr"].shape, "| HR Tensor shape:", sample["hr"].shape)
