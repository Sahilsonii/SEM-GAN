import os
import shutil

src_root = r"C:\Users\Sahil\.gemini\antigravity-ide\scratch\sem_gan_project"
dst_root = r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation"

files_to_sync = [
    ("data/dataset_yolo.py", "data/dataset_yolo.py"),
    ("models/depth_estimator_sfs.py", "models/depth_estimator_sfs.py"),
    ("models/gbdi_analyzer.py", "models/gbdi_analyzer.py"),
    ("models/virtual_pl_mapper.py", "models/virtual_pl_mapper.py"),
    ("models/inpainter_controlnet.py", "models/inpainter_controlnet.py"),
    ("models/live_detector_edl.py", "models/live_detector_edl.py"),
    ("losses/physics_loss.py", "losses/physics_loss.py"),
    ("pipeline/train_gan.py", "pipeline/train_gan.py"),
    ("pipeline/generate_synthetic_dataset.py", "pipeline/generate_synthetic_dataset.py"),
    ("pipeline/train_live_detector.py", "pipeline/train_live_detector.py"),
    ("evaluation/biomed_microscopy_benchmark.py", "evaluation/biomed_microscopy_benchmark.py"),
    ("evaluation/benchmark_full.py", "evaluation/benchmark_full.py"),
    ("run_pipeline.py", "run_pipeline.py"),
]

for src_rel, dst_rel in files_to_sync:
    src_fp = os.path.join(src_root, src_rel)
    dst_fp = os.path.join(dst_root, dst_rel)
    
    os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    if os.path.exists(src_fp):
        shutil.copy2(src_fp, dst_fp)
        print(f"Synced: {dst_rel}")

print("Sync complete!")
