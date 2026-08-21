import os
import shutil

src_root = r"C:\Users\Sahil\.gemini\antigravity-ide\scratch\sem_gan_project"
dst_root = r"C:\Users\Sahil\Downloads\SEM_GAN_Dissertation"

# Copy all directories and files recursively from scratch project to Downloads project
for root, dirs, files in os.walk(src_root):
    # Skip .git, logs, checkpoints cache
    if ".git" in root or "__pycache__" in root:
        continue
        
    rel_path = os.path.relpath(root, src_root)
    dst_dir = os.path.join(dst_root, rel_path) if rel_path != "." else dst_root
    os.makedirs(dst_dir, exist_ok=True)
    
    for f in files:
        if f.endswith(('.pyc', '.pth', '.log')) and not f.endswith('generator.py'):
            continue
        src_file = os.path.join(root, f)
        dst_file = os.path.join(dst_dir, f)
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {os.path.relpath(dst_file, dst_root)}")

# Also create empty __init__.py files in subfolders to ensure python package recognition
for sub in ["data", "models", "losses", "metrics", "pipeline", "evaluation"]:
    p = os.path.join(dst_root, sub, "__init__.py")
    if not os.path.exists(p):
        with open(p, "w") as fp:
            fp.write("# Init\n")
        print(f"Created __init__.py in {sub}")

print("\nFull synchronization complete!")
