import os
import shutil
import random
from pathlib import Path

def prepare_dataset():
    # Define paths
    base_dir = Path(".")
    my_frames = base_dir / "data/frames/original"
    external_data = base_dir / "data/external/mongol_dataset"
    final_output = base_dir / "datasets/mushroom_final"
    
    # Create required directory structure
    for split in ['train', 'val']:
        (final_output / "images" / split).mkdir(parents=True, exist_ok=True)
        (final_output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Process external dataset (Mongol)
    for split_src, split_dst in [('train', 'train'), ('valid', 'val')]:
        src_img_dir = external_data / split_src / "images"
        if not src_img_dir.exists(): 
            continue
            
        for img in src_img_dir.glob("*.jpg"):
            shutil.copy(img, final_output / "images" / split_dst)
            
            label_src = external_data / split_src / "labels" / img.with_suffix('.txt').name
            label_dst = final_output / "labels" / split_dst / label_src.name
            
            if label_src.exists():
                # Remap classes: 0 -> 0 (edible), 1 -> 1 (poisonous), 2 -> 1 (not recommended -> poisonous)
                with open(label_src, 'r') as f_in, open(label_dst, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        class_id = int(parts[0])
                        # Merge "not recommended" with "poisonous" for safety
                        if class_id == 2:
                            class_id = 1
                            
                        parts[0] = str(class_id)
                        f_out.write(" ".join(parts) + "\n")

    # 2. Process local frames
    all_my_images = list(my_frames.glob("*.jpg"))
    random.shuffle(all_my_images)
    
    # 80/20 split for local frames
    split_idx = int(len(all_my_images) * 0.8)
    train_frames = all_my_images[:split_idx]
    val_frames = all_my_images[split_idx:]

    for frames, split in [(train_frames, 'train'), (val_frames, 'val')]:
        for img in frames:
            shutil.copy(img, final_output / "images" / split)
            
            label_src = img.with_suffix('.txt')
            label_dst = final_output / "labels" / split / label_src.name
            
            if label_src.exists():
                shutil.copy(label_src, label_dst)

    # 3. Create mushrooms.yaml configuration file
    # Using posix() to ensure paths use forward slashes even on Windows
    yaml_content = f"""
train: {final_output.absolute().as_posix()}/images/train
val: {final_output.absolute().as_posix()}/images/val

nc: 2
names: ['edible', 'poisonous']
"""
    with open("mushrooms.yaml", "w") as f:
        f.write(yaml_content.strip())

    print(f"Dataset ready at {final_output}. Config file: mushrooms.yaml")

if __name__ == "__main__":
    prepare_dataset()