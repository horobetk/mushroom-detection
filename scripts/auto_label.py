import os
from ultralytics import YOLO
from pathlib import Path

# Load the largest model for generating draft bounding boxes
model = YOLO('yolov8x.pt') 

input_path = Path("data/frames/original")
images = list(input_path.glob("*.jpg"))

print(f"Found {len(images)} frames for auto-labeling...")

for img_path in images:
    # Run object detection
    results = model(img_path, conf=0.25)
    
    # Define path for YOLO format .txt file
    txt_path = img_path.with_suffix('.txt')
    
    with open(txt_path, 'w') as f:
        for box in results[0].boxes:
            # Get coordinates in YOLO format (normalized xywh)
            coords = box.xywhn[0].tolist()
            # Set temporary class 0 (will be manually corrected if needed)
            f.write(f"0 {' '.join(map(str, coords))}\n")

print("Auto-labeling completed. Check .txt files in data/frames/original")