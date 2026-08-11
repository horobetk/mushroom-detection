import os
import math

BASE_LABELS_DIR = "E:/Kiril_Horobets/mushroom_data/mushroom_final/labels"
SPLITS = ["train", "val", "test"]

# Dual-metric thresholds for duplicate detection
IOU_THRESHOLD = 0.75
IOS_THRESHOLD = 0.85
CENTER_DIST_THRESHOLD = 0.06


def calculate_box_metrics(box1, box2):
    # IoU, IoS, and center distance for two YOLO boxes [cls, cx, cy, w, h]
    b1_x1, b1_x2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b1_y1, b1_y2 = box1[2] - box1[4] / 2, box1[2] + box1[4] / 2

    b2_x1, b2_x2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    b2_y1, b2_y2 = box2[2] - box2[4] / 2, box2[2] + box2[4] / 2

    center_dist = math.sqrt((box1[1] - box2[1]) ** 2 + (box1[2] - box2[2]) ** 2)

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    b1_area = box1[3] * box1[4]
    b2_area = box2[3] * box2[4]
    min_area = min(b1_area, b2_area)
    union_area = b1_area + b2_area - inter_area + 1e-6

    iou = inter_area / union_area
    ios = inter_area / (min_area + 1e-6)

    return iou, ios, center_dist


def process_label_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        return False, 0

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            boxes.append([cls_id] + coords)

    filtered_boxes = []
    removed_in_file = 0

    for box_a in boxes:
        is_duplicate = False
        for box_b in filtered_boxes:
            if box_a[0] == box_b[0]:
                iou, ios, center_dist = calculate_box_metrics(box_a, box_b)
                if center_dist < CENTER_DIST_THRESHOLD and (
                    iou > IOU_THRESHOLD or ios > IOS_THRESHOLD
                ):
                    is_duplicate = True
                    removed_in_file += 1
                    break

        if not is_duplicate:
            filtered_boxes.append(box_a)

    if removed_in_file > 0:
        with open(file_path, "w") as f:
            for b in filtered_boxes:
                f.write(f"{b[0]} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n")
        return True, removed_in_file

    return False, 0


def clean_all_splits():
    print("[INFO] Starting bbox duplicate cleanup\n")

    total_cleaned_files = 0
    total_removed_boxes = 0

    for split in SPLITS:
        split_dir = os.path.join(BASE_LABELS_DIR, split)
        if not os.path.exists(split_dir):
            print(f"[SKIP] Directory not found: {split_dir}")
            continue

        txt_files = [f for f in os.listdir(split_dir) if f.endswith(".txt")]
        print(f"[INFO] Scanning split '{split}' ({len(txt_files)} label files)")

        split_cleaned_files = 0
        split_removed_boxes = 0

        for txt_file in txt_files:
            file_path = os.path.join(split_dir, txt_file)
            was_cleaned, removed_count = process_label_file(file_path)
            if was_cleaned:
                split_cleaned_files += 1
                split_removed_boxes += removed_count

        print(
            f"  - split '{split}': cleaned {split_cleaned_files} files, "
            f"removed {split_removed_boxes} boxes"
        )
        total_cleaned_files += split_cleaned_files
        total_removed_boxes += split_removed_boxes

    print("-" * 60)
    print("[OK] Cleanup finished")
    print(f"  modified files: {total_cleaned_files}")
    print(f"  removed boxes:  {total_removed_boxes}")


if __name__ == "__main__":
    clean_all_splits()
