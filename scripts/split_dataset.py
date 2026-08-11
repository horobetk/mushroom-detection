import os
import random
import shutil
import sys

BASE_DIR = "E:/Kiril_Horobets/mushroom_data/mushroom_final"
REGISTRY_FILE = os.path.join(BASE_DIR, "class_registry.txt")

SRC_IMG_DIR = os.path.join(BASE_DIR, "images/train")
SRC_LBL_DIR = os.path.join(BASE_DIR, "labels/train")

VAL_RATIO = 0.10
TEST_RATIO = 0.10

#Additional classes start at 105..
NEW_CLASSES_START_ID = 105


def get_species_list():
    if not os.path.exists(REGISTRY_FILE):
        print(f"[ERR] Registry not found: {REGISTRY_FILE}")
        return {}
    species_map = {}
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                name, id_val = line.strip().split("=")
                species_map[int(id_val)] = name
    return species_map


def run_stratified_split():
    species_map = get_species_list()
    if not species_map:
        return

    for split in ["val", "test"]:
        os.makedirs(os.path.join(BASE_DIR, f"images/{split}"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, f"labels/{split}"), exist_ok=True)

    class_buckets = {
        class_id: []
        for class_id in species_map.keys()
        if class_id >= NEW_CLASSES_START_ID
    }
    all_images = [
        f
        for f in os.listdir(SRC_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        txt_path = os.path.join(SRC_LBL_DIR, txt_name)

        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        if class_id >= NEW_CLASSES_START_ID:
                            class_buckets[class_id].append(img_name)
            except Exception:
                continue

    print(f"[INFO] Stratified split for class IDs >= {NEW_CLASSES_START_ID}")

    for class_id, img_list in class_buckets.items():
        if not img_list:
            continue

        species_name = species_map[class_id]
        total_count = len(img_list)
        random.shuffle(img_list)

        val_size = int(total_count * VAL_RATIO)
        test_size = int(total_count * TEST_RATIO)

        val_files = img_list[:val_size]
        test_files = img_list[val_size : val_size + test_size]

        print(
            f"Class {class_id:03d} [{species_name}]: "
            f"n={total_count} -> val={len(val_files)}, test={len(test_files)}"
        )
        sys.stdout.flush()

        for img_name in val_files:
            base = os.path.splitext(img_name)[0]
            shutil.move(
                os.path.join(SRC_IMG_DIR, img_name),
                os.path.join(BASE_DIR, "images/val", img_name),
            )
            shutil.move(
                os.path.join(SRC_LBL_DIR, base + ".txt"),
                os.path.join(BASE_DIR, "labels/val", base + ".txt"),
            )

        for img_name in test_files:
            base = os.path.splitext(img_name)[0]
            shutil.move(
                os.path.join(SRC_IMG_DIR, img_name),
                os.path.join(BASE_DIR, "images/test", img_name),
            )
            shutil.move(
                os.path.join(SRC_LBL_DIR, base + ".txt"),
                os.path.join(BASE_DIR, "labels/test", base + ".txt"),
            )

    print("[OK] Split finished")
    sys.stdout.flush()


if __name__ == "__main__":
    random.seed(42)
    run_stratified_split()
