import os
import shutil
import yaml
import time
import gc
import sys
import torch
from PIL import Image

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Paths
BASE_DIR = os.path.normpath("E:/Kiril_Horobets/mushroom_data")
FINAL_DIR = os.path.join(BASE_DIR, "mushroom_final")
DEST_IMAGES = os.path.join(FINAL_DIR, "images/train")
DEST_LABELS = os.path.join(FINAL_DIR, "labels/train")
REGISTRY_FILE = os.path.join(FINAL_DIR, "class_registry.txt")

STAGING_IN = os.path.join(BASE_DIR, "staging_in")
STAGING_OUT = os.path.join(BASE_DIR, "staging_out")

IGNORE_DIRS = [
    "mushroom_final",
    "runs",
    "dataset",
    "staging_in",
    "staging_out",
    "train.cache",
    "val.cache",
]
BATCH_SIZE = 300


def get_or_create_id(species_name):
    registry = {}
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    name, id_val = line.strip().split("=")
                    registry[name] = int(id_val)

    if species_name in registry:
        return registry[species_name]

    new_id = max(registry.values()) + 1 if registry else 0
    with open(REGISTRY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{species_name}={new_id}\n")
    return new_id


def generate_yaml():
    registry = {}
    if not os.path.exists(REGISTRY_FILE):
        return
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                name, id_val = line.strip().split("=")
                registry[int(id_val)] = name

    sorted_names = [registry[i] for i in range(len(registry)) if i in registry]

    yaml_data = {
        "path": FINAL_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(sorted_names),
        "names": sorted_names,
    }

    with open(os.path.join(FINAL_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=None)
    print(f"[OK] data.yaml updated, classes={len(sorted_names)}")
    sys.stdout.flush()


def clear_staging():
    for d in [STAGING_IN, STAGING_OUT]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        os.makedirs(d, exist_ok=True)


def save_single_label(f, src_path, class_id, generated_labels):
    # Write YOLO label for one image (empty file if model produced no box)
    txt_name = os.path.splitext(f)[0] + ".txt"
    src_img_path = os.path.join(src_path, f)

    if txt_name in generated_labels:
        with open(generated_labels[txt_name], "r", encoding="utf-8") as f_in, open(
            os.path.join(DEST_LABELS, txt_name), "w", encoding="utf-8"
        ) as f_out:
            for line in f_in.readlines():
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = str(class_id)
                    f_out.write(" ".join(parts[:5]) + "\n")
        shutil.copy2(src_img_path, os.path.join(DEST_IMAGES, f))
    else:
        open(os.path.join(DEST_LABELS, txt_name), "w", encoding="utf-8").close()
        shutil.copy2(src_img_path, os.path.join(DEST_IMAGES, f))


def run_pipeline():
    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_LABELS, exist_ok=True)

    base_model = GroundingDINO(ontology=CaptionOntology({"mushroom": "mushroom"}))
    processed_something = False

    for folder_name in os.listdir(BASE_DIR):
        if folder_name in IGNORE_DIRS:
            continue

        src_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(src_path):
            continue

        class_id = get_or_create_id(folder_name)
        files_to_process = []
        current_time = time.time()

        for f in os.listdir(src_path):
            if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            txt_name = os.path.splitext(f)[0] + ".txt"
            file_full_path = os.path.join(src_path, f)

            if os.path.exists(os.path.join(DEST_LABELS, txt_name)):
                continue
            if current_time - os.path.getmtime(file_full_path) < 10:
                continue

            if not os.path.exists(file_full_path) or os.path.getsize(file_full_path) == 0:
                continue
            try:
                with Image.open(file_full_path) as img:
                    img.verify()
            except Exception:
                print(f"[WARN] Removing corrupted image: {folder_name}/{f}")
                try:
                    os.remove(file_full_path)
                except Exception:
                    pass
                continue

            files_to_process.append(f)

        if not files_to_process:
            continue

        print(f"[INFO] Labeling {folder_name}: {len(files_to_process)} images")
        sys.stdout.flush()
        processed_something = True

        for i in range(0, len(files_to_process), BATCH_SIZE):
            batch_files = files_to_process[i : i + BATCH_SIZE]
            clear_staging()

            for f in batch_files:
                try:
                    shutil.copy2(os.path.join(src_path, f), os.path.join(STAGING_IN, f))
                except Exception:
                    pass

            try:
                # Prefer whole-batch labeling
                with torch.inference_mode():
                    base_model.label(input_folder=STAGING_IN, output_folder=STAGING_OUT)

                generated_labels = {}
                for root, _, files in os.walk(STAGING_OUT):
                    for f in files:
                        if f.endswith(".txt") and f not in ["classes.txt", "data.yaml"]:
                            generated_labels[f] = os.path.join(root, f)

                for f in batch_files:
                    save_single_label(f, src_path, class_id, generated_labels)

            except Exception as batch_error:
                # Batch failed: retry one file at a time
                print(f"[WARN] Batch failed ({batch_error}), falling back to single-item")
                sys.stdout.flush()

                for single_f in batch_files:
                    clear_staging()
                    try:
                        shutil.copy2(
                            os.path.join(src_path, single_f),
                            os.path.join(STAGING_IN, single_f),
                        )
                        with torch.inference_mode():
                            base_model.label(
                                input_folder=STAGING_IN, output_folder=STAGING_OUT
                            )

                        single_labels = {}
                        for root, _, files in os.walk(STAGING_OUT):
                            for sf in files:
                                if sf.endswith(".txt") and sf not in [
                                    "classes.txt",
                                    "data.yaml",
                                ]:
                                    single_labels[sf] = os.path.join(root, sf)

                        save_single_label(single_f, src_path, class_id, single_labels)
                    except Exception as single_error:
                        print(
                            f"[ERR] Dropping broken file {folder_name}/{single_f}: "
                            f"{single_error}"
                        )
                        try:
                            os.remove(os.path.join(src_path, single_f))
                        except Exception:
                            pass
                        continue
            finally:
                clear_staging()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if processed_something:
        generate_yaml()

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return processed_something


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("[INFO] Auto-label daemon started")
    sys.stdout.flush()

    while True:
        had_work = run_pipeline()
        if not had_work:
            print("[INFO] Idle, sleeping 90s")
            sys.stdout.flush()
            time.sleep(90)
