import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".txt"}
IGNORE_DIRS = {".git", "__pycache__", "venv", ".idea"}
OUTPUT_FILE = "project_structure.txt"


def write_tree(dir_path, prefix, file_obj):
    try:
        items = os.listdir(dir_path)
    except PermissionError:
        return

    dirs = sorted(
        [
            d
            for d in items
            if os.path.isdir(os.path.join(dir_path, d)) and d not in IGNORE_DIRS
        ]
    )
    files = sorted(
        [
            f
            for f in items
            if os.path.isfile(os.path.join(dir_path, f)) and f != OUTPUT_FILE
        ]
    )

    images = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    others = [f for f in files if f not in images]

    display_items = []

    for d in dirs:
        display_items.append(("dir", d))

    for f in others:
        display_items.append(("file", f))

    # Show at most 5 image/label files per folder
    for i, img in enumerate(images):
        if i < 5:
            display_items.append(("file", img))
        elif i == 5:
            display_items.append(
                ("summary", f"... and {len(images) - 5} more images/txt files")
            )
            break

    count = len(display_items)
    for i, (item_type, name) in enumerate(display_items):
        is_last = i == count - 1
        connector = "`- " if is_last else "|- "
        file_obj.write(prefix + connector + name + "\n")

        if item_type == "dir":
            extension = "   " if is_last else "|  "
            write_tree(os.path.join(dir_path, name), prefix + extension, file_obj)


if __name__ == "__main__":
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        current_dir_name = os.path.basename(os.getcwd())
        f.write(f"{current_dir_name}/\n")
        write_tree(".", "", f)

    print(f"[OK] Structure written to {OUTPUT_FILE}")
