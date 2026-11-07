#!/usr/bin/env python3
"""Auto-label harvested clean test images with GroundingDINO."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time

import yaml
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png")


def read_registry(registry_file: str) -> dict[str, int]:
    reg: dict[str, int] = {}
    if not os.path.exists(registry_file):
        print(f"[ERR] Registry not found: {registry_file}")
        return reg
    with open(registry_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                name, idv = line.strip().split("=")
                reg[name] = int(idv)
    return reg


def backup_old_test(final_dir: str) -> None:
    for kind in ("images", "labels"):
        src = os.path.join(final_dir, kind, "test")
        if os.path.isdir(src) and os.listdir(src):
            dst = os.path.join(final_dir, kind, "test_polluted_backup")
            if os.path.exists(dst):
                print(f"[INFO] backup already exists, leaving as-is: {dst}")
            else:
                shutil.move(src, dst)
                print(f"[OK] moved leaked test aside: {src} -> {dst}")
        os.makedirs(os.path.join(final_dir, kind, "test"), exist_ok=True)


def clear_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def valid_images(folder: str) -> list[str]:
    files = []
    for f in os.listdir(folder):
        if not f.lower().endswith(IMG_EXTS):
            continue
        full = os.path.join(folder, f)
        if os.path.getsize(full) == 0:
            continue
        try:
            with Image.open(full) as img:
                img.verify()
        except Exception:
            print(f"[WARN] corrupted, skipping: {f}")
            continue
        files.append(f)
    return files


def update_yaml_test_key(final_dir: str) -> None:
    yaml_path = os.path.join(final_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[WARN] data.yaml not found: {yaml_path}")
        return
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data["test"] = "images/test"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=None, allow_unicode=True)
    print("[OK] data.yaml now has test: images/test")


def run(raw_base: str, final_dir: str, batch_size: int) -> None:
    import torch
    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    registry = read_registry(os.path.join(final_dir, "class_registry.txt"))
    if not registry:
        return

    dest_img = os.path.join(final_dir, "images", "test")
    dest_lbl = os.path.join(final_dir, "labels", "test")
    # Staging lives outside mushroom_data so auto_label.py ignores it.
    staging_root = os.path.dirname(os.path.normpath(raw_base))
    staging_in = os.path.join(staging_root, "staging_test_in")
    staging_out = os.path.join(staging_root, "staging_test_out")

    backup_old_test(final_dir)
    os.makedirs(dest_lbl, exist_ok=True)

    model = GroundingDINO(ontology=CaptionOntology({"mushroom": "mushroom"}))

    if not os.path.isdir(raw_base):
        print(f"[ERR] raw base not found: {raw_base}")
        return

    for species in sorted(os.listdir(raw_base)):
        src = os.path.join(raw_base, species)
        if not os.path.isdir(src):
            continue
        if species not in registry:
            print(f"[SKIP] '{species}' not in registry, ignoring.")
            continue
        class_id = registry[species]
        files = valid_images(src)
        if not files:
            continue
        print(f"[INFO] labeling {species} (id={class_id}): {len(files)} images")
        sys.stdout.flush()

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            clear_dir(staging_in)
            clear_dir(staging_out)
            for f in batch:
                try:
                    shutil.copy2(os.path.join(src, f), os.path.join(staging_in, f))
                except OSError:
                    pass
            try:
                with torch.inference_mode():
                    model.label(input_folder=staging_in, output_folder=staging_out)
                generated = {}
                for root, _, gfiles in os.walk(staging_out):
                    for gf in gfiles:
                        if gf.endswith(".txt") and gf not in ("classes.txt", "data.yaml"):
                            generated[gf] = os.path.join(root, gf)
                for f in batch:
                    txt = os.path.splitext(f)[0] + ".txt"
                    out_lbl = os.path.join(dest_lbl, txt)
                    if txt in generated:
                        with open(generated[txt], "r", encoding="utf-8") as fin, \
                             open(out_lbl, "w", encoding="utf-8") as fout:
                            for line in fin:
                                parts = line.split()
                                if len(parts) >= 5:
                                    parts[0] = str(class_id)
                                    fout.write(" ".join(parts[:5]) + "\n")
                    else:
                        open(out_lbl, "w", encoding="utf-8").close()
                    shutil.copy2(os.path.join(src, f), os.path.join(dest_img, f))
            finally:
                clear_dir(staging_in)
                clear_dir(staging_out)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    update_yaml_test_key(final_dir)
    n_img = len([f for f in os.listdir(dest_img) if f.lower().endswith(IMG_EXTS)])
    print(f"\n[DONE] clean test images: {n_img}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-base", default="E:/Kiril_Horobets/clean_test_raw")
    p.add_argument("--final-dir", default="E:/Kiril_Horobets/mushroom_data/mushroom_final")
    p.add_argument("--batch-size", type=int, default=200)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.raw_base, args.final_dir, args.batch_size)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
