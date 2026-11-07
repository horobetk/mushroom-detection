#!/usr/bin/env python3
"""
Harvest a fresh, leak-free test set from iNaturalist.

Context
-------
The original split leaked at the observation level: ~95% of ``images/test``
shared an ``obs_id`` with ``train``/``val`` (the auto-labeller re-copied files
that the splitter had already moved). A trustworthy test set therefore has to be
built from observations the model has never seen.

This script downloads *new* research-grade observations for every registered
species, skipping any ``obs_id`` already present in ``images/train`` or
``images/val``. Photos land in ``<base>/clean_test_raw/<Species>/`` using the
same ``{obs_id}_{photo_id}.jpg`` convention, ready for ``label_clean_test.py``.

It does NOT touch train/val and never renames existing files.

Author: Kiril Horobets, WUT 2026
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Set

import requests

IMG_EXTS = (".jpg", ".jpeg", ".png")
API_URL = "https://api.inaturalist.org/v1/observations"


def obs_id_of(filename: str) -> str:
    return os.path.basename(filename).split("_")[0]


def collect_seen_obs_ids(*folders: str) -> Set[str]:
    seen: Set[str] = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(IMG_EXTS):
                seen.add(obs_id_of(name))
    return seen


def read_species_list(registry_file: str) -> list[str]:
    """Return folder-style species names (e.g. 'Boletus_edulis') from the registry."""
    species = []
    if not os.path.exists(registry_file):
        print(f"[ERR] Registry not found: {registry_file}")
        return species
    with open(registry_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                name = line.strip().split("=")[0]
                species.append(name)
    return species


def count_valid_files(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1
        for n in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, n)) and os.path.getsize(os.path.join(folder, n)) > 0
    )


def fetch_page(session: requests.Session, taxon_name: str, page: int, per_page: int,
               max_retries: int) -> list:
    url = (
        f"{API_URL}?taxon_name={taxon_name.replace(' ', '+')}"
        f"&has[]=photos&quality_grade=research&per_page={per_page}&page={page}"
    )
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("results", [])
            if resp.status_code in (429, 500, 503):
                print(f"  [{resp.status_code}] {taxon_name} p.{page}, retry {attempt + 1}/{max_retries}")
                time.sleep(5)
        except requests.RequestException as exc:
            print(f"  net error {taxon_name} p.{page} ({attempt + 1}/{max_retries}): {exc}")
            time.sleep(5)
    return []


def harvest_species(session: requests.Session, folder_species: str, seen_ids: Set[str],
                    out_base: str, target: int, max_photos_per_obs: int,
                    max_pages: int, per_page: int, max_retries: int) -> int:
    taxon_name = folder_species.replace("_", " ")
    out_dir = os.path.join(out_base, folder_species)
    os.makedirs(out_dir, exist_ok=True)

    have = count_valid_files(out_dir)
    if have >= target:
        print(f"Skip: {folder_species} (already {have} fresh photos).")
        return have

    print(f"\n>>> {folder_species}: need {target}, have {have} <<<")
    sys.stdout.flush()
    downloaded = have

    for page in range(1, max_pages + 1):
        if downloaded >= target:
            break
        results = fetch_page(session, taxon_name, page, per_page, max_retries)
        if not results:
            print(f"  no more data at page {page}")
            break

        new_here = 0
        for obs in results:
            if downloaded >= target:
                break
            oid = str(obs.get("id"))
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            photos = obs.get("photos", [])[:max_photos_per_obs]
            for photo in photos:
                if downloaded >= target:
                    break
                img_url = photo["url"].replace("square", "large")
                fname = os.path.join(out_dir, f"{oid}_{photo['id']}.jpg")
                if os.path.exists(fname):
                    continue
                try:
                    data = session.get(img_url, timeout=10).content
                    with open(fname, "wb") as fh:
                        fh.write(data)
                    downloaded += 1
                    new_here += 1
                    time.sleep(0.3)
                except requests.RequestException:
                    continue
        print(f"  page {page}: +{new_here} (total {downloaded}/{target})")
        sys.stdout.flush()

    return downloaded


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default="E:/Kiril_Horobets/mushroom_data/mushroom_final")
    p.add_argument("--raw-base", default="E:/Kiril_Horobets/clean_test_raw",
                   help="Where fresh per-species photos are staged (kept OUTSIDE "
                        "mushroom_data so auto_label.py never picks it up).")
    p.add_argument("--target", type=int, default=50,
                   help="Fresh photos to collect per class.")
    p.add_argument("--max-photos-per-obs", type=int, default=3,
                   help="Cap photos per observation to maximise observation diversity.")
    p.add_argument("--max-pages", type=int, default=40)
    p.add_argument("--per-page", type=int, default=50)
    p.add_argument("--max-retries", type=int, default=3)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    registry = os.path.join(args.base_dir, "class_registry.txt")
    img_train = os.path.join(args.base_dir, "images", "train")
    img_val = os.path.join(args.base_dir, "images", "val")

    species = read_species_list(registry)
    if not species:
        return
    seen = collect_seen_obs_ids(img_train, img_val)
    print(f"[INFO] classes: {len(species)}, train/val obs_id to avoid: {len(seen)}")
    os.makedirs(args.raw_base, exist_ok=True)

    summary = {}
    with requests.Session() as session:
        for sp in species:
            got = harvest_species(session, sp, seen, args.raw_base, args.target,
                                  args.max_photos_per_obs, args.max_pages,
                                  args.per_page, args.max_retries)
            summary[sp] = got

    thin = {k: v for k, v in summary.items() if v < args.target}
    print(f"\n[DONE] fresh photos staged under {args.raw_base}")
    print(f"[INFO] classes below target ({args.target}): {len(thin)}")
    for k, v in sorted(thin.items(), key=lambda kv: kv[1]):
        print(f"   {v:4d}  {k}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
