#!/usr/bin/env python3
"""
iNaturalist Research-Grade photo harvester for mushroom species.

Used on the university workstation to build the local image corpus under
E:/Kiril_Horobets/mushroom_data. Target: up to ~1000 photos per species
(20 pages x 50 observations), with resume/skip and HTTP retries.

Author: Kiril Horobets, WUT 2026
"""

from __future__ import annotations

import os
import sys
import time

import requests

# Target list for the species-level corpus (105 taxa, including look-alikes).
SPECIES_LIST = [
    "Boletus edulis", "Amanita muscaria", "Cantharellus cibarius", "Imleria badia",
    "Russula cyanoxantha", "Suillus luteus", "Lactarius deliciosus", "Macrolepiota procera",
    "Armillaria mellea", "Coprinus comatus", "Agaricus campestris", "Boletus reticulatus",
    "Leccinum scabrum", "Xerocomus chrysenteron", "Amanita pantherina", "Amanita phalloides",
    "Russula virescens", "Russula emetica", "Clitocybe nebularis", "Pleurotus ostreatus",
    "Craterellus cornucopioides", "Hydnum repandum", "Fistulina hepatica", "Calvatia gigantea",
    "Morchella esculenta", "Gyromitra esculenta", "Hygrophorus marzuolus", "Tricholoma equestre",
    "Lepista nuda", "Flammulina velutipes", "Panellus stipticus", "Trametes versicolor",
    "Ganoderma lucidum", "Daedaleopsis confragosa", "Fomes fomentarius", "Piptoporus betulinus",
    "Schizophyllum commune", "Auricularia auricula-judae", "Exidia glandulosa", "Sarcoscypha coccinea",
    "Tremella mesenterica", "Clavulina coralloides", "Cantharellus tubaeformis",
    "Hygrocybe conica", "Laccaria laccata", "Mycena galericulata", "Mycena pura",
    "Collybia dryophila", "Gymnopilus junonius", "Hypholoma fasciculare", "Pholiota squarrosa",
    "Stropharia aeruginosa", "Psilocybe semilanceata", "Inocybe geophylla", "Cortinarius armillatus",
    "Cortinarius violaceus", "Paxillus involutus", "Chroogomphus rutilus", "Gomphidius glutinosus",
    "Boletus pinophilus", "Leccinum aurantiacum", "Tylopilus felleus", "Scleroderma citrinum",
    "Lycoperdon perlatum", "Bovista plumbea", "Cyathus striatus", "Crucibulum laeve",
    "Helvella crispa", "Verpa bohemica", "Amanita rubescens", "Amanita citrina",
    "Amanita fulva", "Russula nigricans", "Russula xerampelina", "Russula paludosa",
    "Lactarius torminosus", "Lactarius turpis", "Lactarius piperatus", "Lactarius rufus",
    "Suillus granulatus", "Suillus grevillei", "Xerocomellus pruinatus", "Pseudoboletus parasiticus",
    "Gyroporus castaneus", "Clitopilus prunulus", "Entoloma sinuatum", "Lepiota procera",
    "Chlorophyllum rhacodes", "Agaricus arvensis", "Marasmius oreades", "Volvariella gloiocephala",
    "Pluteus cervinus", "Tubaria furfuracea", "Galerina marginata", "Phallus impudicus",
    "Hygrophoropsis aurantiaca", "Amanita virosa", "Agaricus xanthodermus", "Rubroboletus satanas",
    "Omphalotus olearius", "Caloboletus calopus", "Lepiota cristata", "Tricholoma sulphureum",
    "Cortinarius rubellus", "Lactarius helvus",
]

BASE_DATA_DIR = "E:/Kiril_Horobets/mushroom_data"
TARGET_PER_CLASS = 1000
PAGES = 20
PER_PAGE = 50
MAX_RETRIES = 3


def count_valid_files(folder: str) -> int:
    """Count existing non-empty files in a species folder."""
    if not os.path.isdir(folder):
        return 0
    return sum(
        1
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.getsize(os.path.join(folder, name)) > 0
    )


def download_species(session: requests.Session, species: str) -> bool:
    """
    Download Research-Grade photos for one taxon.
    Returns True when the folder already has TARGET_PER_CLASS valid images.
    """
    folder = f"{BASE_DATA_DIR}/{species.replace(' ', '_')}"
    os.makedirs(folder, exist_ok=True)

    existing_files = count_valid_files(folder)
    if existing_files >= TARGET_PER_CLASS:
        print(f"Skip: {species} (already has {existing_files} valid photos).")
        sys.stdout.flush()
        return True

    print(f"\n>>> Launching downloader for: {species} (Current VALID count: {existing_files}) <<<")
    sys.stdout.flush()

    for page in range(1, PAGES + 1):
        encoded_name = species.replace(" ", "+")
        url = (
            "https://api.inaturalist.org/v1/observations"
            f"?taxon_name={encoded_name}&has[]=photos&quality_grade=research"
            f"&per_page={PER_PAGE}&page={page}"
        )

        print(f"--- Processing: {species} | Page: {page} ---")
        sys.stdout.flush()

        results = []
        for retry in range(MAX_RETRIES):
            try:
                response = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    break
                if response.status_code in (429, 500, 503):
                    print(
                        f"Server returned {response.status_code} for {species} P.{page}. "
                        f"Retrying in 5s... ({retry + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(5)
            except Exception as e:
                print(f"Network error on page {page} (Attempt {retry + 1}/{MAX_RETRIES}): {e}")
                time.sleep(5)

        if not results:
            print(f"No more data accessible for {species} at page {page}.")
            break

        for obs in results:
            for photo in obs.get("photos", []):
                img_url = photo["url"].replace("square", "large")
                filename = os.path.join(folder, f"{obs['id']}_{photo['id']}.jpg")

                if os.path.exists(filename):
                    continue
                try:
                    img_data = session.get(img_url, timeout=10).content
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    time.sleep(0.3)
                except Exception:
                    continue

    final_count = count_valid_files(folder)
    return final_count >= TARGET_PER_CLASS


def main() -> None:
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    # Persistent session reuses TCP connections and reduces descriptor churn.
    with requests.Session() as session:
        while True:
            all_completed = True
            print("\n=== STARTING DATASET AUDIT CYCLE ===")
            sys.stdout.flush()

            for species in SPECIES_LIST:
                if not download_species(session, species):
                    all_completed = False

            if all_completed:
                print(
                    f"\nSUCCESS: all {len(SPECIES_LIST)} species have "
                    f"{TARGET_PER_CLASS}+ photos. Pipeline finished."
                )
                sys.stdout.flush()
                break

            print(
                "\nCycle finished. Some species need more data or the server was down. "
                "Sleeping 60s before next scan..."
            )
            sys.stdout.flush()
            time.sleep(60)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
