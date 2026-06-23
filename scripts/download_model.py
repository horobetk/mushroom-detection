#!/usr/bin/env python3
"""
Pobieranie bazowego (pretrenowanego) modelu YOLO11 dla projektu MVP.

Domyslnie pobiera YOLO11n (nano) - najlzejszy wariant, idealny do
wdrozenia na urzadzeniach mobilnych (TFLite / Android).
Zastepuje wczesniej uzywany model YOLOv8n.

Autor: Kiril Horobets
Politechnika Warszawska, 2026
Praca inzynierska: System rozpoznawania grzybow na urzadzeniach mobilnych
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# Dostepne warianty YOLO11 (od najlzejszego do najciezszego)
AVAILABLE_MODELS = [
    "yolo11n.pt",  # nano  - ~2.6M param  (MVP / mobile)
    "yolo11s.pt",  # small - ~9.4M param
    "yolo11m.pt",  # medium
    "yolo11l.pt",  # large
    "yolo11x.pt",  # extra-large
]

DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_OUTPUT_DIR = "weights"


def download_model(model_name: str, output_dir: str) -> Path:
    """
    Pobiera wskazany pretrenowany model YOLO11 i zapisuje go w katalogu wyjsciowym.

    Args:
        model_name (str): Nazwa modelu, np. 'yolo11n.pt'.
        output_dir (str): Katalog docelowy na plik wag.

    Returns:
        Path: Sciezka do pobranego pliku wag.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = out_dir / model_name

    if target_path.exists():
        print(f"Model juz istnieje: {target_path} (pomijam pobieranie)")
        return target_path

    print(f"Pobieranie bazowego modelu: {model_name} ...")

    # Konstruktor YOLO() automatycznie pobiera wagi z najnowszego release Ultralytics.
    # Plik laduje sie w biezacym katalogu roboczym, dlatego przenosimy go do out_dir.
    YOLO(model_name)

    downloaded = Path(model_name)
    if downloaded.exists() and downloaded.resolve() != target_path.resolve():
        downloaded.replace(target_path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Nie udalo sie zlokalizowac pobranego pliku modelu: {model_name}"
        )

    size_mb = target_path.stat().st_size / (1024 * 1024)
    print("\nGotowe!")
    print(f"   - Model:    {model_name}")
    print(f"   - Lokacja:  {target_path.resolve()}")
    print(f"   - Rozmiar:  {size_mb:.1f} MB")

    return target_path


def main():
    parser = argparse.ArgumentParser(
        description="Pobieranie bazowego modelu YOLO11 (domyslnie nano)."
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Wariant modelu YOLO11 do pobrania (domyslnie: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Katalog docelowy na wagi (domyslnie: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    download_model(args.model, args.output)


if __name__ == "__main__":
    main()
