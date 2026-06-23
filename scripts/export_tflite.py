#!/usr/bin/env python3
"""
Eksport wytrenowanego modelu YOLO11 do formatu TensorFlow Lite (.tflite).

Format TFLite jest docelowym formatem dla aplikacji mobilnej (Android / CameraX).
Skrypt obsluguje trzy tryby precyzji:
  - int8  : kwantyzacja 8-bitowa (najmniejszy rozmiar, najszybsza inferencja na
            CPU mobilnych; wymaga datasetu kalibracyjnego --data),
  - fp16  : polowiczna precyzja (dobry kompromis rozmiar/dokladnosc),
  - fp32  : pelna precyzja (najwieksza dokladnosc, najwiekszy plik).

Autor: Kiril Horobets
Politechnika Warszawska, 2026
Praca inzynierska: System rozpoznawania grzybow na urzadzeniach mobilnych
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Wymuszenie UTF-8 na konsoli (np. Windows cp1252 + sciezki z polskimi/cyrylickimi znakami)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Tryb headless - spojnie z train_mvp.py (klaster bez serwera X).
os.environ.setdefault("MPLBACKEND", "Agg")

from ultralytics import YOLO  # noqa: E402

DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"
DEFAULT_DATA = "datasets/mushrooms_mvp/mushrooms_mvp.yaml"


def _run_export(weights: Path, quant: str, imgsz: int, data: str) -> Path:
    """
    Pojedyncza proba eksportu modelu do .tflite w zadanym trybie precyzji.

    Returns:
        Path: Sciezka do wygenerowanego pliku .tflite.
    """
    model = YOLO(str(weights))

    export_kwargs = {
        "format": "tflite",
        "imgsz": imgsz,
    }

    if quant == "int8":
        # Kwantyzacja int8 wymaga reprezentatywnego datasetu do kalibracji.
        export_kwargs["int8"] = True
        export_kwargs["data"] = data
    elif quant == "fp16":
        export_kwargs["half"] = True
    # fp32 - parametry domyslne (bez int8/half)

    exported = model.export(**export_kwargs)
    return Path(str(exported))


def export_model(args) -> Path:
    """
    Eksportuje model .pt do .tflite zgodnie z wybranym trybem precyzji.

    Jesli wybrano int8 i eksport sie nie powiedzie (np. brak / niekompatybilne
    biblioteki TensorFlow na danej maszynie), nastepuje automatyczne przelaczenie
    na tryb fp16, ktory jest mniej wymagajacy.

    Returns:
        Path: Sciezka do wygenerowanego pliku .tflite.
    """
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Nie znaleziono wag: {weights}\n"
            f"Najpierw uruchom trening (scripts/train_mvp.py)."
        )

    print("\n" + "=" * 60)
    print(" EKSPORT DO TENSORFLOW LITE")
    print("=" * 60)
    print(f"  Wagi:      {weights}")
    print(f"  Tryb:      {args.quant}")
    print(f"  Imgsz:     {args.imgsz}")
    if args.quant == "int8":
        print(f"  Kalibracja:{args.data}")
    print("=" * 60 + "\n")

    try:
        exported_path = _run_export(weights, args.quant, args.imgsz, args.data)
    except Exception as exc:  # noqa: BLE001
        if args.quant == "int8" and not args.no_fallback:
            print(f"\n[UWAGA] Eksport int8 nie powiodl sie: {exc}")
            print("[UWAGA] Automatyczne przelaczenie na tryb fp16...\n")
            exported_path = _run_export(weights, "fp16", args.imgsz, args.data)
        else:
            raise

    print(f"\nEksport zakonczony: {exported_path}")

    # Opcjonalne skopiowanie do wskazanej lokalizacji (np. android/app/models/)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(exported_path, out_path)
        print(f"Skopiowano model do: {out_path.resolve()}")
        exported_path = out_path

    size_mb = exported_path.stat().st_size / (1024 * 1024)
    print(f"Rozmiar modelu: {size_mb:.2f} MB")
    return exported_path


def main():
    parser = argparse.ArgumentParser(
        description="Eksport modelu YOLO11 do TensorFlow Lite."
    )
    parser.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS,
                        help=f"Sciezka do wag .pt (domyslnie: {DEFAULT_WEIGHTS})")
    parser.add_argument("--quant", "-q", default="int8",
                        choices=["int8", "fp16", "fp32"],
                        help="Tryb precyzji eksportu (domyslnie: int8)")
    parser.add_argument("--data", "-d", default=DEFAULT_DATA,
                        help="Dataset kalibracyjny dla int8 "
                             f"(domyslnie: {DEFAULT_DATA})")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Rozmiar obrazu wejsciowego (domyslnie: 640)")
    parser.add_argument("--output", "-o", default=None,
                        help="Opcjonalna sciezka docelowa pliku .tflite "
                             "(np. android/app/models/mushroom_detector.tflite)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Wylacz automatyczny fallback int8 -> fp16 przy bledzie")

    args = parser.parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
