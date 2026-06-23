#!/usr/bin/env python3
"""
Trening modelu MVP detektora grzybow w oparciu o YOLO11n.

Skrypt:
  - inicjalizuje bazowy model YOLO11n (pretrenowany),
  - wczytuje konfiguracje datasetu (mushrooms_mvp.yaml),
  - uruchamia trening z parametrami domyslnymi: epochs=30, imgsz=640, batch=16.

Przystosowany do uruchamiania na zdalnym klastrze obliczeniowym (Linux):
  - automatyczny wybor GPU (CUDA) z fallbackiem na CPU,
  - tryb headless (brak okien GUI / wykresow) - backend matplotlib 'Agg',
  - brak interaktywnych podgladow (Ultralytics 'show=False').

Autor: Kiril Horobets
Politechnika Warszawska, 2026
Praca inzynierska: System rozpoznawania grzybow na urzadzeniach mobilnych
"""

import argparse
import os
import sys
from pathlib import Path

# Wymuszenie UTF-8 na konsoli (np. Windows cp1252 + sciezki z polskimi/cyrylickimi znakami)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# --- Tryb headless: ustawiamy PRZED importem bibliotek rysujacych wykresy ---
# Gwarantuje, ze na klastrze bez serwera X nie nastapi proba otwarcia okna GUI.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_VERBOSE", "True")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_MODEL = "weights/yolo11n.pt"
DEFAULT_DATA = "datasets/mushrooms_mvp/mushrooms_mvp.yaml"


def resolve_device(requested: str) -> str:
    """
    Ustala urzadzenie obliczeniowe.

    Args:
        requested (str): 'auto', 'cpu', 'cuda' lub indeks GPU (np. '0', '0,1').

    Returns:
        str: Urzadzenie zaakceptowane przez Ultralytics ('cpu', '0', '0,1', ...).
    """
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        if cuda_available:
            print(f"[device] Wykryto GPU: {torch.cuda.get_device_name(0)} -> uzywam '0'")
            return "0"
        print("[device] Brak GPU/CUDA -> fallback na CPU")
        return "cpu"

    if requested in ("cpu",):
        print("[device] Wymuszono CPU")
        return "cpu"

    # Uzytkownik zazadal GPU ('cuda', '0', '0,1', ...)
    if not cuda_available:
        print(f"[device] Zazadano '{requested}', ale CUDA niedostepne -> fallback na CPU")
        return "cpu"

    device = "0" if requested == "cuda" else requested
    print(f"[device] Uzywam GPU: '{device}'")
    return device


def train(args) -> str:
    """
    Uruchamia trening YOLO11 i zwraca sciezke do najlepszych wag.

    Returns:
        str: Sciezka do pliku best.pt.
    """
    device = resolve_device(args.device)

    # Project resolwujemy do sciezki absolutnej, aby wyniki trafialy do biezacego
    # repozytorium niezaleznie od globalnych ustawien Ultralytics (runs_dir),
    # ktore moga wskazywac na stara lokalizacje projektu.
    project = str(Path(args.project).resolve())

    print("\n" + "=" * 60)
    print(" TRENING MVP - YOLO11n")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Imgsz:   {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {device}")
    print(f"  Workers: {args.workers}")
    print(f"  Project: {project}")
    print("=" * 60 + "\n")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        plots=True,    # zapis wykresow do plikow (nie wyswietla okien)
        show=False,    # headless - bez podgladu GUI
        verbose=True,
    )

    # Lokalizacja najlepszych wag (Ultralytics zapisuje je w save_dir/weights/best.pt)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = os.path.join(project, args.name)
    best_path = os.path.join(str(save_dir), "weights", "best.pt")

    print("\nTrening zakonczony.")
    print(f"   - Wyniki:        {save_dir}")
    print(f"   - Najlepsze wagi: {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Trening MVP detektora grzybow (YOLO11n)."
    )
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Sciezka do wag bazowych (domyslnie: {DEFAULT_MODEL})")
    parser.add_argument("--data", "-d", default=DEFAULT_DATA,
                        help=f"Plik konfiguracyjny datasetu (domyslnie: {DEFAULT_DATA})")
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="Liczba epok (domyslnie: 30)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Rozmiar obrazu wejsciowego (domyslnie: 640)")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="Rozmiar batcha (domyslnie: 16)")
    parser.add_argument("--device", default="auto",
                        help="Urzadzenie: 'auto', 'cpu', 'cuda', '0', '0,1' "
                             "(domyslnie: auto z fallbackiem na CPU)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Liczba watkow data loadera (domyslnie: 8)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping - epoki bez poprawy (domyslnie: 15)")
    parser.add_argument("--project", default="runs/detect",
                        help="Katalog projektu na wyniki (domyslnie: runs/detect)")
    parser.add_argument("--name", default="train",
                        help="Nazwa eksperymentu (domyslnie: train)")
    parser.add_argument("--exist-ok", action="store_true",
                        help="Pozwol nadpisac istniejacy katalog eksperymentu")
    parser.add_argument("--resume", action="store_true",
                        help="Wznow przerwany trening")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
