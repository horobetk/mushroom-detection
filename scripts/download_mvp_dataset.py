#!/usr/bin/env python3
"""
Pobieranie i przygotowanie szybkiego datasetu MVP dla detektora grzybow (YOLO11).

Pipeline:
  1. Pobranie publicznego datasetu klasyfikacyjnego grzybow przez Kaggle API
     (obrazy ulozone w folderach wg nazwy klasy).
  2. Wybor klas i ograniczenie liczby obrazow na klase (domyslnie 50).
  3. Podzial na train / val.
  4. Konwersja klasyfikacji -> format detekcji YOLO: dla kazdego obrazu tworzony
     jest plik .txt z jednym, "mockowym" bounding boxem pokrywajacym centralne
     80% obrazu. To rozwiazanie tymczasowe na potrzeby MVP, zanim wprowadzimy
     pelna autorozmiar (auto-labeling).
  5. Wygenerowanie pliku konfiguracyjnego *.yaml gotowego do treningu.

UWAGA - Kaggle API (dane uwierzytelniajace):
  Skrypt NIE zawiera zadnych kluczy w kodzie. Dane logowania pobierane sa ze
  standardowych zrodel biblioteki Kaggle:
    - plik  ~/.kaggle/kaggle.json   (Windows: C:\\Users\\<user>\\.kaggle\\kaggle.json)
    - lub zmienne srodowiskowe: KAGGLE_USERNAME oraz KAGGLE_KEY
  Token wygenerujesz na stronie: https://www.kaggle.com/settings -> "Create New Token".

Autor: Kiril Horobets
Politechnika Warszawska, 2026
Praca inzynierska: System rozpoznawania grzybow na urzadzeniach mobilnych
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

# Wymuszenie UTF-8 na konsoli (np. Windows cp1252 + sciezki z polskimi/cyrylickimi znakami)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from tqdm import tqdm

# Domyslny publiczny dataset klasyfikacyjny grzybow (obrazy w folderach wg rodzaju).
# Mozna podmienic dowolnym datasetem o strukturze: <klasa>/<obraz>.jpg
DEFAULT_DATASET = "maysee/mushrooms-classification-common-genuss-images"

# Rozszerzenia traktowane jako obrazy
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Domyslne sciezki
DEFAULT_RAW_DIR = "data/external/kaggle_mushrooms"
DEFAULT_OUTPUT_DIR = "datasets/mushrooms_mvp"
DEFAULT_YAML = "datasets/mushrooms_mvp/mushrooms_mvp.yaml"


def check_kaggle_credentials() -> bool:
    """
    Sprawdza, czy dostepne sa dane logowania do Kaggle API.

    Kolejnosc poszukiwania:
      1. zmienne srodowiskowe KAGGLE_USERNAME + KAGGLE_KEY,
      2. standardowy plik ~/.kaggle/kaggle.json,
      3. plik kaggle.json w katalogu wskazanym przez KAGGLE_CONFIG_DIR,
      4. plik kaggle.json w biezacym katalogu repozytorium (wygodne lokalnie).

    Jesli kaggle.json zostanie znaleziony lokalnie, ustawiamy KAGGLE_CONFIG_DIR,
    aby biblioteka kaggle uzyla wlasciwej lokalizacji.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True

    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir and (Path(config_dir) / "kaggle.json").exists():
        return True

    local_json = Path.cwd() / "kaggle.json"
    if local_json.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(local_json.parent)
        print(f"[kaggle] Uzywam lokalnego pliku: {local_json}")
        return True

    return False


def print_credentials_help():
    """Wyswietla instrukcje konfiguracji Kaggle API."""
    print("\n[BLAD] Brak danych uwierzytelniajacych Kaggle API.\n")
    print("Skonfiguruj dostep na jeden z dwoch sposobow:")
    print("  1) Plik kaggle.json:")
    print("     - Wejdz na https://www.kaggle.com/settings")
    print("     - 'Create New Token' -> pobierze sie kaggle.json")
    print(r"     - Umiesc go w:  C:\Users\<user>\.kaggle\kaggle.json")
    print("  2) Zmienne srodowiskowe:")
    print("     setx KAGGLE_USERNAME twoj_login")
    print("     setx KAGGLE_KEY twoj_klucz_api")
    print("\nPo konfiguracji uruchom skrypt ponownie.\n")


def download_kaggle_dataset(dataset_slug: str, raw_dir: Path) -> Path:
    """
    Pobiera i rozpakowuje dataset z Kaggle.

    Args:
        dataset_slug (str): Identyfikator datasetu, np. 'autor/nazwa-datasetu'.
        raw_dir (Path): Katalog docelowy na surowe dane.

    Returns:
        Path: Katalog z rozpakowanymi danymi.
    """
    # Import lokalny - biblioteka kaggle probuje sie uwierzytelnic przy imporcie,
    # dlatego importujemy ja dopiero po sprawdzeniu danych logowania.
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pobieranie datasetu z Kaggle: {dataset_slug}")
    print(f"   - Katalog docelowy: {raw_dir.resolve()}")

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(raw_dir), unzip=True, quiet=False)

    print("Pobieranie zakonczone.")
    return raw_dir


def find_class_folders(raw_dir: Path, min_images: int = 1) -> dict:
    """
    Wyszukuje foldery klas (katalogi zawierajace bezposrednio obrazy).

    Nazwa klasy = nazwa folderu, w ktorym leza obrazy. Dziala takze dla
    struktur zagniezdzonych (np. train/<klasa>/img.jpg).

    Args:
        raw_dir (Path): Katalog z rozpakowanym datasetem.
        min_images (int): Minimalna liczba obrazow, aby uznac folder za klase.

    Returns:
        dict: {nazwa_klasy: [sciezki_do_obrazow]}
    """
    classes = {}

    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            class_name = path.parent.name
            classes.setdefault(class_name, []).append(path)

    # Odfiltruj foldery z mala liczba obrazow
    classes = {k: v for k, v in classes.items() if len(v) >= min_images}
    return classes


def filter_classes(all_classes: dict, wanted: list) -> dict:
    """
    Filtruje wykryte klasy wg listy zadanej przez uzytkownika (dopasowanie
    bez rozrozniania wielkosci liter, czesciowe).
    """
    if not wanted:
        return all_classes

    selected = {}
    for want in wanted:
        want_lower = want.strip().lower()
        for class_name, images in all_classes.items():
            if want_lower in class_name.lower():
                selected[class_name] = images

    return selected


def write_mock_label(label_path: Path, class_id: int, coverage: float):
    """
    Zapisuje plik etykiety w formacie YOLO z jednym mockowym bounding boxem
    pokrywajacym centralne `coverage` (np. 0.8 = 80%) obrazu.

    Format YOLO (znormalizowany): <class_id> <x_center> <y_center> <width> <height>
    Box wycentrowany: x_center = y_center = 0.5.
    """
    line = f"{class_id} 0.5 0.5 {coverage:.6f} {coverage:.6f}\n"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(line)


def build_yolo_dataset(
    class_map: dict,
    output_dir: Path,
    per_class: int,
    val_split: float,
    coverage: float,
    seed: int,
) -> dict:
    """
    Buduje dataset detekcyjny YOLO z mapowania klas (klasyfikacja -> detekcja).

    Args:
        class_map (dict): {nazwa_klasy: [sciezki_obrazow]}.
        output_dir (Path): Katalog wyjsciowy datasetu.
        per_class (int): Maks. liczba obrazow na klase.
        val_split (float): Udzial zbioru walidacyjnego (0.0 - 1.0).
        coverage (float): Pokrycie mockowego boxa (0.0 - 1.0).
        seed (int): Ziarno losowosci dla powtarzalnosci.

    Returns:
        dict: Statystyki budowy datasetu.
    """
    random.seed(seed)

    # Stabilne, posortowane nazwy klas -> deterministyczne ID
    class_names = sorted(class_map.keys())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Utworzenie struktury katalogow
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"classes": {}, "train": 0, "val": 0, "total": 0}

    for class_name in class_names:
        class_id = class_to_id[class_name]
        images = list(class_map[class_name])
        random.shuffle(images)
        images = images[:per_class]

        if not images:
            continue

        split_idx = int(len(images) * (1.0 - val_split))
        # Gwarantujemy min. 1 obraz w val, jesli klasa ma >= 2 obrazy
        if val_split > 0 and split_idx >= len(images) and len(images) >= 2:
            split_idx = len(images) - 1

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for split, split_images in [("train", train_imgs), ("val", val_imgs)]:
            for i, img_path in enumerate(
                tqdm(split_images, desc=f"{class_name} [{split}]", unit="img", leave=False)
            ):
                # Bezpieczna, unikalna nazwa pliku
                safe_class = class_name.replace(" ", "_").replace("/", "_")
                stem = f"{safe_class}_{class_id:02d}_{split}_{i:04d}"

                img_dst = output_dir / "images" / split / f"{stem}{img_path.suffix.lower()}"
                lbl_dst = output_dir / "labels" / split / f"{stem}.txt"

                shutil.copy(img_path, img_dst)
                write_mock_label(lbl_dst, class_id, coverage)

        stats["classes"][class_name] = {
            "id": class_id,
            "train": len(train_imgs),
            "val": len(val_imgs),
        }
        stats["train"] += len(train_imgs)
        stats["val"] += len(val_imgs)

    stats["total"] = stats["train"] + stats["val"]
    stats["names"] = class_names
    return stats


def write_yaml(output_dir: Path, yaml_path: Path, class_names: list):
    """Tworzy plik konfiguracyjny YAML dla treningu YOLO."""
    names_str = ", ".join(f"'{n}'" for n in class_names)
    content = (
        f"# Dataset MVP grzybow - format detekcji YOLO (mockowe bounding boxy)\n"
        f"# Wygenerowano automatycznie przez scripts/download_mvp_dataset.py\n\n"
        f"train: {output_dir.absolute().as_posix()}/images/train\n"
        f"val: {output_dir.absolute().as_posix()}/images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_str}]\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def print_summary(stats: dict, yaml_path: Path):
    """Wyswietla podsumowanie zbudowanego datasetu."""
    print("\n" + "=" * 60)
    print(" PODSUMOWANIE DATASETU MVP")
    print("=" * 60)
    for class_name, info in stats["classes"].items():
        print(
            f"  [{info['id']:>2}] {class_name:<28} "
            f"train={info['train']:>3}  val={info['val']:>3}"
        )
    print("-" * 60)
    print(f"  Klasy:  {len(stats['classes'])}")
    print(f"  Train:  {stats['train']}")
    print(f"  Val:    {stats['val']}")
    print(f"  RAZEM:  {stats['total']} obrazow")
    print(f"  Config: {yaml_path}")
    print("=" * 60)
    print("\nTrening MVP uruchomisz np. komenda:")
    print(f"  yolo detect train model=weights/yolo11n.pt data={yaml_path} epochs=50 imgsz=640\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pobieranie i przygotowanie datasetu MVP grzybow w formacie YOLO."
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help=f"Slug datasetu Kaggle (domyslnie: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--classes", "-c", nargs="*", default=None,
        help="Lista klas do uwzglednienia (dopasowanie czesciowe). "
             "Domyslnie: wszystkie wykryte klasy.",
    )
    parser.add_argument(
        "--per-class", "-n", type=int, default=50,
        help="Maks. liczba obrazow na klase (domyslnie: 50)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Udzial zbioru walidacyjnego, 0.0-1.0 (domyslnie: 0.2)",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.8,
        help="Pokrycie mockowego bounding boxa, 0.0-1.0 (domyslnie: 0.8 = 80%%)",
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR,
        help=f"Katalog na surowe dane z Kaggle (domyslnie: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT_DIR,
        help=f"Katalog wyjsciowy datasetu YOLO (domyslnie: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--yaml", default=DEFAULT_YAML,
        help=f"Sciezka pliku konfiguracyjnego YAML (domyslnie: {DEFAULT_YAML})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Ziarno losowosci dla powtarzalnego podzialu (domyslnie: 42)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Pomin pobieranie z Kaggle (uzyj juz pobranych danych w --raw-dir).",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output)
    yaml_path = Path(args.yaml)

    # 1. Pobranie danych z Kaggle
    if not args.skip_download:
        if not check_kaggle_credentials():
            print_credentials_help()
            sys.exit(1)
        try:
            download_kaggle_dataset(args.dataset, raw_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[BLAD] Pobieranie z Kaggle nie powiodlo sie: {exc}")
            print("Sprawdz nazwe datasetu oraz dane logowania Kaggle API.")
            sys.exit(1)
    else:
        print(f"Pomijam pobieranie. Uzywam danych z: {raw_dir.resolve()}")

    if not raw_dir.exists():
        print(f"[BLAD] Katalog z danymi nie istnieje: {raw_dir}")
        sys.exit(1)

    # 2. Wykrycie klas
    print("\nWyszukiwanie folderow klas...")
    all_classes = find_class_folders(raw_dir)
    if not all_classes:
        print("[BLAD] Nie znaleziono zadnych obrazow w folderach klas.")
        sys.exit(1)

    print(f"Wykryto {len(all_classes)} potencjalnych klas:")
    for name, imgs in sorted(all_classes.items()):
        print(f"   - {name}: {len(imgs)} obrazow")

    # 3. Filtrowanie klas
    selected = filter_classes(all_classes, args.classes)
    if not selected:
        print("\n[BLAD] Po filtrowaniu nie pozostala zadna klasa. "
              "Sprawdz argument --classes.")
        sys.exit(1)

    if args.classes:
        print(f"\nWybrano {len(selected)} klas wg filtra: {list(selected.keys())}")

    # 4. + 5. Budowa datasetu YOLO i zapis YAML
    print("\nBudowanie datasetu w formacie detekcji YOLO...")
    stats = build_yolo_dataset(
        class_map=selected,
        output_dir=output_dir,
        per_class=args.per_class,
        val_split=args.val_split,
        coverage=args.coverage,
        seed=args.seed,
    )

    write_yaml(output_dir, yaml_path, stats["names"])
    print_summary(stats, yaml_path)


if __name__ == "__main__":
    main()
