# Mushroom Detection: mobilna identyfikacja grzybów

**Praca inżynierska, Politechnika Warszawska, Wydział Elektryczny**

Autonomiczny system detekcji i klasyfikacji **147 gatunków** grzybów na Androidzie:
offline, w czasie rzeczywistym, z priorytetem bezpieczeństwa mykologicznego.

| | |
|---|---|
| **Autor** | Kiril Horobets |
| **Promotor** | dr inż. Witold Czajewski |
| **Kierunek** | Informatyka Stosowana |
| **Rok** | 2025/2026 |
| **Repo** | [github.com/kirilhorobets/mushroom-recognition](https://github.com/kirilhorobets/mushroom-recognition) |

---

## Cel

Wspomóc zbieracza przy identyfikacji owocników rozłożonych w kadrze (stół / kosz / las),
ze szczególnym naciskiem na **gatunki bliźniacze** (jadalne vs śmiertelnie trujące).

Założenia twarde:

- detekcja wieloobiektowa (ramka + klasa na każdy owocnik)
- działanie **bez sieci** (edge AI)
- wysoka czułość dla klas toksycznych (asymetryczne progi UI)
- inferencja na strumieniu kamery

---

## Stack

| Warstwa | Technologie |
|---------|-------------|
| Trening | Python, PyTorch, Ultralytics **YOLO11m**, CUDA |
| Auto-anotacja | **GroundingDINO** (Autodistill), ontology `"mushroom"` |
| Dane | iNaturalist Research Grade → lokalny zbiór YOLO |
| Eksport | SavedModel → **TensorFlow Lite FP16** (~38,58 MB) |
| Android | Kotlin, Jetpack Compose, CameraX, Room, TFLite Interpreter |

**Model produkcyjny:** YOLO11m, `imgsz=640`, **147 klas**.  
Wariant Nano (YOLO11n) służy wyłącznie jako baseline ablacyjny (~48% mAP₅₀).

### Wyniki referencyjne (zbiór testowy)

| Wariant | mAP₅₀ | mAP₅₀–₉₅ | Rozmiar |
|---------|-------|----------|---------|
| YOLO11m (PyTorch) | 88,6% | 84,1% | ~38,8 MB |
| YOLO11m (TFLite FP16) | 88,5% | 84,0% | **38,58 MB** |
| YOLO11n (baseline) | 48,0% | 42,1% | ~5,6 MB |

Źródło: `docs/thesis_results/ablation_study.csv`.

### Progi bezpieczeństwa (Android)

| Sygnał UI | Próg | Zachowanie |
|-----------|------|------------|
| Ostrzeżenie toksyczne | `conf ≥ 0,18` | natychmiastowy status UNSAFE |
| Etykieta „Jadalny” | `conf ≥ 0,60` + głosowanie klatek | zielony status SAFE |
| Strefa niepewności | pomiędzy | NEUTRAL (bez zielonej etykiety) |

Implementacja: `MushroomRegistry.displayStatus` + `DetectionTracker` (vote 4/5).

---

## Struktura repozytorium

```
mushroom-detection/
├── android/                 # Aplikacja Kotlin (Compose + CameraX + Room + TFLite)
├── scripts/
│   ├── train.py             # Trening / fine-tune YOLO11
│   ├── gather_data.py       # Akwizycja iNaturalist (Research Grade)
│   ├── export_tflite.py     # Eksport PT → SavedModel → TFLite (FP16/FP32/INT8)
│   ├── auto_label.py        # Auto-anotacja GroundingDINO
│   ├── split_dataset.py     # Podział train/val/test
│   ├── utils/               # extract_frames, NMS cleaner, dataset tools
│   └── tests/               # Testy akademickie 1–5 → docs/thesis_results/
├── docs/thesis_results/     # CSV / JSON / wykresy z ewaluacji
├── dyploma/                 # Praca dyplomowa (LaTeX, XeLaTeX)
├── data/                    # Placeholdery lokalnych danych (treść poza gitem)
├── weights/                 # Lokalne wagi .pt (poza gitem)
├── requirements.txt
└── README.md
```

Duże artefakty (wagi `.pt`, modele `.tflite`, surowe dane, build Android/LaTeX)
są w `.gitignore` i nie trafiają do repozytorium.

---

## Potok danych (skrót)

```
iNaturalist (research grade)
        ↓
gather_data.py  (105 taksonów → cel ≥1000 zdjęć / klasa)
        ↓
auto_label.py  (GroundingDINO → YOLO labels)
        ↓
clean_duplicate_boxes_v2.py  (NMS / dedup)
        ↓
split_dataset.py  (train/val)
harvest_clean_test.py + label_clean_test.py  (test izolowany po obs_id)
        ↓
train.py  (YOLO11m, transfer learning)
        ↓
export_tflite.py  (--quant fp16)
        ↓
android/app/src/main/assets/*.tflite
```

---

## Quick start (Python)

### Wymagania

- Python 3.10+ (zalecane 3.11/3.12)
- (opcjonalnie) NVIDIA GPU + CUDA pod PyTorch
- TensorFlow potrzebny do `export_tflite.py`

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

PyTorch z CUDA: zainstaluj build zgodny z Twoim sterownikiem z
[pytorch.org](https://pytorch.org/get-started/locally/), potem dopiero `requirements.txt`.

### Trening

```bash
python scripts/train.py \
  --model weights/yolo11m.pt \
  --data path/to/data.yaml \
  --epochs 200 \
  --imgsz 640 \
  --batch 16 \
  --device auto
```

### Eksport TFLite FP16

```bash
python scripts/export_tflite.py \
  --weights path/to/best.pt \
  --quant fp16 \
  --imgsz 640
```

Skopiuj wynikowy `.tflite` do `android/app/src/main/assets/`
(nazwa musi zgadzać się z ścieżką w `MushroomDetector`).

### Testy akademickie

```bash
cd scripts/tests
python test1.py   # macierz bezpieczeństwa (asymetryczne progi)
python test2.py   # odporność na szum
python test3.py   # threshold sweep / F1
python test4.py   # latency breakdown
python test5.py   # podsumowanie ablacji
```

Domyślne ścieżki modelu/danych: `scripts/tests/_paths.py`.  
Wyniki: `docs/thesis_results/`.

### Auto-anotacja

```bash
python scripts/auto_label.py
```

Ścieżki bazowe ustawione są w skrypcie (lokalny dysk z danymi).

---

## Android

```bash
cd android
./gradlew assembleDebug
# Windows: gradlew.bat assembleDebug
```

- `minSdk 26`, `targetSdk 34`, Kotlin + Compose
- CameraX `ImageAnalysis` → center crop 640 → TFLite → NMS → tracker → overlay
- Historia znalezisk: Room + Coil

Wymagane lokalnie: plik modelu `.tflite` w `app/src/main/assets/`
oraz poprawny `local.properties` (SDK path; plik jest w `.gitignore`).

---

## Praca dyplomowa (LaTeX)

Źródła w `dyploma/`. Kompilacja:

```bash
cd dyploma
xelatex EE-dyplom.tex
biber EE-dyplom
xelatex EE-dyplom.tex
xelatex EE-dyplom.tex
```

---

## Disclaimer

Aplikacja ma charakter **edukacyjny / badawczy**.
**Nie zastępuje** ekspertyzy mykologa. Nie spożywaj grzybów wyłącznie na podstawie wyniku modelu.

---

## Licencja

Projekt w ramach pracy inżynierskiej na Politechnice Warszawskiej.  
© 2026 Kiril Horobets
