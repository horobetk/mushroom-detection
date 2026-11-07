# Mushroom Detection: mobilna identyfikacja grzybów

**Praca inżynierska, Politechnika Warszawska, Wydział Elektryczny**

Autonomiczny system detekcji i klasyfikacji **147 klas** grzybów (145 odrębnych gatunków)
na Androidzie: offline, z interaktywnym odczytem z podglądu kamery (poniżej 1 s),
z priorytetem bezpieczeństwa mykologicznego.

| | |
|---|---|
| **Autor** | Kiril Horobets |
| **Promotor** | dr inż. Witold Czajewski |
| **Kierunek** | Informatyka Stosowana |
| **Rok** | 2025/2026 |
| **Repo** | [github.com/horobetk/mushroom-detection](https://github.com/horobetk/mushroom-detection) |

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
Wariant Nano (YOLO11n) służy wyłącznie jako baseline ablacyjny.

### Wyniki referencyjne (izolowany zbiór testowy, 32 727 obrazów)

| Wariant | mAP₅₀ | mAP₅₀–₉₅ | P | R |
|---------|-------|----------|---|---|
| YOLO11m (PyTorch) | 82,75% | 78,23% | 80,90% | 75,32% |
| YOLO11n (baseline) | 48,54% | 42,82% | 51,49% | 48,71% |

Artefakt TFLite FP16 (38,58 MB) porównano z PyTorch na losowej próbie 2 500 obrazów
testowych: mAP₅₀ 79,14% wobec 84,67% (`docs/thesis_results/tflite_parity.json`).

### Progi bezpieczeństwa (Android)

| Sygnał UI | Warunek | Zachowanie |
|-----------|---------|------------|
| Ostrzeżenie toksyczne | klasa trująca / śmiertelnie trująca, `conf ≥ 0,18` | natychmiastowe ostrzeżenie |
| Etykieta „Jadalny” | klasa jadalna, `conf ≥ 0,60` i zgodność w 4 z 5 klatek | zielony status |
| Pozostałe przypadki | m.in. klasy niejadalne, `conf ≥ 0,20` | szary status neutralny |

Implementacja: `MushroomRegistry.displayStatus` + `DetectionTracker`.

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
│   ├── split_dataset.py     # Podział korpusu uczącego train/val
│   ├── harvest_clean_test.py, label_clean_test.py, clean_test_split.py
│   │                        # Izolowany zbiór testowy (nowe obs_id)
│   ├── analyze_field_research.py  # Metryki z Trybu Badawczego (teren)
│   ├── utils/               # Czyszczenie duplikatów ramek i narzędzia pomocnicze
│   └── tests/               # Skrypty ewaluacji → docs/thesis_results/
├── docs/thesis_results/     # CSV / JSON / wykresy z ewaluacji
├── dyploma/                 # Praca dyplomowa (LaTeX, XeLaTeX)
├── research_data/           # Anotacja i wyniki ewaluacji terenowej
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
harvest_clean_test.py + label_clean_test.py + clean_test_split.py  (test izolowany po obs_id)
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
python test1_edibility_safety.py   # macierz bezpieczeństwa (asymetryczne progi)
python test2_robustness_noise.py   # odporność na szum
python test3_threshold_sweep.py    # threshold sweep / F1
python test4_latency_breakdown.py  # latency breakdown
python test5_ablation_summary.py   # podsumowanie ablacji
python test_h2_baseline.py         # CFPR H2 (PyTorch)
python test_tflite_parity.py       # mAP .pt vs TFLite FP16
python test_tflite_cfpr.py         # CFPR H2 na TFLite FP16 (dekoder jak w aplikacji)
python test_category_map.py        # mAP / P / R po kategoriach toksykologicznych
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
- CameraX `ImageAnalysis` (4 FPS) → letterbox 640×640 → TFLite FP16 → NMS → tracker → nakładka
- Historia znalezisk: Room + Coil

Wymagane lokalnie: plik modelu `best_float16.tflite` w `app/src/main/assets/`
(udostępniony w zakładce Releases razem z APK) oraz `local.properties` ze ścieżką SDK.

Pomiar opóźnień na telefonie: kompilacja debug zapisuje `inference_profile.csv`
(czas każdego etapu klatki) i `response_profile.csv` (czas do pierwszej ramki
i do zielonej etykiety) w `Android/data/com.pw.mushroom/files/`.

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
