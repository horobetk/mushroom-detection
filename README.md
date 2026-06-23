# 🍄 Mushroom Recognition Mobile App

**Praca inżynierska - Politechnika Warszawska**

System rozpoznawania grzybów na urządzeniach mobilnych z wykorzystaniem głębokich sieci neuronowych

---

##  Informacje o projekcie

**Autor:** Kiril Horobets  
**Promotor:** dr inż. Witold Czajewski  
**Uczelnia:** Politechnika Warszawska, Wydział Elektryczny  
**Rok akademicki:** 2024/2025

---

##  Cel pracy

Opracowanie aplikacji mobilnej Android wspomagającej identyfikację grzybów niejadalnych i trujących wśród zebranych grzybów rozłożonych na stole. 

**Główne założenia:**
- ✅ Detekcja wielu grzybów jednocześnie (multi-object detection)
- ✅ Priorytet: **bezpieczeństwo** - wysoki recall dla gatunków trujących
- ✅ Działanie **offline** na urządzeniu mobilnym
- ✅ Real-time inference

---

##  Architektura techniczna

### Stack technologiczny

```
┌─────────────────────────────────────┐
│   Training Pipeline (Python)        │
├─────────────────────────────────────┤
│  • PyTorch                          │
│  • YOLO11 Nano (Ultralytics)        │
│  • OpenCV (preprocessing)           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Model Conversion                  │
├─────────────────────────────────────┤
│  PyTorch → ONNX → TensorFlow Lite   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Mobile App (Android)              │
├─────────────────────────────────────┤
│  • Kotlin                           │
│  • TensorFlow Lite                  │
│  • CameraX API                      │
└─────────────────────────────────────┘
```

### Wybór modelu: YOLO11 Nano

**Dlaczego YOLO11 Nano?**
-  Szybkość: ~50 FPS na urządzeniach mobilnych
-  Rozmiar: ~6MB (możliwość offline deployment)
-  Accuracy: lepsza niż YOLOv8n przy mniejszej liczbie parametrów (~2.6M)
-  Łatwość treningu i konwersji do TFLite

**Alternatywy rozważane:**
- YOLOv8n (poprzednia generacja, zastąpiona przez YOLO11n)
- YOLO11s/m (za duże dla mobile)
- MobileNet SSD (niższa accuracy)
- EfficientDet (wolniejszy inference)

---

##  Dataset

### Źródła danych
1. **Nagrania wideo** - dostarczone przez promotora
   - Ekstrakcja klatek co N-tą
   - Manual annotation (bounding boxes)
   
2. **Publiczne datasety:**
   - iNaturalist Fungi dataset
   - Kaggle Mushroom datasets

### Preprocessing pipeline
```python
# Pseudo-code
video → extract_frames(every_n=30) → 
resize(640x640) → augmentation → 
annotation(YOLO_format) → train/val/test split
```

**Target:** 1500-2000 annotated images

---

##  Struktura projektu

```
mushroom-recognition/
│
├── data/                      # Datasets
│   ├── raw/                   # Raw video files
│   ├── frames/                # Extracted frames
│   ├── annotations/           # YOLO format labels
│   └── processed/             # Train/val/test splits
│
├── training/                  # Model training
│   ├── train.py               # Training script
│   ├── config.yaml            # YOLO11 config
│   ├── evaluate.py            # Evaluation metrics
│   └── convert_to_tflite.py  # Model conversion
│
├── android/                   # Android app
│   ├── app/
│   │   ├── src/
│   │   ├── models/            # .tflite models
│   │   └── build.gradle
│   └── README.md
│
├── scripts/                   # Utility scripts
│   ├── extract_frames.py      # Video → frames
│   ├── augmentation.py        # Data augmentation
│   └── visualize.py           # Visualization tools
│
├── docs/                      # Documentation
│   └── thesis/                # LaTeX thesis files
│
├── requirements.txt           # Python dependencies
└── README.md
```

---

##  Quick Start

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Extract frames from video
```bash
python scripts/extract_frames.py \
    --input data/raw/video.mp4 \
    --output data/frames/ \
    --frame_rate 30
```

### Train YOLO11 model
```bash
python scripts/train_mvp.py \
    --data mushrooms_mvp.yaml \
    --model weights/yolo11n.pt \
    --epochs 30 \
    --imgsz 640
```

### Convert to TensorFlow Lite
```bash
python training/convert_to_tflite.py \
    --weights runs/train/exp/weights/best.pt \
    --output models/mushroom_detector.tflite
```

---

##  Metryki i ewaluacja

**Kluczowe metryki:**
- **Recall** (dla gatunków trujących) - **PRIORYTET**
- Precision
- mAP@0.5
- Inference time (ms)
- Model size (MB)

**Safety-first approach:**  
Lepiej oznaczyć jadalny grzyb jako niebezpieczny (false positive) niż pominąć trujący grzyb (false negative).

---

##  Przegląd literatury

Kluczowe publikacje wykorzystane w projekcie:

1. **YOLO Series:**
   - Redmon et al. - YOLOv3 (2018)
   - Jocher - YOLOv5, YOLOv8, YOLO11 (Ultralytics)

2. **Mobile ML:**
   - Howard et al. - MobileNets (2017)
   - Sandler et al. - MobileNetV2 (2018)

3. **Object Detection:**
   - Lin et al. - Focal Loss, RetinaNet (2017)
   - Tan & Le - EfficientDet (2020)

4. **Datasets:**
   - Van Horn et al. - iNaturalist Dataset (2018)

*Pełna bibliografia dostępna w pracy dyplomowej.*

---

## 🛠️ Technologies Used

**Training:**
- ![Python](https://img.shields.io/badge/Python-3.8+-blue)
- ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
- ![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-green)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue)

**Mobile:**
- ![Kotlin](https://img.shields.io/badge/Kotlin-1.9-purple)
- ![TFLite](https://img.shields.io/badge/TensorFlow_Lite-2.x-orange)
- ![Android](https://img.shields.io/badge/Android-API_24+-green)

---

##  Harmonogram prac

- ✅ **Styczeń 2026:** Koncepcja, architektura, przegląd literatury
- 🔄 **Luty 2026:** Ekstrakcja danych, anotacja, baseline model
- ⏳ **Marzec 2026:** Trening modelu, optymalizacja hyperparameters
- ⏳ **Kwiecień 2026:** Implementacja aplikacji Android
- ⏳ **Maj 2026:** Testy, ewaluacja, analiza wyników
- ⏳ **Czerwiec 2026:** Finalizacja pracy dyplomowej

---

##  Disclaimer

**Uwaga:** Aplikacja ma charakter edukacyjny i nie zastępuje konsultacji z ekspertem mykoologicznym. Nie należy spożywać grzybów wyłącznie na podstawie automatycznej identyfikacji.

---

##  Kontakt

**Kiril Horobets**  
Politechnika Warszawska  
Email: kiril.horobets@pw.edu.pl  

**Promotor:**  
dr inż. Witold Czajewski  
Politechnika Warszawska, Wydział Elektryczny

---

## 📄 Licencja

Projekt wykonany w ramach pracy inżynierskiej na Politechnice Warszawskiej.  
© 2026 Kiril Horobets
