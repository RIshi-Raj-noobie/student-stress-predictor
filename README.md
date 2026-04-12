# 🦺 Construction Site PPE Violation Detector

> **Capstone BYOP Project · AIML (2nd Year) · Computer Vision**

A real-time computer-vision system that detects **Personal Protective Equipment (PPE) violations** on construction sites — specifically workers missing helmets or safety vests — using **YOLOv8** object detection and **OpenCV**.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Example Outputs](#-example-outputs)
- [Evaluation Metrics](#-evaluation-metrics)
- [Dataset](#-dataset)
- [Future Improvements](#-future-improvements)
- [References](#-references)

---

## 🔍 Problem Statement

Construction sites are among the **most dangerous work environments** globally. The International Labour Organization (ILO) reports that the construction sector accounts for **~17% of all workplace fatalities**, many of which are caused by workers not wearing proper PPE (helmets, vests, gloves).

Manual monitoring by safety officers is costly, slow, and inconsistent. This project proposes an **automated real-time vision system** that:

1. Processes live camera feeds or recorded footage.
2. Detects workers and their PPE status (helmet / no helmet, vest / no vest).
3. Flags violations with bounding-box annotations and an on-screen alert.
4. Logs all violation events for audit and reporting.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎯 Real-time detection | YOLOv8n at ≥15 FPS on CPU |
| 📷 Multi-input support | Image, video file, or live webcam |
| 🔴 Violation alerts | Red bounding box + "⚠ VIOLATION" label |
| 📊 HUD overlay | Live FPS, person count, violation count |
| 🖥️ Streamlit dashboard | Upload & inspect images/videos in-browser |
| 📝 Violation logging | Timestamped JSON log per session |
| 📈 Evaluation module | mAP@0.5, Precision, Recall, F1 |
| 🔁 Demo mode | Runs without GPU via OpenCV synthetic detections |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV 4.x |
| Numerical Computing | NumPy |
| Deep Learning Backend | PyTorch (via Ultralytics) |
| Dashboard / UI | Streamlit |
| Data Visualisation | Matplotlib, Seaborn |
| Language | Python 3.9 – 3.11 |

---

## 📁 Project Structure

```
construction-safety-cv/
│
├── src/
│   ├── detect.py           ← Main detection pipeline (image + video)
│   ├── train.py            ← YOLOv8 fine-tuning script
│   ├── evaluate.py         ← mAP, Precision, Recall, F1, IoU metrics
│   ├── app.py              ← Streamlit dashboard
│   └── download_data.py    ← Dataset download / setup helper
│
├── data/
│   ├── raw/                ← Original dataset (images + YOLO labels)
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   ├── processed/          ← Augmented / pre-processed frames
│   └── annotations/        ← Any additional annotation files
│
├── models/
│   ├── pretrained/         ← Downloaded or trained .pt weights
│   └── trained/            ← Output from train.py (runs/)
│
├── notebooks/
│   └── EDA_and_Baseline.ipynb   ← Exploratory data analysis
│
├── outputs/
│   ├── detections/         ← Annotated output images
│   ├── videos/             ← Annotated output videos
│   └── reports/            ← Evaluation JSON / CSV reports
│
├── docs/
│   └── project_report.md   ← Full academic project report
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.9 – 3.11**
- `pip` (bundled with Python)
- *(Optional)* NVIDIA GPU with CUDA 11.8+ for faster inference

### Step 1 – Clone the repository

```bash
git clone https://github.com/<your-username>/construction-safety-cv.git
cd construction-safety-cv
```

### Step 2 – Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` auto-installs PyTorch. If you have a GPU, install the CUDA-enabled torch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4 – Download the dataset (optional, for training)

```bash
python src/download_data.py --dest data/raw
```

This creates the directory structure and `data.yaml`. Download the actual images from the [Roboflow dataset link](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) and place them in `data/raw/`.

---

## 🚀 Usage

### 1. Run detection on an image

```bash
python src/detect.py --source path/to/image.jpg --output outputs/detections
```

### 2. Run detection on a video

```bash
python src/detect.py --source path/to/video.mp4 --output outputs/videos --display
```

### 3. Run detection on live webcam

```bash
python src/detect.py --source 0 --display
```

### 4. Use a custom trained model

```bash
python src/detect.py --source path/to/image.jpg --model models/pretrained/ppe_yolov8n.pt
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run src/app.py
```

Then open `http://localhost:8501` in your browser.

### 6. Fine-tune YOLOv8 on your data

```bash
python src/train.py --data data/raw/data.yaml --epochs 50 --batch 16 --device cpu
```

### 7. Evaluate model performance

```bash
python src/evaluate.py
```

---

## 🖼️ Example Outputs

### Image Detection

| Input | Output |
|---|---|
| Raw construction site photo | Annotated with green boxes (PPE OK) and red boxes (violations) |
| Workers without helmets | Red bounding box with "No-Helmet ⚠ VIOLATION" label |

### Video / Webcam

- Live HUD showing: **Status (SAFE / UNSAFE)**, persons detected, violation count, FPS
- Annotated output video saved to `outputs/videos/`

### Dashboard

- Upload image or video in browser
- Side-by-side before/after view
- Metric cards: persons, violations, confidence scores
- Per-detection table with violation flags

---

## 📈 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| mAP@0.5 | Mean Average Precision at IoU=0.5 | > 0.70 |
| Precision | TP / (TP + FP) | > 0.75 |
| Recall | TP / (TP + FN) | > 0.70 |
| F1-Score | Harmonic mean of P & R | > 0.72 |
| FPS | Frames per second (CPU) | ≥ 15 |

*Metrics are computed on the validation split using `src/evaluate.py`.*

---

## 📦 Dataset

| Dataset | Source | Images | Format |
|---|---|---|---|
| Construction Site Safety | [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) | ~2,600 | YOLOv8 |
| PPE Detection | [Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) | ~3,000 | YOLO |
| COCO (persons) | [COCO](https://cocodataset.org) | 118k+ | COCO JSON |

Classes: `Hardhat · Mask · NO-Hardhat · NO-Mask · NO-Safety Vest · Person · Safety Cone · Safety Vest · machinery · vehicle`

---

## 🔮 Future Improvements

- **Multi-camera support** — Monitor multiple CCTV feeds simultaneously
- **Alert system** — Send SMS/email notifications on violation detection
- **Zone-based rules** — Define restricted areas requiring full PPE
- **Tracking** — Use ByteTrack/DeepSORT to track workers across frames
- **Edge deployment** — Export to ONNX / TFLite for Raspberry Pi or Jetson
- **Larger model** — Switch from YOLOv8n → YOLOv8m for higher accuracy
- **Glove / boot detection** — Extend to full 7-class PPE coverage

---

## 📚 References

1. Jocher, G. et al. *YOLOv8* (2023). Ultralytics. https://github.com/ultralytics/ultralytics
2. Redmon, J. & Farhadi, A. *YOLOv3: An Incremental Improvement* (2018). arXiv:1804.02767
3. Bradski, G. *The OpenCV Library*. Dr. Dobb's Journal (2000).
4. Lin, T. et al. *Microsoft COCO: Common Objects in Context* (2014). ECCV.
5. ILO. *Safety and Health at Work*. https://www.ilo.org/global/topics/safety-and-health-at-work
6. Roboflow. *Construction Site Safety Dataset*. https://universe.roboflow.com

---

## 🧑‍💻 Author

Rishiraj Singh Tomar · B.Tech AIML · Vit Bhopal University
Roll No:23BAI10849· Academic Year: 2025-26

---

*This project was developed as part of the Bring Your Own Project (BYOP) Capstone for the Artificial Intelligence & Machine Learning programme.*
