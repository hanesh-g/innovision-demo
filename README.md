# 👁 Security Platform — YOLOv11m + InsightFace

A full-stack unified security web application that performs real-time human detection and face recognition on uploaded videos and live webcam feeds. Built with Flask, OpenCV, Ultralytics YOLOv11m, and InsightFace.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square&logo=flask)
![YOLOv11](https://img.shields.io/badge/YOLO-v11m-green?style=flat-square)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?style=flat-square&logo=opencv)

---

## 🎯 Features

- 🎬 **Video & Webcam Support** — Process uploaded files (MP4, AVI, MOV) or live camera streams.
- 🔍 **YOLOv11m Person Detection** — High-accuracy human detection on every frame.
- 👤 **Advanced Face Recognition** — 4-stage pipeline (YOLO → SCRFD → 5-point Alignment → ArcFace) for robust identification.
- 🚨 **Multi-Tier Alert System**:
  - **Restricted Entry**: Alerts when a known blocklisted person is detected (🔴 Red).
  - **Intruder Detection**: Alerts when an unknown/unrecognized face is detected (🟣 Purple). Toggleable via UI.
  - **Zone Intrusion**: Draw custom polygonal zones; alerts when a person steps inside (🟠 Amber).
- 🛡️ **Enrollment Database** — SQLite-backed face enrollment system to register Authorized and Blocklisted individuals.
- 📡 **Real-time UI Updates** — Server-Sent Events (SSE) stream live alerts and processing progress to the dashboard.
- 📊 **CSV Export** — Detailed coordinate and recognition data for every frame.
- 🌐 **H264 Re-encoding** — Automatic browser-compatible video processing via ffmpeg.

---

## 🏗 Architecture & Pipeline

```text
1. Input Source (Webcam or Uploaded Video)
       ↓
2. YOLOv11m (Person Detection)
       ↓
3. InsightFace SCRFD (Face Detection & Landmarks)
       ↓
4. Geometric Alignment (5-point norm crop)
       ↓
5. InsightFace ArcFace (w600k_r50 Embedding Extraction)
       ↓
6. SQLite Database (Cosine Similarity Matching)
       ↓
7. Event Router (Triggers Restricted Entry, Intruder, or Zone Alerts)
       ↓
8. Server-Sent Events (SSE) → Flask Frontend
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Person Detection | YOLOv11m (Ultralytics) |
| Face Recognition | InsightFace (SCRFD + ArcFace buffalo_m) |
| Database | SQLite |
| Computer Vision | OpenCV (cv2) |
| Backend | Flask (with threading and SSE) |
| Frontend | Vanilla HTML / CSS / JS |

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+ (3.11 or 3.12 recommended)
- NVIDIA GPU with CUDA (optional but heavily recommended for performance)
- ffmpeg installed and added to system PATH

### 1. Clone the repository
```bash
git clone https://github.com/hanesh-g/innovision-demo.git
cd innovision-demo
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install PyTorch with CUDA
```bash
# Example for CUDA 11.8 (Check PyTorch website for your specific CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Install ffmpeg (Windows)
```bash
winget install --id Gyan.FFmpeg -e
```
*Ensure ffmpeg is in your system PATH.*

---

## 🚀 Running the App

```bash
# Activate venv
venv\Scripts\activate

# Start Flask server
python app.py
```

Server starts at `http://localhost:5000`.

> **Note:** On the very first run, the app will automatically download the YOLOv11m weights (~40MB) and the InsightFace `buffalo_m` model bundle (~330MB) into your home directory (`~/.insightface/models/`). This may take a few minutes.

---

## 📁 Project Structure

```text
innovision-demo/
├── app.py              # Flask backend, SSE streams, threading logic
├── pipeline.py         # 4-stage computer vision inference pipeline
├── database.py         # SQLite operations, face enrollment, alert logging
├── index.html          # Unified dashboard UI
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # SQLite database storage (auto-created)
├── uploads/            # Temp storage for input videos (auto-created)
├── outputs/            # Processed video outputs (auto-created)
└── coords/             # CSV coordinate exports (auto-created)
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
