# 👁 HumanDetect — YOLOv11m Human Detection Web App

A full-stack web application that performs real-time human detection on uploaded videos using **YOLOv11m**. Built with Flask, OpenCV, and Ultralytics.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square&logo=flask)
![YOLOv11](https://img.shields.io/badge/YOLO-v11m-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?style=flat-square&logo=opencv)
![CUDA](https://img.shields.io/badge/CUDA-12.6%2B-76B900?style=flat-square&logo=nvidia)

---

## 🎯 Features

- 🎬 **Upload any video** — MP4, AVI, MOV, MKV
- 🔍 **YOLOv11m inference** on every single frame
- 👤 **Person class only** — filters COCO class 0
- 🟩 **Bounding boxes** drawn with confidence scores
- 📊 **CSV export** — every detection with frame, coordinates, confidence
- 🎞 **Original FPS & resolution preserved**
- 🌐 **H264 re-encoding** via ffmpeg for browser playback
- 🖥 **Side-by-side comparison** — original vs annotated
- 📋 **Job history** — track all processed videos in a session

---

## 🖼 UI Preview

| Panel | Description |
|-------|-------------|
| Run Detection | Upload, configure parameters, run inference |
| Results | Side-by-side original vs annotated video |
| Job History | All jobs this session with CSV download |
| How It Works | Full pipeline and architecture explanation |

---

## 🏗 Architecture

```
Browser (index.html)
       │
       │  HTTP POST /process (multipart video)
       ▼
Flask Server (app.py : localhost:5000)
       │
       ├── OpenCV VideoCapture → frame extraction
       ├── YOLOv11m inference (classes=[0]) per frame
       ├── Draw bounding boxes + HUD overlay
       ├── Log coordinates → CSV
       ├── OpenCV VideoWriter → raw mp4v
       ├── ffmpeg → H264 re-encode (browser compatible)
       └── make_response(video_bytes) → browser
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection Model | YOLOv11m (Ultralytics) |
| Deep Learning | PyTorch + CUDA |
| Computer Vision | OpenCV (cv2) |
| Backend | Flask + flask-cors |
| Video Encoding | ffmpeg libx264 |
| Frontend | Vanilla HTML / CSS / JS |
| Data Export | Python csv module |

---

## ⚙️ Installation

### Prerequisites
- Python 3.11 or 3.12 (recommended)
- NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- ffmpeg installed and added to PATH
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/human-detection-yolov11m.git
cd human-detection-yolov11m
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install PyTorch with CUDA
```bash
# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Install ffmpeg (Windows)
```bash
winget install --id Gyan.FFmpeg -e
```
Then add ffmpeg to your system PATH.

---

## 🚀 Running the App

```bash
# Activate venv
venv\Scripts\activate

# Start Flask server
python app.py
```

Server starts at `http://localhost:5000`

Open `index.html` in Chrome or Edge.

---

## 📁 Project Structure

```
human-detection-yolov11m/
├── app.py              # Flask backend — inference pipeline
├── index.html          # Frontend web UI
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .gitignore          # Excludes venv, model weights, outputs
├── uploads/            # Temp storage for input videos (auto-created)
├── outputs/            # Processed video outputs (auto-created)
└── coords/             # CSV coordinate exports (auto-created)
```

> **Note:** `yolo11m.pt` (~40MB) downloads automatically on first run via Ultralytics.

---

## 📊 CSV Output Format

Each processed video generates a CSV with one row per detection:

| Column | Description |
|--------|-------------|
| frame | Frame number |
| person_id | Detection index within that frame |
| x1, y1 | Top-left corner of bounding box (pixels) |
| x2, y2 | Bottom-right corner of bounding box (pixels) |
| width_px | Bounding box width in pixels |
| height_px | Bounding box height in pixels |
| confidence | Model confidence score (0–1) |
| frame_width | Original video width |
| frame_height | Original video height |

---

## ⚡ Performance

| GPU | Inference Speed |
|-----|----------------|
| RTX 4090 | ~150+ FPS |
| RTX 3080 / 4080 | ~80–120 FPS |
| RTX 3060 / 4060 | ~30–60 FPS |
| CPU only | ~1–5 FPS |

---

## 📄 License

MIT License — free to use, modify, and distribute.
