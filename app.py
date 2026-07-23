"""
Integrated YOLOv11m Security Platform
- YOLOv11m Human Detection
- InsightFace Face Recognition
- Real-time SSE streaming for live video preview and alerts
- Webcam MJPEG streaming
"""

import cv2
import json
import uuid
import csv
import io
import os
import subprocess
import tempfile
import numpy as np
import threading
import queue
import base64
import time
from pathlib import Path
from flask import Flask, request, jsonify, make_response, send_from_directory, Response
from flask_cors import CORS

# Import our new modules
import database
import pipeline

app = Flask(__name__)
CORS(app)

PROJECT_DIR = Path(__file__).resolve().parent

# IMPORTANT: these live OUTSIDE the project folder, in the OS temp directory.
BASE_TMP_DIR = Path(tempfile.gettempdir()) / "yolov11m_human_detection"
UPLOAD_DIR = BASE_TMP_DIR / "uploads"
OUTPUT_DIR = BASE_TMP_DIR / "outputs"
COORDS_DIR = BASE_TMP_DIR / "coords"
PHOTOS_DIR = BASE_TMP_DIR / "photos"

for d in [UPLOAD_DIR, OUTPUT_DIR, COORDS_DIR, PHOTOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants from previous version
PERSON_CLASS_ID = 0
BOX_COLOR       = (0, 200, 80)     # green  - person outside any zone
LABEL_BG        = (0, 150, 60)
ZONE_BOX_COLOR  = (0, 140, 255)    # orange - person inside a zone
ZONE_LABEL_BG   = (0, 100, 210)
ALERT_BOX_COLOR = (0, 0, 255)      # red - blocklisted person
ALERT_LABEL_BG  = (0, 0, 200)
AUTH_BOX_COLOR  = (255, 144, 30)   # blue - authorized person
AUTH_LABEL_BG   = (200, 100, 0)
ZONE_LINE_COLOR = (0, 220, 255)    
TEXT_COLOR      = (255, 255, 255)

CONF_MIN, CONF_MAX = 0.15, 0.75
IOU_MIN, IOU_MAX   = 0.30, 0.75
CALIBRATION_SAMPLES = 15

# Global State
_jobs = {}
_webcam_thread = None
_webcam_running = False
_webcam_frame = None
_webcam_lock = threading.Lock()
_alert_subscribers = []
_enrolled_faces_cache = []

# Alert debounce: prevents the same alert from spamming the DB/UI
_alert_cooldowns = {}  # key: "blocklist:{name}" or "zone:{zone_name}" → last alert timestamp
ALERT_COOLDOWN_SECS = 5  # Don't re-alert the same event within 5 seconds

def _should_alert(key):
    """Return True if enough time has passed since the last alert for this key."""
    now = time.time()
    last = _alert_cooldowns.get(key, 0)
    if now - last >= ALERT_COOLDOWN_SECS:
        _alert_cooldowns[key] = now
        return True
    return False

def refresh_faces_cache():
    global _enrolled_faces_cache
    _enrolled_faces_cache = database.get_all_embeddings()

refresh_faces_cache()

def push_alert_to_subscribers(alert_dict):
    # Iterate over a copy so concurrent SSE disconnects don't mutate the list mid-loop
    for q in list(_alert_subscribers):
        try:
            q.put_nowait(alert_dict)
        except queue.Full:
            pass

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ============================================================
# ZONE GEOMETRY & CALIBRATION HELPERS
# ============================================================

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    if n < 3: return False
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    xinters = x1 if y1 == y2 else (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    return inside

def normalize_zones(zones_input, width, height):
    zones = []
    for z in zones_input or []:
        pts = z.get("points", [])
        if len(pts) >= 3:
            pixel_pts = [(float(p[0]) * width, float(p[1]) * height) for p in pts]
            zones.append({"name": z.get("name", "Zone"), "points": pixel_pts})
    return zones

def sample_frame_indices(total_frames, n_samples=CALIBRATION_SAMPLES):
    if total_frames <= 0: return []
    if total_frames <= n_samples: return list(range(total_frames))
    step = total_frames / n_samples
    return [int(i * step) for i in range(n_samples)]

def compute_brightness(gray_frame): return float(gray_frame.mean())
def compute_blur_sharpness(gray_frame): return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

def iou_xyxy(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def otsu_split(values):
    if len(values) < 6: return None
    arr = np.clip(np.array(values, dtype=np.float32) * 255, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([arr], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0: return None
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b, weight_b, max_var, best_t = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0: continue
        weight_f = total - weight_b
        if weight_f == 0: break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var, best_t = var_between, t
    return best_t / 255.0

def get_ffmpeg_executable():
    import shutil
    cmd = shutil.which("ffmpeg")
    if cmd: return cmd
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def reencode_to_h264(input_path, output_path):
    ffmpeg_cmd = get_ffmpeg_executable()
    try:
        result = subprocess.run([
            ffmpeg_cmd, "-y",
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ], capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

# ============================================================
# PROCESSING PIPELINE LOGIC
# ============================================================

def process_frame(frame, conf_thresh, iou_thresh, zones, thickness, font_scale,
                  job_id=None, frame_num=None, faces_override=None):
    """
    Run YOLO, run face detection, draw boxes, trigger alerts.
    faces_override: if provided (list), skip pipeline.detect_faces and use these instead.
                    Allows caller to cache face detections for efficiency (e.g. webcam).
    """
    # 1. Detect Persons
    detections = pipeline.detect_persons(frame, conf_thresh, iou_thresh)
    
    # 2. Detect Faces for Recognition
    if faces_override is not None:
        faces = faces_override
    else:
        faces = pipeline.detect_faces(frame) if len(detections) > 0 else []
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    zone_counts = {z["name"]: 0 for z in zones}
    zone_tags = []
    
    # For CSV
    processed_dets = []

    for det in detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        count += 1
        
        # Zone logic
        ref_x, ref_y = (x1 + x2) / 2, y2
        zone_name = None
        for z in zones:
            if point_in_polygon(ref_x, ref_y, z["points"]):
                zone_name = z["name"]
                zone_counts[zone_name] += 1
                break
        zone_tags.append(zone_name)
        
        # Face matching logic for this person box
        matched_name = None
        matched_status = None
        matched_sim = 0.0
        
        # Find if any detected face falls inside this person's bounding box
        for face in faces:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            # Center of face
            fcx, fcy = (fx1+fx2)/2, (fy1+fy2)/2
            if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                # Face belongs to this person
                match, sim = pipeline.match_face(face.normed_embedding, _enrolled_faces_cache)
                if match:
                    matched_name = match["name"]
                    matched_status = match["status"]
                    matched_sim = sim
                    break
        
        # Determine styling
        b_color = BOX_COLOR
        l_bg = LABEL_BG
        
        if matched_status == "blocklisted":
            b_color = ALERT_BOX_COLOR
            l_bg = ALERT_LABEL_BG
            # Trigger Alert (debounced)
            alert_key = f"blocklist:{matched_name}"
            if _should_alert(alert_key):
                alert_dict = database.save_alert(matched_name, matched_sim, job_id, frame_num,
                                                 alert_type="blocklist")
                push_alert_to_subscribers(alert_dict)
            
        elif matched_status == "authorized":
            b_color = AUTH_BOX_COLOR
            l_bg = AUTH_LABEL_BG
        elif zone_name:
            b_color = ZONE_BOX_COLOR
            l_bg = ZONE_LABEL_BG
            # Trigger zone intrusion alert (debounced)
            alert_key = f"zone:{zone_name}"
            if _should_alert(alert_key):
                alert_dict = database.save_alert(
                    f"Zone Intrusion: {zone_name}", 0.0, job_id, frame_num,
                    alert_type="zone_intrusion"
                )
                push_alert_to_subscribers(alert_dict)
            
        label = f"Person {conf:.2f}"
        if matched_name:
            label = f"{matched_name} {matched_sim:.2f}"
        elif zone_name:
            label += f" | {zone_name}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, thickness)
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + bl), l_bg, -1)
        cv2.putText(frame, label, (x1 + 2, ly), font, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
        
        processed_dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf,
            "zone": zone_name, "matched_name": matched_name, "matched_status": matched_status
        })

    # Draw zones
    for z in zones:
        pts = np.array(z["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=ZONE_LINE_COLOR, thickness=2)
        label_x, label_y = int(z["points"][0][0]), max(int(z["points"][0][1]) - 10, 14)
        cv2.putText(frame, z["name"], (label_x, label_y), font, 0.6, ZONE_LINE_COLOR, 2, cv2.LINE_AA)

    return frame, count, zone_counts, processed_dets

def overlay_hud(frame, frame_num, total, count, zone_counts=None):
    if total:
        hud = f"Frame {frame_num}/{total}  |  Persons: {count}"
    else:
        hud = f"LIVE  |  Persons: {count}"
        
    if zone_counts:
        hud += "  |  " + "  ".join(f"{name}: {n}" for name, n in zone_counts.items())
    cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    return frame

# ============================================================
# FLASK ROUTES - MAIN APP
# ============================================================

@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(PROJECT_DIR, "index.html")

@app.route("/health", methods=["GET"])
def health():
    ffmpeg_cmd = get_ffmpeg_executable()
    try:
        subprocess.run([ffmpeg_cmd, "-version"], capture_output=True, timeout=5)
        ffmpeg = True
    except Exception:
        ffmpeg = False
        
    # Check InsightFace model
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_m"
    insightface_ok = model_dir.exists() and any(model_dir.iterdir())
        
    return jsonify({
        "status": "ok", 
        "model": "yolo11m", 
        "cuda": _cuda_available(), 
        "ffmpeg": ffmpeg,
        "insightface": insightface_ok
    })

# ============================================================
# FLASK ROUTES - PROCESSING (BACKGROUND JOB)
# ============================================================

def _process_worker(job_id, input_path, conf_thresh, iou_thresh, thickness, font_scale, zones, raw_path, output_path, coords_path):
    job = _jobs[job_id]
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        job['status']['error'] = "Cannot open video"
        job['status']['done'] = True
        job['frame_queue'].put(None)
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    total_det   = 0
    peak        = 0
    frame_num   = 0
    all_coords  = []
    zone_peak   = {z["name"]: 0 for z in zones}
    zone_total  = {z["name"]: 0 for z in zones}

    job['status']['total_frames'] = total
    
    # Track processing fps for the preview HUD
    preview_t_last = time.time()
    preview_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        frame, count, zone_counts, processed_dets = process_frame(
            frame, conf_thresh, iou_thresh, zones, thickness, font_scale, job_id, frame_num
        )
        frame = overlay_hud(frame, frame_num, total, count, zone_counts)
        
        total_det += count
        peak = max(peak, count)
        for name, n in zone_counts.items():
            zone_total[name] += n
            zone_peak[name] = max(zone_peak[name], n)
            
        writer.write(frame)

        for i, d in enumerate(processed_dets):
            all_coords.append({
                "frame": frame_num,
                "person_id": i + 1,
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "width_px": d["x2"]-d["x1"],
                "height_px": d["y2"]-d["y1"],
                "confidence": round(d["conf"], 4),
                "frame_width": width,
                "frame_height": height,
                "zone": d["zone"] or "",
                "matched_name": d["matched_name"] or "",
                "matched_status": d["matched_status"] or ""
            })

        # Update Job Status
        now = time.time()
        elapsed = now - preview_t_last
        if elapsed > 0:
            preview_fps = round(1.0 / elapsed, 1)
        preview_t_last = now

        job['status']['frame_num'] = frame_num
        job['status']['progress_pct'] = int((frame_num / total) * 100) if total else 0
        job['status']['peak'] = peak
        job['status']['count'] = count
        job['status']['zone_counts'] = zone_counts
        job['status']['preview_fps'] = preview_fps
        
        # Push EVERY frame to SSE (queue drops if consumer is slow)
        # Resize to 480px wide — fast to encode/decode, keeps UI smooth
        preview_width = min(480, width)
        preview_height = int(height * (preview_width / width))
        preview_frame = cv2.resize(frame, (preview_width, preview_height))
        
        _, jpeg = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
        b64_frame = base64.b64encode(jpeg).decode()
        
        stats_json = json.dumps({
            "frame": frame_num, "total": total, "count": count,
            "zones": zone_counts, "fps": preview_fps
        })
        
        try:
            job['frame_queue'].put_nowait({
                "frame": b64_frame,
                "stats": stats_json
            })
        except queue.Full:
            pass  # Drop frame if browser reads slowly — prevents blocking YOLO

    cap.release()
    writer.release()
    
    # Save CSV
    with open(str(coords_path), "w", newline="") as csvfile:
        fieldnames = ["frame","person_id","x1","y1","x2","y2","width_px","height_px",
                      "confidence","frame_width","frame_height","zone","matched_name","matched_status"]
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(all_coords)

    job['status']['encoding'] = True
    h264_ok = reencode_to_h264(raw_path, output_path)
    
    if h264_ok:
        raw_path.unlink(missing_ok=True)
    else:
        # if ffmpeg fails, fallback to raw
        job['status']['ffmpeg_fail'] = True
    
    input_path.unlink(missing_ok=True)
    
    job['status']['done'] = True
    job['status']['stats'] = {
        "frames": frame_num,
        "total_detections": int(total_det),
        "peak_per_frame": int(peak),
        "coords_rows": len(all_coords),
        "zones": [{"name": n, "peak": zone_peak[n], "total_frames_occupied": zone_total[n]} for n in zone_peak]
    }
    job['frame_queue'].put(None) # EOF sentinel

@app.route("/process/start", methods=["POST"])
def process_start():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    conf_thresh = float(request.form.get("conf", 0.40))
    iou_thresh  = float(request.form.get("iou", 0.45))
    box_thick   = int(request.form.get("thickness", 2))
    font_size   = int(request.form.get("font_size", 16))
    font_scale  = (font_size - 10) / (28 - 10) * 0.65 + 0.35

    try:
        zones_input = json.loads(request.form.get("zones", "[]"))
    except (json.JSONDecodeError, TypeError):
        zones_input = []

    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or ".mp4"
    
    input_path   = UPLOAD_DIR / f"{job_id}_input{suffix}"
    raw_path     = OUTPUT_DIR / f"{job_id}_raw.mp4"
    output_path  = OUTPUT_DIR / f"{job_id}_detected.mp4"
    coords_path  = COORDS_DIR / f"{job_id}_coords.csv"

    file.save(str(input_path))
    
    # Just need first frame for zone normalization width/height
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    zones = normalize_zones(zones_input, width, height)

    _jobs[job_id] = {
        "frame_queue": queue.Queue(maxsize=30),
        "status": {
            "job_id": job_id,
            "progress_pct": 0,
            "done": False,
            "encoding": False,
            "total_frames": 0,
            "frame_num": 0,
        },
        "paths": {
            "output_path": output_path,
            "raw_path": raw_path,
            "coords_path": coords_path
        }
    }

    t = threading.Thread(target=_process_worker, args=(
        job_id, input_path, conf_thresh, iou_thresh, box_thick, font_scale, zones, raw_path, output_path, coords_path
    ))
    t.start()
    _jobs[job_id]['thread'] = t

    return jsonify({"job_id": job_id})

@app.route("/process/stream/<job_id>", methods=["GET"])
def process_stream(job_id):
    if job_id not in _jobs:
        return "Job not found", 404

    def generate():
        q = _jobs[job_id]['frame_queue']
        while True:
            item = q.get()
            if item is None:
                break
            # Send stats
            yield f"event: stats\ndata: {item['stats']}\n\n"
            # Send frame
            yield f"event: frame\ndata: {item['frame']}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route("/process/status/<job_id>", methods=["GET"])
def process_status(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(_jobs[job_id]['status'])

@app.route("/process/result/<job_id>", methods=["GET"])
def process_result(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = _jobs[job_id]
    if not job['status']['done']:
        return jsonify({"error": "Job not finished"}), 400
        
    final_path = job['paths']['output_path']
    if job['status'].get('ffmpeg_fail'):
        final_path = job['paths']['raw_path']
        
    with open(str(final_path), "rb") as f:
        video_bytes = f.read()

    response = make_response(video_bytes)
    response.status_code = 200
    response.headers["Content-Type"] = "video/mp4"
    response.headers["Content-Disposition"] = f'inline; filename="detected_{job_id}.mp4"'
    
    stats_json = json.dumps(job['status'].get('stats', {}))
    response.headers["X-Stats"] = stats_json
    response.headers["X-Job-Id"] = job_id
    response.headers["Access-Control-Expose-Headers"] = "X-Stats, X-Job-Id"
    
    return response

# Backwards compatible route if the old frontend code calls it
@app.route("/process", methods=["POST"])
def process_video():
    return jsonify({"error": "Please use /process/start flow instead"}), 400

@app.route("/coords/<job_id>", methods=["GET"])
def get_coords(job_id):
    job_id = job_id.replace("..", "").replace("/", "").replace("\\", "")
    coords_path = COORDS_DIR / f"{job_id}_coords.csv"
    if not coords_path.exists():
        return jsonify({"error": "Coordinates file not found"}), 404
    with open(str(coords_path), "rb") as f:
        csv_bytes = f.read()
    response = make_response(csv_bytes)
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = f'attachment; filename="detections_{job_id}.csv"'
    return response

# ============================================================
# WEBCAM ROUTES
# ============================================================

_webcam_error = None  # Holds last webcam error message

def _webcam_worker(conf, iou, thickness, font_scale, zones_input):
    """Unified webcam worker — uses the same process_frame as video processing."""
    global _webcam_running, _webcam_frame, _webcam_error
    _webcam_error = None

    # On Windows, cv2.CAP_DSHOW avoids the long 5-second probe delay
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
    cap = cv2.VideoCapture(0, backend)
    
    if not cap.isOpened():
        _webcam_error = "Cannot open camera. Is it connected and not in use by another app?"
        print(f"[ERROR] {_webcam_error}")
        _webcam_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get actual dimensions for zone normalization
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zones = normalize_zones(zones_input, width, height)
    
    frame_num = 0
    RECOG_SKIP = 3  # Run face recognition every 3rd frame (expensive)
    last_faces = []
    
    while _webcam_running:
        ret, frame = cap.read()
        if not ret:
            _webcam_error = "Camera stopped sending frames."
            break
            
        frame_num += 1

        # Face detection only every Nth frame for performance
        if frame_num % RECOG_SKIP == 0 or frame_num == 1:
            # Quick person check to avoid running InsightFace on empty frames
            detections_peek = pipeline.detect_persons(frame, conf, iou)
            last_faces = pipeline.detect_faces(frame) if len(detections_peek) > 0 else []

        # Unified pipeline — same function as video processing
        frame, count, zone_counts, _ = process_frame(
            frame, conf, iou, zones, thickness, font_scale,
            job_id="WEBCAM", frame_num=frame_num,
            faces_override=last_faces
        )
        frame = overlay_hud(frame, frame_num, 0, count, zone_counts)
        
        ret2, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret2:
            with _webcam_lock:
                _webcam_frame = jpeg.tobytes()

    cap.release()
    _webcam_running = False

@app.route("/webcam/start", methods=["POST"])
def webcam_start():
    global _webcam_thread, _webcam_running, _webcam_frame, _webcam_error
    if _webcam_running:
        return jsonify({"status": "already running"})

    # Accept detection config from the frontend
    conf = float(request.form.get("conf", 0.40))
    iou = float(request.form.get("iou", 0.45))
    thickness = int(request.form.get("thickness", 2))
    font_size = int(request.form.get("font_size", 16))
    font_scale = (font_size - 10) / (28 - 10) * 0.65 + 0.35
    try:
        zones_input = json.loads(request.form.get("zones", "[]"))
    except (json.JSONDecodeError, TypeError):
        zones_input = []

    _webcam_frame = None   # Clear stale frame from previous session
    _webcam_error = None   # Clear stale error
    _webcam_running = True
    _webcam_thread = threading.Thread(
        target=_webcam_worker,
        args=(conf, iou, thickness, font_scale, zones_input),
        daemon=True
    )
    _webcam_thread.start()
    # Give the thread time to open the camera and report any errors
    time.sleep(1.5)

    # If the worker failed to open the camera, report it immediately
    if _webcam_error:
        _webcam_running = False
        return jsonify({"error": _webcam_error}), 400

    return jsonify({"status": "started"})

@app.route("/webcam/status", methods=["GET"])
def webcam_status():
    return jsonify({
        "running": _webcam_running,
        "error": _webcam_error,
        "has_frame": _webcam_frame is not None
    })

@app.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    global _webcam_running
    _webcam_running = False
    return jsonify({"status": "stopped"})

@app.route("/webcam/stream", methods=["GET"])
def webcam_stream():
    if not _webcam_running:
        return "Webcam not started. Call /webcam/start first.", 400
        
    def generate():
        while _webcam_running:
            with _webcam_lock:
                frame = _webcam_frame
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================
# ENROLL ROUTES
# ============================================================

@app.route("/enroll", methods=["POST"])
def enroll():
    if "photo" not in request.files:
        return jsonify({"error": "No photo file"}), 400
        
    name = request.form.get("name")
    status = request.form.get("status", "authorized")
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
        
    file = request.files["photo"]
    suffix = Path(file.filename).suffix or ".jpg"
    photo_filename = f"{uuid.uuid4()}{suffix}"
    photo_path = PHOTOS_DIR / photo_filename
    file.save(str(photo_path))
    
    # Extract embedding
    img = cv2.imread(str(photo_path))
    if img is None:
        photo_path.unlink()
        return jsonify({"error": "Invalid image format"}), 400
        
    embedding = pipeline.get_embedding(img)
    if embedding is None:
        photo_path.unlink()
        return jsonify({"error": "No face detected in photo"}), 400
        
    person_id = database.add_person(name, status, photo_path, embedding)
    refresh_faces_cache()
    
    return jsonify({
        "status": "success", 
        "person_id": person_id,
        "name": name,
        "role": status
    })

@app.route("/people", methods=["GET"])
def list_people():
    people = database.get_all_people()
    return jsonify(people)

@app.route("/people/<int:person_id>", methods=["DELETE"])
def delete_person(person_id):
    database.delete_person(person_id)
    refresh_faces_cache()
    return jsonify({"status": "deleted"})

# ============================================================
# ALERTS ROUTES
# ============================================================

@app.route("/alerts", methods=["GET"])
def get_alerts():
    alerts = database.get_recent_alerts(limit=50)
    return jsonify(alerts)

@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    database.clear_alerts()
    return jsonify({"status": "cleared"})

@app.route("/alerts/stream", methods=["GET"])
def stream_alerts():
    q = queue.Queue(maxsize=10)
    _alert_subscribers.append(q)
    
    def generate():
        try:
            while True:
                alert = q.get()
                yield f"data: {json.dumps(alert)}\n\n"
        except GeneratorExit:
            _alert_subscribers.remove(q)
            
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route("/auto_calibrate", methods=["POST"])
def auto_calibrate():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_path = UPLOAD_DIR / f"{job_id}_calib{suffix}"
    file.save(str(tmp_path))

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": "Cannot open video"}), 400

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sample_idxs = sample_frame_indices(total)

    confidences, per_frame_counts, overlap_ratios = [], [], []
    brightness_vals, blur_vals = [], []

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(compute_brightness(gray))
        blur_vals.append(compute_blur_sharpness(gray))

        detections = pipeline.detect_persons(frame, 0.05, 0.5)
        frame_boxes = []
        for det in detections:
            confidences.append(float(det[4]))
            frame_boxes.append(det[:4])

        per_frame_counts.append(len(frame_boxes))
        if len(frame_boxes) >= 2:
            pair_overlaps = [
                iou_xyxy(frame_boxes[i], frame_boxes[j])
                for i in range(len(frame_boxes))
                for j in range(i + 1, len(frame_boxes))
            ]
            pair_overlaps = [o for o in pair_overlaps if o > 0]
            if pair_overlaps:
                overlap_ratios.append(float(np.mean(pair_overlaps)))

    cap.release()
    tmp_path.unlink(missing_ok=True)

    reasoning = []

    split = otsu_split(confidences)
    rec_conf = split if split is not None else 0.40
    if split is None:
        reasoning.append("Too few candidate detections in the sampled frames — starting from the standard baseline.")

    avg_brightness = float(np.mean(brightness_vals)) if brightness_vals else 128.0
    avg_blur       = float(np.mean(blur_vals)) if blur_vals else 200.0

    if avg_brightness < 70:
        rec_conf -= 0.07
        reasoning.append(f"Dark footage (avg brightness {avg_brightness:.0f}/255) — lowered confidence so dim detections aren't missed.")
    elif avg_brightness > 190:
        rec_conf += 0.03
        reasoning.append(f"Bright, high-contrast footage (avg brightness {avg_brightness:.0f}/255) — raised confidence slightly.")

    if avg_blur < 80:
        rec_conf -= 0.05
        reasoning.append(f"Footage looks soft/blurry (sharpness {avg_blur:.0f}) — lowered confidence to compensate.")

    rec_conf = float(np.clip(rec_conf, CONF_MIN, CONF_MAX))

    avg_persons  = float(np.mean(per_frame_counts)) if per_frame_counts else 0.0
    avg_overlap  = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

    if avg_overlap > 0.25 or avg_persons > 8:
        rec_iou = 0.65
        reasoning.append(f"Crowded scene (avg {avg_persons:.1f} people/frame, box overlap {avg_overlap:.2f}) — raised IoU so adjacent people aren't merged into one box.")
    elif avg_overlap > 0.12 or avg_persons > 4:
        rec_iou = 0.50
        reasoning.append(f"Moderate crowd density (avg {avg_persons:.1f} people/frame) — using a balanced IoU.")
    else:
        rec_iou = 0.40
        reasoning.append(f"Sparse scene (avg {avg_persons:.1f} people/frame) — tighter IoU to clean up duplicate boxes.")

    rec_iou = float(np.clip(rec_iou, IOU_MIN, IOU_MAX))

    if not reasoning:
        reasoning.append("Typical lighting and crowd density — using balanced defaults.")

    return jsonify({
        "recommended_conf": round(rec_conf, 2),
        "recommended_iou": round(rec_iou, 2),
        "reasoning": reasoning,
        "details": {
            "sampled_frames": len(sample_idxs),
            "avg_persons_per_frame": round(avg_persons, 2),
            "avg_brightness": round(avg_brightness, 1),
            "avg_blur_sharpness": round(avg_blur, 1),
            "avg_overlap": round(avg_overlap, 3),
            "total_candidate_detections": len(confidences)
        }
    })

if __name__ == "__main__":
    print(f"[INFO] CUDA: {_cuda_available()}")
    print("[INFO] Open this in your browser: http://localhost:5000")
    # threaded=True is required for SSE streaming
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)