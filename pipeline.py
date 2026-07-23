import cv2
import numpy as np
import threading
import urllib.request
import os
import tarfile
from pathlib import Path
from ultralytics import YOLO

# Shared Models
MODEL_YOLO = None
MODEL_APP = None # InsightFace model

# Constants
PERSON_CLASS_ID = 0
CONF_THRESH_DEFAULT = 0.40
IOU_THRESH_DEFAULT = 0.45
RECOG_THRESH = 0.35 # Cosine similarity threshold

def _ensure_insightface_model():
    """
    Downloads the buffalo_m model if it doesn't exist in ~/.insightface/models/buffalo_m
    """
    home = Path.home()
    model_dir = home / ".insightface" / "models" / "buffalo_m"
    if model_dir.exists() and any(model_dir.iterdir()):
        return

    print("[INFO] Downloading InsightFace buffalo_m model (~190MB)...")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir.parent / "buffalo_m.zip"
    
    # URL for buffalo_m (this is a public mirror often used for InsightFace models)
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir.parent)
        zip_path.unlink()
        print("[INFO] Download and extraction complete.")
    except Exception as e:
        print(f"[ERROR] Failed to download InsightFace model: {e}")

def get_models():
    global MODEL_YOLO, MODEL_APP
    if MODEL_YOLO is None:
        try:
            print("[INFO] Loading YOLOv11m...")
            MODEL_YOLO = YOLO("yolo11m.pt")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            MODEL_YOLO = None
    
    if MODEL_APP is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            print("[INFO] Checking InsightFace models...")
            _ensure_insightface_model()
            
            print("[INFO] Loading FaceAnalysis (buffalo_m)...")
            MODEL_APP = FaceAnalysis(name='buffalo_m', root=str(Path.home() / ".insightface"))
            # -1 for CPU, 0 for GPU (if onnxruntime-gpu is installed)
            ctx_id = 0 if _cuda_available() else -1
            MODEL_APP.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except ImportError:
            print("[WARN] insightface package not found. Face recognition disabled.")
            MODEL_APP = None
        except Exception as e:
            print(f"[ERROR] Failed to load InsightFace model: {e}")
            MODEL_APP = None
            
    return MODEL_YOLO, MODEL_APP

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def detect_persons(frame, conf=CONF_THRESH_DEFAULT, iou=IOU_THRESH_DEFAULT):
    """Run YOLO to find persons."""
    yolo, _ = get_models()
    if not yolo:
        return []
    
    device = "cuda" if _cuda_available() else "cpu"
    results = yolo(frame, classes=[PERSON_CLASS_ID], conf=conf, iou=iou, verbose=False, device=device)
    
    detections = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                detections.append([*b, c]) # [x1, y1, x2, y2, conf]
    return detections

def detect_faces(frame):
    """Extract face bounding boxes using InsightFace."""
    _, app = get_models()
    if not app:
        return []
    
    faces = app.get(frame)
    return faces # list of Face objects (has .bbox, .normed_embedding)

def get_embedding(img):
    """
    Given a cropped face image (or full image containing a face),
    returns the embedding of the largest face found.
    """
    _, app = get_models()
    if not app:
        return None
        
    faces = app.get(img)
    if not faces:
        return None
    
    # Pick largest face if multiple
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
    return faces[0].normed_embedding

def extract_all_faces(frame):
    """Get all face embeddings and bboxes in a frame."""
    return detect_faces(frame)

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two 1D numpy arrays."""
    if emb1 is None or emb2 is None:
        return 0.0
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def match_face(face_embedding, enrolled_faces, threshold=RECOG_THRESH):
    """
    Find best match in enrolled_faces.
    enrolled_faces: list of dicts {"id":..., "name":..., "status":..., "embedding":...}
    Returns (best_match_dict, similarity_score) or (None, 0)
    """
    if face_embedding is None or not enrolled_faces:
        return None, 0.0
        
    best_match = None
    best_sim = -1
    
    for person in enrolled_faces:
        sim = cosine_similarity(face_embedding, person["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_match = person
            
    if best_sim >= threshold:
        return best_match, float(best_sim)
    return None, float(best_sim)

# Initialize on module load
get_models()
