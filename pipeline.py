"""
pipeline.py — OS-portable 4-stage face recognition pipeline.

Stages:
  1. Person detection     — YOLO11m
  2. Face detection       — SCRFD (via buffalo_m)
  3. Landmark alignment   — 5-point norm_crop (via InsightFace)
  4. Embedding extraction — ArcFace w600k_r50 (via buffalo_m)

Works on Windows, Linux, macOS — no hardcoded paths.
Models auto-download via InsightFace and ultralytics on first run.
"""

import cv2
import shutil
import numpy as np
import urllib.request
from pathlib import Path
from ultralytics import YOLO

# Constants
PERSON_CLASS_ID = 0
CONF_THRESH_DEFAULT = 0.40
IOU_THRESH_DEFAULT = 0.45
RECOG_THRESH = 0.35  # Cosine similarity threshold

# Path.home() resolves correctly on Windows, Linux, and macOS.
_MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_m"

# Shared model references (loaded once)
_yolo = None
_det = None   # SCRFD face detector
_rec = None   # ArcFace recognizer


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _ensure_models():
    """
    Verify buffalo_m model files exist. If not, trigger an auto-download
    via FaceAnalysis and fix any nested-folder extraction issues.
    """
    det_path = _MODEL_DIR / "det_2.5g.onnx"
    rec_path = _MODEL_DIR / "w600k_r50.onnx"

    if det_path.exists() and rec_path.exists():
        return True

    # Try auto-download via FaceAnalysis
    try:
        from insightface.app import FaceAnalysis
        print("[INFO] Downloading buffalo_m model bundle (~330MB, first run only)...")
        app = FaceAnalysis(name="buffalo_m")
        app.prepare(ctx_id=-1, det_size=(640, 640))

        # Fix nested folder issue that can occur on first extraction
        nested = _MODEL_DIR / "buffalo_m"
        if nested.exists() and nested.is_dir():
            for f in nested.iterdir():
                shutil.move(str(f), str(_MODEL_DIR / f.name))
            nested.rmdir()

        print("[INFO] buffalo_m model ready.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download InsightFace models: {e}")
        return False


def _load_models():
    """Load all models once (expensive — done once per process, not per frame)."""
    global _yolo, _det, _rec

    if _yolo is None:
        try:
            print("[INFO] Loading YOLOv11m...")
            _yolo = YOLO("yolo11m.pt")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")

    if _det is None or _rec is None:
        try:
            from insightface.model_zoo import model_zoo

            if not _ensure_models():
                print("[WARN] InsightFace models not available. Face recognition disabled.")
                return

            det_path = _MODEL_DIR / "det_2.5g.onnx"
            rec_path = _MODEL_DIR / "w600k_r50.onnx"

            if det_path.exists() and _det is None:
                print("[INFO] Loading SCRFD face detector...")
                _det = model_zoo.get_model(str(det_path))
                ctx_id = 0 if _cuda_available() else -1
                _det.prepare(ctx_id=ctx_id, input_size=(640, 640))

            if rec_path.exists() and _rec is None:
                print("[INFO] Loading ArcFace recognizer...")
                _rec = model_zoo.get_model(str(rec_path))
                ctx_id = 0 if _cuda_available() else -1
                _rec.prepare(ctx_id=ctx_id)

        except ImportError:
            print("[WARN] insightface package not found. Face recognition disabled.")
        except Exception as e:
            print(f"[ERROR] Failed to load InsightFace models: {e}")


# ── Stage 1: Person Detection ─────────────────────────────────────────────────

def detect_persons(frame, conf=CONF_THRESH_DEFAULT, iou=IOU_THRESH_DEFAULT):
    """
    Stage 1: YOLO11m person detection.

    Returns:
        List of [x1, y1, x2, y2, confidence] arrays.
    """
    if _yolo is None:
        return []

    device = "cuda" if _cuda_available() else "cpu"
    results = _yolo(frame, classes=[PERSON_CLASS_ID], conf=conf, iou=iou, verbose=False, device=device)

    detections = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                detections.append([*b, c])  # [x1, y1, x2, y2, conf]
    return detections


# ── Stage 2: Face Detection (SCRFD) ───────────────────────────────────────────

def detect_faces_scrfd(img):
    """
    Stage 2: SCRFD face detection + 5-point landmark localization.

    Args:
        img: BGR image (full frame or person crop).

    Returns:
        (bboxes, kpss) — face boxes [x1,y1,x2,y2,conf] and 5x2 landmark arrays.
        Returns (np.array([]), np.array([])) if no detector available.
    """
    if _det is None:
        return np.array([]), np.array([])

    bboxes, kpss = _det.detect(img, max_num=0, metric="default")
    return bboxes, kpss


# ── Stage 3: Alignment ────────────────────────────────────────────────────────

def align_face(img, kps):
    """
    Stage 3: Geometric alignment using 5 landmarks → 112x112 crop.

    Args:
        img: BGR image.
        kps: 5x2 array of (x, y) landmark coordinates.

    Returns:
        112x112x3 aligned face image.
    """
    from insightface.utils import face_align
    return face_align.norm_crop(img, kps, image_size=112)


# ── Stage 4: Embedding ────────────────────────────────────────────────────────

def get_embedding_from_aligned(aligned_face):
    """
    Stage 4: ArcFace embedding extraction + L2 normalization.

    Args:
        aligned_face: 112x112x3 BGR image from align_face().

    Returns:
        512-dimensional unit-norm float32 embedding vector.
    """
    if _rec is None:
        return None

    emb = _rec.get_feat(aligned_face).flatten()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def extract_all_faces(img):
    """
    Full 4-stage pipeline on a single image.

    Runs YOLO person detection, then per-person: SCRFD face detection,
    5-point alignment, ArcFace embedding extraction.

    Args:
        img: BGR image as numpy array.

    Returns:
        List of dicts, one per successfully processed face:
            {
                "person_bbox": [x1, y1, x2, y2, conf],
                "face_bbox":   np.ndarray [x1, y1, x2, y2, confidence] (global coords),
                "embedding":   np.ndarray shape (512,), unit-norm,
            }
        Empty list if no faces found.
    """
    if _det is None or _rec is None:
        return []

    results = []
    person_boxes = detect_persons(img)

    for det in person_boxes:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        person_crop = img[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        bboxes, kpss = detect_faces_scrfd(person_crop)
        if len(bboxes) == 0:
            # Person detected but no face found — still record it
            results.append({
                "person_bbox": [x1, y1, x2, y2, conf],
                "face_bbox": None,
                "embedding": None,
            })
            continue

        # Pick highest-confidence face
        best_idx = int(np.argmax(bboxes[:, 4]))
        kps = kpss[best_idx]

        aligned = align_face(person_crop, kps)
        embedding = get_embedding_from_aligned(aligned)

        # Convert crop-local face coordinates back to full-frame coordinates
        local_face_bbox = bboxes[best_idx].copy()
        global_face_bbox = np.array([
            local_face_bbox[0] + x1,
            local_face_bbox[1] + y1,
            local_face_bbox[2] + x1,
            local_face_bbox[3] + y1,
            local_face_bbox[4]  # Keep detection confidence score
        ], dtype=np.float32)

        results.append({
            "person_bbox": [x1, y1, x2, y2, conf],
            "face_bbox": global_face_bbox,
            "embedding": embedding,
        })

    return results


# ── Legacy compatibility functions ─────────────────────────────────────────────

def detect_faces(frame):
    """
    Legacy wrapper: Extract face bounding boxes using InsightFace FaceAnalysis.
    Used by the old process_frame code. New code should use extract_all_faces().
    """
    # Try to use FaceAnalysis if available (backward compat)
    try:
        from insightface.app import FaceAnalysis
        # Use a lightweight approach — detect faces directly with SCRFD
        if _det is None:
            return []
        bboxes, kpss = detect_faces_scrfd(frame)
        if len(bboxes) == 0:
            return []

        # Build face-like objects for backward compatibility
        face_results = []
        for i in range(len(bboxes)):
            aligned = align_face(frame, kpss[i])
            emb = get_embedding_from_aligned(aligned)
            if emb is not None:
                face_obj = type('Face', (), {
                    'bbox': bboxes[i][:4],
                    'normed_embedding': emb
                })()
                face_results.append(face_obj)
        return face_results
    except Exception:
        return []


def get_embedding(img):
    """
    Given an image containing a face, returns the embedding of the largest face found.
    Used for enrollment.
    """
    if _det is None or _rec is None:
        return None

    bboxes, kpss = detect_faces_scrfd(img)
    if len(bboxes) == 0:
        return None

    # Pick largest face by area
    if len(bboxes) > 1:
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best_idx = int(np.argmax(areas))
    else:
        best_idx = 0

    aligned = align_face(img, kpss[best_idx])
    return get_embedding_from_aligned(aligned)


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two 1D numpy arrays."""
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def match_face(face_embedding, enrolled_faces, threshold=RECOG_THRESH):
    """
    Find best match in enrolled_faces.
    enrolled_faces: list of dicts {"id":..., "name":..., "status":..., "embedding":...}
    Returns (best_match_dict, similarity_score) or (None, best_sim)
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


# Initialize models on module load
_load_models()
