"""
=============================================================================
Construction Site Safety Violation Detector
=============================================================================
Author      : AIML Capstone Project
Description : Detects PPE violations (no helmet, no vest) on construction
              sites using YOLOv8 object detection on images and video.
=============================================================================
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Try to import ultralytics (YOLOv8). Falls back to a lightweight OpenCV-only
# demo mode so the project still runs without a GPU / heavy install.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – running OpenCV-only demo mode.")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Person",
    1: "Helmet",
    2: "No-Helmet",      # PPE violation
    3: "Safety-Vest",
    4: "No-Vest",        # PPE violation
    5: "Gloves",
    6: "Safety-Boots",
}

# Colour palette per class  (BGR)
CLASS_COLORS = {
    "Person":        (255, 178,  50),
    "Helmet":        ( 50, 205,  50),
    "No-Helmet":     ( 50,  50, 255),   # red  → violation
    "Safety-Vest":   ( 50, 205,  50),
    "No-Vest":       ( 50,  50, 255),   # red  → violation
    "Gloves":        (255, 255,   0),
    "Safety-Boots":  (255, 255,   0),
    "Unknown":       (200, 200, 200),
}

VIOLATION_CLASSES = {"No-Helmet", "No-Vest"}

CONF_THRESHOLD  = 0.45   # Minimum confidence to accept a detection
IOU_THRESHOLD   = 0.45   # NMS IoU threshold


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resize frame to a square (letterbox style) while preserving aspect ratio.
    YOLOv8's built-in pre-processing is used when YOLO_AVAILABLE=True, but
    we still call this for display normalisation.
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def draw_detection(frame: np.ndarray,
                   x1: int, y1: int, x2: int, y2: int,
                   label: str, confidence: float,
                   is_violation: bool) -> np.ndarray:
    """
    Draw a bounding box and label onto *frame* (in-place).
    Violations are drawn with a thicker red box.
    """
    color      = CLASS_COLORS.get(label, CLASS_COLORS["Unknown"])
    thickness  = 3 if is_violation else 2
    text       = f"{label} {confidence:.2f}"

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

    # Label text
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)

    # Violation warning icon
    if is_violation:
        cv2.putText(frame, "⚠ VIOLATION", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return frame


def overlay_stats(frame: np.ndarray,
                  total_persons: int,
                  violations: int,
                  fps: float) -> np.ndarray:
    """
    Draws a semi-transparent HUD in the top-left corner.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark background panel
    cv2.rectangle(overlay, (0, 0), (280, 115), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status_color = (0, 0, 255) if violations > 0 else (0, 200, 0)
    status_text  = "UNSAFE" if violations > 0 else "SAFE"

    cv2.putText(frame, f"Status: {status_text}",       (10,  25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(frame, f"Persons detected: {total_persons}", (10,  50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Violations:  {violations}",   (10,  73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if violations else (0, 200, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",              (10,  96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
    return frame


def calculate_iou(box1, box2) -> float:
    """
    Intersection-over-Union for two boxes [x1,y1,x2,y2].
    Used in the evaluation module.
    """
    xi1 = max(box1[0], box2[0]);  yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]);  yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Core detector class
# ─────────────────────────────────────────────────────────────────────────────

class SafetyDetector:
    """
    Wraps a YOLOv8 model (or a dummy demo mode) and exposes a simple
    process_frame() interface.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.device         = device
        self.violation_log  = []
        self.frame_count    = 0
        self.total_violations = 0

        if YOLO_AVAILABLE:
            print(f"[INFO] Loading model: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(device)
            print("[INFO] Model ready.")
        else:
            self.model = None
            print("[INFO] Demo mode – bounding boxes are synthetic.")

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray):
        """
        Run inference on a single BGR frame.
        Returns (annotated_frame, detection_list).
        detection_list: [{"label", "confidence", "bbox":[x1,y1,x2,y2], "violation": bool}]
        """
        self.frame_count += 1
        detections = []

        if YOLO_AVAILABLE and self.model is not None:
            results = self.model.predict(
                source     = frame,
                conf       = CONF_THRESHOLD,
                iou        = IOU_THRESHOLD,
                device     = self.device,
                verbose    = False,
            )[0]

            for box in results.boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label      = CLASS_NAMES.get(cls_id, "Unknown")
                is_viol    = label in VIOLATION_CLASSES

                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "violation":  is_viol,
                })

                frame = draw_detection(frame, x1, y1, x2, y2,
                                       label, conf, is_viol)

        else:
            # ── Demo mode: draw a single synthetic bounding box ──────────
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4
            frame = draw_detection(frame, x1, y1, x2, y2,
                                   "No-Helmet", 0.87, True)
            detections.append({
                "label":      "No-Helmet",
                "confidence": 0.87,
                "bbox":       [x1, y1, x2, y2],
                "violation":  True,
            })

        # ── Aggregate stats ──────────────────────────────────────────────
        persons    = sum(1 for d in detections if d["label"] == "Person")
        violations = sum(1 for d in detections if d["violation"])
        self.total_violations += violations

        if violations > 0:
            self.violation_log.append({
                "frame":      self.frame_count,
                "timestamp":  datetime.now().isoformat(),
                "violations": violations,
            })

        return frame, detections, persons, violations

    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        return {
            "total_frames":     self.frame_count,
            "total_violations": self.total_violations,
            "violation_events": len(self.violation_log),
            "log":              self.violation_log,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Processing pipelines
# ─────────────────────────────────────────────────────────────────────────────

def process_image(detector: SafetyDetector,
                  input_path: str,
                  output_dir: str = "outputs/detections") -> str:
    """Run detection on a single image and save annotated result."""
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    annotated, detections, persons, violations = detector.process_frame(frame)
    annotated = overlay_stats(annotated, persons, violations, fps=0.0)

    os.makedirs(output_dir, exist_ok=True)
    stem      = Path(input_path).stem
    out_path  = os.path.join(output_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"[INFO] Image saved → {out_path}")
    print(f"       Persons: {persons} | Violations: {violations}")
    return out_path


def process_video(detector: SafetyDetector,
                  input_path: str,
                  output_dir: str = "outputs/videos",
                  display: bool  = False) -> str:
    """
    Frame-by-frame video processing with optional live preview.
    Writes an annotated output video.
    """
    cap = cv2.VideoCapture(0 if input_path == "0" else input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {input_path}")

    # Video writer setup
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_in   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_path).stem if input_path != "0" else "webcam"
    out_path = os.path.join(output_dir, f"{stem}_detected.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps_in, (w_in, h_in))

    prev_time = time.time()
    print("[INFO] Processing video … Press Q to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _, persons, violations = detector.process_frame(frame)

        # FPS calculation
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        annotated = overlay_stats(annotated, persons, violations, fps)
        writer.write(annotated)

        if display:
            cv2.imshow("Safety Monitor", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    summary = detector.get_summary()
    print(f"\n[SUMMARY] Frames: {summary['total_frames']} | "
          f"Total violations: {summary['total_violations']}")
    print(f"[INFO] Video saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Construction Site PPE Violation Detector"
    )
    p.add_argument("--source",  required=True,
                   help="Path to image/video file or '0' for webcam")
    p.add_argument("--model",   default="models/pretrained/ppe_yolov8n.pt",
                   help="Path to YOLOv8 .pt weights")
    p.add_argument("--device",  default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--output",  default="outputs/detections",
                   help="Output directory")
    p.add_argument("--display", action="store_true",
                   help="Show live preview window (video/webcam only)")
    return p.parse_args()


def main():
    args    = parse_args()
    detector = SafetyDetector(model_path=args.model, device=args.device)

    src = args.source
    ext = Path(src).suffix.lower() if src != "0" else ".mp4"

    if src == "0" or ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        process_video(detector, src,
                      output_dir=args.output, display=args.display)
    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        process_image(detector, src, output_dir=args.output)
    else:
        print(f"[ERROR] Unsupported source: {src}")


if __name__ == "__main__":
    main()
