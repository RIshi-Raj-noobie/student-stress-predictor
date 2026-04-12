"""
=============================================================================
train.py  –  Fine-tune YOLOv8 on the PPE Dataset
=============================================================================
Usage:
    python src/train.py --data data/raw/data.yaml --epochs 50 --batch 16
=============================================================================
"""

import argparse
import os
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 PPE fine-tuning")
    p.add_argument("--data",    default="data/raw/data.yaml",
                   help="Path to data.yaml")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="Base model weights (n/s/m/l/x)")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="cpu",
                   choices=["cpu","cuda","mps"])
    p.add_argument("--project", default="models/trained",
                   help="Save results to project/name")
    p.add_argument("--name",    default="ppe_run1")
    return p.parse_args()


def main():
    args = parse_args()

    if not YOLO_AVAILABLE:
        print("[ERROR] ultralytics not installed.")
        print("        Run: pip install ultralytics")
        return

    if not os.path.exists(args.data):
        print(f"[ERROR] data.yaml not found: {args.data}")
        print("        Run: python src/download_data.py --dest data/raw")
        return

    print("=" * 55)
    print("  Starting YOLOv8 PPE Detector Training")
    print("=" * 55)
    print(f"  Model  : {args.model}")
    print(f"  Data   : {args.data}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch}")
    print(f"  Device : {args.device}")
    print("=" * 55)

    model = YOLO(args.model)

    results = model.train(
        data    = args.data,
        epochs  = args.epochs,
        batch   = args.batch,
        imgsz   = args.imgsz,
        device  = args.device,
        project = args.project,
        name    = args.name,
        # Augmentation (helps generalisation)
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,
        degrees = 0.0,
        flipud  = 0.0,
        fliplr  = 0.5,
        mosaic  = 1.0,
    )

    # Export best weights
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        dest = Path("models/pretrained/ppe_yolov8n.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n[INFO] Best weights saved → {dest}")

    print("\n[DONE] Training complete.")
    print(f"       Results in: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
