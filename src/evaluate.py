"""
=============================================================================
evaluate.py  –  Evaluation Metrics for PPE Violation Detector
=============================================================================
Computes:
  • Precision, Recall, F1-Score  (per class and macro-average)
  • Mean Average Precision (mAP@0.5)
  • Intersection-over-Union (IoU) per detection
  • Confusion matrix (text-based for portability)
=============================================================================
"""

import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# IoU utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(pred_box: list, gt_box: list) -> float:
    """
    Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes.

    Args:
        pred_box : predicted bounding box  [x1, y1, x2, y2]
        gt_box   : ground-truth bounding box [x1, y1, x2, y2]

    Returns:
        float  IoU value in [0, 1]
    """
    # Intersection rectangle
    ix1 = max(pred_box[0], gt_box[0])
    iy1 = max(pred_box[1], gt_box[1])
    ix2 = min(pred_box[2], gt_box[2])
    iy2 = min(pred_box[3], gt_box[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_pred = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
    area_gt   = (gt_box[2]-gt_box[0])   * (gt_box[3]-gt_box[1])
    union = area_pred + area_gt - inter

    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class AP (Pascal VOC style)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using the 11-point interpolation
    (compatible with PASCAL VOC 2010 protocol).
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precision[recall >= thr]
        ap += (np.max(prec_at_rec) if len(prec_at_rec) else 0.0)
    return ap / 11.0


def evaluate_detections(predictions: list,
                         ground_truths: list,
                         iou_threshold: float = 0.5,
                         class_names: dict = None) -> dict:
    """
    Evaluate object-detection results across one or more images.

    Args:
        predictions  : list of dicts, each:
                        {"image_id": str,
                         "label": str,
                         "confidence": float,
                         "bbox": [x1,y1,x2,y2]}
        ground_truths: list of dicts, each:
                        {"image_id": str,
                         "label": str,
                         "bbox": [x1,y1,x2,y2]}
        iou_threshold: float  (default 0.5  → mAP@0.5)
        class_names  : optional dict {id: name} for display

    Returns:
        dict with keys:
            per_class_ap, mAP, precision, recall, f1
    """
    # Organise ground-truth by (image_id, label)
    gt_by_img_cls = defaultdict(list)
    for gt in ground_truths:
        gt_by_img_cls[(gt["image_id"], gt["label"])].append(
            {"bbox": gt["bbox"], "matched": False}
        )

    # Sort predictions by descending confidence
    preds_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    # Collect TP/FP arrays per class
    class_tp_fp = defaultdict(lambda: {"tp": [], "fp": [], "n_gt": 0})
    for gt in ground_truths:
        class_tp_fp[gt["label"]]["n_gt"] += 1

    for pred in preds_sorted:
        cls   = pred["label"]
        img   = pred["image_id"]
        p_box = pred["bbox"]

        gt_list = gt_by_img_cls.get((img, cls), [])
        best_iou  = -1.0
        best_idx  = -1

        for idx, gt_entry in enumerate(gt_list):
            iou = compute_iou(p_box, gt_entry["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and not gt_list[best_idx]["matched"]:
            class_tp_fp[cls]["tp"].append(1)
            class_tp_fp[cls]["fp"].append(0)
            gt_list[best_idx]["matched"] = True
        else:
            class_tp_fp[cls]["tp"].append(0)
            class_tp_fp[cls]["fp"].append(1)

    # Compute per-class AP
    per_class_ap  = {}
    all_precisions = []
    all_recalls    = []

    for cls, data in class_tp_fp.items():
        tp_cum = np.cumsum(data["tp"])
        fp_cum = np.cumsum(data["fp"])
        n_gt   = data["n_gt"]

        recall_arr    = tp_cum / (n_gt + 1e-6)
        precision_arr = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = compute_ap(recall_arr, precision_arr)
        per_class_ap[cls] = round(ap, 4)

        if len(recall_arr):
            all_recalls.append(recall_arr[-1])
            all_precisions.append(precision_arr[-1])

    mAP       = round(float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0, 4)
    precision = round(float(np.mean(all_precisions)) if all_precisions else 0.0, 4)
    recall    = round(float(np.mean(all_recalls))    if all_recalls    else 0.0, 4)
    f1        = round(2 * precision * recall / (precision + recall + 1e-6), 4)

    return {
        "per_class_ap": per_class_ap,
        "mAP":          mAP,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text-based confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
    """Build a confusion matrix (true classes × predicted classes)."""
    n   = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    cm  = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, labels: list) -> None:
    """Pretty-print confusion matrix to stdout."""
    col_w = max(10, max(len(l) for l in labels) + 2)
    header = "True\\Pred".ljust(col_w) + "".join(l.ljust(col_w) for l in labels)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(labels):
        row = row_label.ljust(col_w) + "".join(str(v).ljust(col_w) for v in cm[i])
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test / demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic example with 2 images
    gt = [
        {"image_id": "img1", "label": "No-Helmet", "bbox": [100,100,200,200]},
        {"image_id": "img1", "label": "Person",    "bbox": [50, 50, 300,400]},
        {"image_id": "img2", "label": "No-Vest",   "bbox": [120,120,220,300]},
    ]
    pred = [
        {"image_id": "img1", "label": "No-Helmet", "confidence": 0.91, "bbox": [105,105,205,205]},
        {"image_id": "img1", "label": "Person",    "confidence": 0.85, "bbox": [ 55, 55,305,405]},
        {"image_id": "img2", "label": "No-Vest",   "confidence": 0.78, "bbox": [115,115,215,295]},
        {"image_id": "img2", "label": "No-Helmet", "confidence": 0.60, "bbox": [  0,  0, 10, 10]},  # FP
    ]

    results = evaluate_detections(pred, gt)
    print("=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for cls, ap in results["per_class_ap"].items():
        print(f"  AP [{cls:>14s}]: {ap:.4f}")
    print(f"\n  mAP@0.5   : {results['mAP']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print("=" * 50)
