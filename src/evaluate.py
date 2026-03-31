"""
evaluate.py
===========
Loads the saved best model and produces a full evaluation report:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (saved as PNG)
  - Feature Importance plot (saved as PNG)
  - Classification report printed to console
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, json

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR   = os.path.join("data",   "processed")
MODEL_PATH      = os.path.join("models", "best_model.pkl")
ENCODER_PATH    = os.path.join("models", "label_encoder.pkl")
RESULTS_PATH    = os.path.join("models", "training_results.json")
REPORTS_DIR     = "reports"


def load_artifacts():
    """Load test data, trained model, and label encoder."""
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("[INFO] Model and data loaded successfully.")
    return X_test, y_test, model, encoder


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Generate and save a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — Student Stress Predictor", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] Confusion matrix → {path}")


def plot_feature_importance(model, feature_names, save_dir, top_n=15):
    """
    Plot the top-N most important features.
    Works for tree-based models (Random Forest, XGBoost).
    Skips gracefully if the model doesn't expose feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        print("[SKIP] Feature importance not available for this model type.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        y=np.array(feature_names)[indices][::-1],
        width=importances[indices][::-1],
        color="#4A90D9",
    )
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] Feature importance chart → {path}")


def evaluate_pipeline(feature_names=None):
    """
    End-to-end evaluation pipeline.
    Prints metrics and saves plots to reports/.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  STUDENT STRESS PREDICTOR — EVALUATION REPORT")
    print("=" * 55)

    # ── Load artifacts ──────────────────────────────────────────────────────
    X_test, y_test, model, encoder = load_artifacts()
    class_names = list(encoder.classes_)

    # ── Predictions ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Metrics ─────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\n  Detailed Classification Report:")
    print("  " + "-" * 50)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # ── Confusion matrix ────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred, class_names, REPORTS_DIR)

    # ── Feature importance ──────────────────────────────────────────────────
    if feature_names is None:
        # Generate dummy names if real names not provided
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    plot_feature_importance(model, feature_names, REPORTS_DIR)

    print("\n[DONE] Evaluation complete. Check the 'reports/' folder.\n")
    return acc


if __name__ == "__main__":
    evaluate_pipeline()
