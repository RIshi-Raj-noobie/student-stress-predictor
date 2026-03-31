"""
train.py
========
Trains multiple ML models for Student Stress-Level Prediction
and saves the best-performing model to disk.

Models trained:
  1. Random Forest (main model)
  2. XGBoost Classifier
  3. Logistic Regression (baseline)

Best model is selected by validation accuracy.
"""

import numpy as np
import os, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ─── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
RESULTS_PATH    = os.path.join(MODEL_DIR, "training_results.json")


def load_processed_data() -> tuple:
    """Load preprocessed numpy arrays saved by preprocess.py."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    print(f"[INFO] Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """
    Return a dict of model_name → model_instance.

    All hyper-parameters are kept simple and beginner-friendly.
    Comments explain every parameter choice.
    """
    models = {
        # ── Random Forest ──────────────────────────────────────────────────
        # An ensemble of 200 decision trees. Very robust — rarely overfits.
        # Great baseline for tabular data.
        "Random Forest": RandomForestClassifier(
            n_estimators=200,      # Number of trees
            max_depth=None,        # Let trees grow fully (RF handles overfitting via bagging)
            min_samples_split=5,   # Minimum samples needed to split a node
            random_state=42,       # For reproducibility
            n_jobs=-1              # Use all CPU cores
        ),

        # ── XGBoost ────────────────────────────────────────────────────────
        # Gradient boosting — typically beats RF on structured/tabular data.
        # Builds trees sequentially, each correcting the previous one.
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,    # How fast to learn (lower = safer)
            max_depth=6,           # Maximum tree depth
            subsample=0.8,         # Use 80% of data per tree (reduces overfitting)
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        ),

        # ── Logistic Regression ─────────────────────────────────────────────
        # A simple linear model. Useful as a baseline to measure improvement.
        "Logistic Regression": LogisticRegression(
            max_iter=1000,         # Allow enough iterations to converge
            multi_class="auto",
            random_state=42
        ),
    }
    return models


def train_and_evaluate(models: dict,
                       X_train, X_test,
                       y_train, y_test) -> dict:
    """
    Train each model, evaluate on test set, return results dict.
    """
    results = {}

    for name, model in models.items():
        print(f"\n[TRAINING] {name} ...")

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": round(float(acc), 4), "model": model}

        print(f"  ✔ {name} — Accuracy: {acc:.4f}")

    return results


def save_best_model(results: dict) -> str:
    """
    Identify the best model by accuracy, save it, and log results to JSON.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Find best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]

    print(f"\n[RESULT] Best Model: {best_name} "
          f"(Accuracy: {results[best_name]['accuracy']})")

    # Save the model
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"[SAVED]  Model saved → {BEST_MODEL_PATH}")

    # Save results as JSON (without model objects, they aren't JSON-serialisable)
    json_results = {k: {"accuracy": v["accuracy"]} for k, v in results.items()}
    json_results["best_model"] = best_name
    with open(RESULTS_PATH, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"[SAVED]  Results saved → {RESULTS_PATH}")

    return best_name


def train_pipeline():
    """Full training pipeline: load data → build models → train → save best."""
    print("=" * 55)
    print("  STUDENT STRESS PREDICTOR — TRAINING PIPELINE")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_processed_data()
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    best    = save_best_model(results)

    print("\n[DONE] Training complete.\n")
    return best


if __name__ == "__main__":
    train_pipeline()
