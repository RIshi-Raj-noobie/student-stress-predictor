"""
preprocess.py
=============
Data loading and preprocessing for Student Stress & Mental Health Predictor.
Handles missing values, feature encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os, joblib

# ─── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH      = os.path.join("data", "raw",       "student_stress.csv")
PROCESSED_DIR      = os.path.join("data", "processed")
SCALER_PATH        = os.path.join("models",            "scaler.pkl")
ENCODER_PATH       = os.path.join("models",            "label_encoder.pkl")


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV file into a Pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Drop duplicate rows
      - Fill missing numeric values with column median
      - Fill missing categorical values with mode
    """
    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows")

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("[INFO] Missing values handled.")
    return df


def encode_features(df: pd.DataFrame, target_col: str = "stress_level") -> pd.DataFrame:
    """
    Encode categorical columns using LabelEncoder.
    Save the encoder for reuse during inference.
    """
    os.makedirs("models", exist_ok=True)
    le = LabelEncoder()

    # Encode the target column
    df[target_col] = le.fit_transform(df[target_col])
    joblib.dump(le, ENCODER_PATH)
    print(f"[INFO] Target classes: {list(le.classes_)}")

    # Encode any other object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


def scale_features(X_train: np.ndarray,
                   X_test:  np.ndarray) -> tuple:
    """
    Standardise features to zero mean and unit variance.
    Fit ONLY on training data to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    print("[INFO] Features scaled. Scaler saved.")
    return X_train_scaled, X_test_scaled


def preprocess_pipeline(path: str = RAW_DATA_PATH,
                        target_col: str = "stress_level",
                        test_size: float = 0.2,
                        random_state: int = 42) -> tuple:
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays, ready for model)
    feature_names                      (list of column names)
    """
    # Step 1 – Load
    df = load_data(path)

    # Step 2 – Clean
    df = clean_data(df)

    # Step 3 – Encode labels
    df = encode_features(df, target_col)

    # Step 4 – Split features / target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = list(df.drop(columns=[target_col]).columns)

    # Step 5 – Train / test split (stratified so class balance is preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Step 6 – Scale
    X_train, X_test = scale_features(X_train, X_test)

    # Step 7 – Save processed arrays for notebooks / re-use
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)
    print("[INFO] Processed data saved to data/processed/")

    return X_train, X_test, y_train, y_test, feature_names


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess_pipeline()
