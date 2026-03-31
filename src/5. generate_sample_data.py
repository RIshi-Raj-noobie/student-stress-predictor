"""
generate_sample_data.py
=======================
Generates a realistic synthetic dataset for the Student Stress Predictor
when you don't have a real dataset yet.

Run:
    python src/generate_sample_data.py

Output:
    data/raw/student_stress.csv  (1000 student records)

Features
--------
study_hours          : 0–16  hours/day
sleep_hours          : 3–12  hours/night
physical_activity    : 0–14  hrs/week
social_media_hours   : 0–10  hrs/day
extracurricular      : 0 = None, 1 = 1–2, 2 = 3+
financial_stress     : 0 = Low, 1 = Med, 2 = High
relationship_quality : 0 = Poor, 1 = Avg, 2 = Good
diet_quality         : 0 = Poor, 1 = Avg, 2 = Good
attendance_rate      : 40–100 %
cgpa                 : 4.0–10.0

Target
------
stress_level : Low / Medium / High  (derived from a weighted formula)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000   # Number of student records

def generate_dataset(n: int = N) -> pd.DataFrame:
    # ── Raw feature generation ──────────────────────────────────────────────
    study_hours         = np.random.randint(0, 17, n)
    sleep_hours         = np.random.randint(3, 13, n)
    physical_activity   = np.random.randint(0, 15, n)
    social_media_hours  = np.random.randint(0, 11, n)
    extracurricular     = np.random.choice([0, 1, 2], n, p=[0.35, 0.45, 0.20])
    financial_stress    = np.random.choice([0, 1, 2], n, p=[0.40, 0.35, 0.25])
    relationship_qual   = np.random.choice([0, 1, 2], n, p=[0.20, 0.50, 0.30])
    diet_quality        = np.random.choice([0, 1, 2], n, p=[0.25, 0.50, 0.25])
    attendance_rate     = np.random.randint(40, 101, n)
    cgpa                = np.round(np.random.uniform(4.0, 10.0, n), 2)

    # ── Stress scoring formula (domain knowledge-driven) ────────────────────
    # Higher study hours, financial stress, social media → more stress
    # Better sleep, physical activity, diet → less stress
    stress_score = (
        0.25 * study_hours
        - 0.30 * sleep_hours
        - 0.20 * physical_activity
        + 0.15 * social_media_hours
        + 0.25 * financial_stress
        - 0.15 * relationship_qual
        - 0.10 * diet_quality
        + 0.05 * extracurricular
        - 0.02 * (attendance_rate / 10)
        - 0.10 * (cgpa / 2)
        + np.random.normal(0, 0.5, n)   # Add realistic noise
    )

    # ── Bin stress score into 3 classes ─────────────────────────────────────
    low_thresh  = np.percentile(stress_score, 33)
    high_thresh = np.percentile(stress_score, 67)

    stress_level = np.where(
        stress_score <= low_thresh,  "Low",
        np.where(stress_score <= high_thresh, "Medium", "High")
    )

    # ── Assemble DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "study_hours":        study_hours,
        "sleep_hours":        sleep_hours,
        "physical_activity":  physical_activity,
        "social_media_hours": social_media_hours,
        "extracurricular":    extracurricular,
        "financial_stress":   financial_stress,
        "relationship_quality": relationship_qual,
        "diet_quality":       diet_quality,
        "attendance_rate":    attendance_rate,
        "cgpa":               cgpa,
        "stress_level":       stress_level,
    })

    return df


if __name__ == "__main__":
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    df = generate_dataset()
    out_path = os.path.join("data", "raw", "student_stress.csv")
    df.to_csv(out_path, index=False)

    print(f"[INFO] Dataset saved to {out_path}")
    print(f"[INFO] Shape: {df.shape}")
    print("\nClass Distribution:")
    print(df["stress_level"].value_counts())
    print("\nSample Rows:")
    print(df.head())
