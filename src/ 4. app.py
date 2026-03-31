"""
app.py  (Streamlit UI)
======================
Interactive web application for the Student Stress-Level Predictor.

Run:
    streamlit run src/app.py

Features:
  - Enter student lifestyle data via sliders / dropdowns
  - Predict stress level (Low / Medium / High)
  - Display confidence probability bar chart
  - Show top contributing factors
"""

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

# ─── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "best_model.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Stress Predictor 🎓",
    page_icon="🧠",
    layout="centered",
)

# ─── Load Model Artifacts ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

try:
    model, scaler, encoder = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─── Title ──────────────────────────────────────────────────────────────────
st.title("🧠 Student Stress Level Predictor")
st.markdown(
    """
    *Enter your daily habits and academic details below.
    The AI will predict your current stress level and suggest action steps.*
    """
)
st.divider()

# ─── Input Form ─────────────────────────────────────────────────────────────
st.subheader("📋 Tell us about yourself")

col1, col2 = st.columns(2)

with col1:
    study_hours        = st.slider("📚 Study Hours per Day",         0, 16, 6)
    sleep_hours        = st.slider("😴 Sleep Hours per Night",       3, 12, 7)
    physical_activity  = st.slider("🏃 Physical Activity (hrs/week)", 0, 14, 3)
    social_media_hours = st.slider("📱 Social Media Hours per Day",  0, 10, 2)

with col2:
    extracurricular    = st.selectbox("🎭 Extracurricular Activities", ["None", "1–2", "3+"])
    financial_stress   = st.selectbox("💰 Financial Stress Level",     ["Low", "Medium", "High"])
    relationship_qual  = st.selectbox("💬 Relationship Quality",       ["Poor", "Average", "Good"])
    diet_quality       = st.selectbox("🥗 Diet Quality",               ["Poor", "Average", "Good"])

attendance_rate    = st.slider("📅 Class Attendance Rate (%)", 40, 100, 80)
cgpa               = st.slider("🎓 Current CGPA",               4.0, 10.0, 7.5, step=0.1)

st.divider()

# ─── Prediction ─────────────────────────────────────────────────────────────
STRESS_LABELS   = {0: "🟢 Low Stress",  1: "🟡 Medium Stress", 2: "🔴 High Stress"}
STRESS_ADVICE   = {
    0: "Great! You seem to be managing things well. Keep up your healthy habits.",
    1: "Watch out — some areas need attention. Try improving sleep and reducing screen time.",
    2: "High stress detected! Consider talking to a counsellor, taking breaks, and prioritising sleep.",
}
STRESS_COLOURS  = {0: "green", 1: "orange", 2: "red"}

# Encode categorical inputs the same way the model was trained
def encode_cat(val, options):
    return options.index(val)

if st.button("🔮 Predict My Stress Level", type="primary"):
    if not model_loaded:
        st.error("⚠️ Model not found. Please run `train.py` first.")
    else:
        # Build feature vector (must match training feature order)
        features = np.array([[
            study_hours,
            sleep_hours,
            physical_activity,
            social_media_hours,
            encode_cat(extracurricular, ["None", "1–2", "3+"]),
            encode_cat(financial_stress, ["Low", "Medium", "High"]),
            encode_cat(relationship_qual, ["Poor", "Average", "Good"]),
            encode_cat(diet_quality, ["Poor", "Average", "Good"]),
            attendance_rate,
            cgpa,
        ]])

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # ── Display result ───────────────────────────────────────────────
        label  = STRESS_LABELS[prediction]
        colour = STRESS_COLOURS[prediction]

        st.subheader("📊 Prediction Result")
        st.markdown(
            f"<h2 style='color:{colour};'>{label}</h2>",
            unsafe_allow_html=True
        )
        st.info(STRESS_ADVICE[prediction])

        # ── Confidence chart ─────────────────────────────────────────────
        st.subheader("🔍 Confidence Breakdown")
        prob_df = pd.DataFrame({
            "Stress Level": ["Low", "Medium", "High"],
            "Confidence (%)": [round(p * 100, 1) for p in probabilities],
        })
        st.bar_chart(prob_df.set_index("Stress Level"))

        # ── Tips ─────────────────────────────────────────────────────────
        st.subheader("💡 Quick Tips")
        tips = []
        if sleep_hours < 7:
            tips.append("🛏️ Aim for 7–9 hours of sleep every night.")
        if study_hours > 10:
            tips.append("⏱️ Consider Pomodoro technique to avoid burnout.")
        if physical_activity < 3:
            tips.append("🏃 Even a 30-minute walk 3× per week reduces stress significantly.")
        if social_media_hours > 4:
            tips.append("📵 Try a 1-hour daily social-media detox.")
        if not tips:
            tips.append("✅ You're doing great — keep maintaining your healthy routine!")
        for tip in tips:
            st.markdown(f"- {tip}")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("BYOP Capstone Project · AI/ML Department · Built with Streamlit & scikit-learn")
