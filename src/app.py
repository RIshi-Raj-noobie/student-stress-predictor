"""
=============================================================================
app.py  –  Streamlit Dashboard for Construction Site Safety Monitor
=============================================================================
Run with:  streamlit run src/app.py
=============================================================================
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import sys
from pathlib import Path
from datetime import datetime

# Ensure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))
from detect import SafetyDetector, overlay_stats

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SafeSite CV Monitor",
    page_icon   = "🦺",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
  .violation-box {background:#ff4b4b22;border:2px solid #ff4b4b;
                  border-radius:8px;padding:10px;margin:4px 0}
  .safe-box     {background:#00c85322;border:2px solid #00c853;
                 border-radius:8px;padding:10px;margin:4px 0}
  .metric-label {font-size:0.85em;color:#888}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hard-hat.png", width=80)
    st.title("SafeSite CV Monitor")
    st.markdown("---")

    mode = st.radio("Input Mode", ["📷 Upload Image", "🎥 Upload Video", "📡 Webcam (live)"])

    st.markdown("### Model Settings")
    model_path = st.text_input("Model path", "models/pretrained/ppe_yolov8n.pt")
    conf_thr   = st.slider("Confidence threshold", 0.10, 0.95, 0.45, 0.05)
    device     = st.selectbox("Device", ["cpu", "cuda", "mps"])

    st.markdown("---")
    st.caption("Capstone BYOP Project · AIML · 2025")

# ─────────────────────────────────────────────────────────────────────────────
# Cached detector (avoids reloading on every Streamlit re-run)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector(mp, dev):
    return SafetyDetector(model_path=mp, device=dev)

detector = load_detector(model_path, device)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🦺 Construction Site PPE Violation Detector")
st.markdown(
    "Upload a construction-site image or video. "
    "The system detects workers **missing helmets or safety vests** "
    "and flags them as violations in real time."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Mode: Image
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📷 Upload Image":
    uploaded = st.file_uploader("Upload construction site image",
                                type=["jpg","jpeg","png","bmp"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection…"):
            annotated, detections, persons, violations = detector.process_frame(frame)
            annotated = overlay_stats(annotated, persons, violations, fps=0.0)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col2:
            st.subheader("Detection Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Persons Detected", persons)
        m2.metric("PPE Violations",   violations,
                  delta="UNSAFE" if violations else "SAFE",
                  delta_color="inverse")
        m3.metric("Detections Total", len(detections))

        # Detection table
        if detections:
            st.subheader("Detection Details")
            for d in detections:
                icon = "🚨" if d["violation"] else "✅"
                cls  = "violation-box" if d["violation"] else "safe-box"
                st.markdown(
                    f'<div class="{cls}">{icon} <b>{d["label"]}</b> '
                    f'— Confidence: {d["confidence"]:.2%}</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Mode: Video
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "🎥 Upload Video":
    uploaded_vid = st.file_uploader("Upload video file",
                                    type=["mp4","avi","mov","mkv"])
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        stframe  = st.empty()
        progress = st.progress(0)
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_n  = 0

        viol_col, safe_col, fps_col = st.columns(3)
        viol_ph = viol_col.empty()
        safe_ph = safe_col.empty()
        fps_ph  = fps_col.empty()

        t0 = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1
            annotated, _, persons, violations = detector.process_frame(frame)
            fps = frame_n / max(time.time() - t0, 1e-6)
            annotated = overlay_stats(annotated, persons, violations, fps)

            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_column_width=True)
            progress.progress(min(frame_n / total_fr, 1.0))
            viol_ph.metric("Violations", violations)
            safe_ph.metric("Persons",    persons)
            fps_ph.metric("FPS",         f"{fps:.1f}")

        cap.release()
        os.unlink(tfile.name)
        st.success("✅ Video processing complete!")

        summary = detector.get_summary()
        st.info(
            f"📊 **Summary** — Frames processed: {summary['total_frames']} | "
            f"Total violations detected: {summary['total_violations']}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Mode: Webcam
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.warning(
        "⚠️  Webcam mode requires the app to run **locally** with camera access. "
        "Click **Start** to begin streaming."
    )
    start = st.button("▶ Start Webcam")
    stop  = st.button("⏹ Stop")
    frame_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        t0  = time.time()
        fr  = 0
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access webcam.")
                break
            fr += 1
            annotated, _, persons, violations = detector.process_frame(frame)
            fps = fr / max(time.time() - t0, 1e-6)
            annotated = overlay_stats(annotated, persons, violations, fps)
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_column_width=True)
        cap.release()
