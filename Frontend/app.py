# Frontend/app.py

import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
import joblib
import av

from Models.cnn_model import predict as predict_cnn
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎙️ Emotion Recognition from Voice</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Choose a tab below to upload or record audio for emotion prediction.</p>", unsafe_allow_html=True)

# ----------------------------------------
# Load Label Encoder once
# ----------------------------------------
@st.cache_resource
def load_label_encoder():
    encoder_path = os.path.join(PROJECT_ROOT, "Trained_Models", "label_encoder.pkl")
    return joblib.load(encoder_path)

le = load_label_encoder()

# ----------------------------------------
# Tabs: Upload vs Record
# ----------------------------------------
tab_upload, tab_record = st.tabs(["Upload Audio", "Record Audio"])

# --- Upload Audio Tab ---
with tab_upload:
    st.header("📁 Upload an Audio File")
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path, format="audio/wav")

        model_choice = st.selectbox("🎛️ Choose a model:", ["CNN", "SVM", "MLP", "KNN"], key="upload_model")
        if st.button("Predict Emotion", key="predict_upload"):
            if model_choice == "CNN":
                with st.spinner("Predicting..."):
                    idx = predict_cnn(tmp_path)
                    emotion = le.inverse_transform([idx])[0]
                st.success(f"Predicted Emotion (CNN): **{emotion}**")
            else:
                st.info(f"Model `{model_choice}` not yet integrated.")

# --- Record Audio Tab ---
with tab_record:
    st.header("🎤 Record Your Own Voice")

    # Audio processor
    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.sample_rate = None

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame)
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate
            return frame

    ctx = webrtc_streamer(
        key="record_audio",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.info("🎙️ Recording... speak now.")
    else:
        st.info("▶️ Click Start above to begin recording.")

    model_choice_record = st.selectbox("🎛️ Choose a model:", ["CNN", "SVM", "MLP", "KNN"], key="record_model")

    if st.button("Save & Predict Recording"):
        recorder = ctx.audio_processor
        if not recorder or not recorder.frames:
            st.warning("No audio captured. Please click Start and grant mic access.")
        else:
            # Concatenate frames to mono numpy array
            audio_np = np.concatenate([f.to_ndarray().mean(axis=0) for f in recorder.frames])
            sr = recorder.sample_rate or 48000

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_np, sr)
                tmp_path = tmp.name

            st.success("Recording saved!")
            st.audio(tmp_path, format="audio/wav")

            if model_choice_record == "CNN":
                with st.spinner("Predicting..."):
                    idx = predict_cnn(tmp_path)
                    emotion = le.inverse_transform([idx])[0]
                st.success(f"Predicted Emotion (CNN): **{emotion}**")
            else:
                st.info(f"Model `{model_choice_record}` not yet integrated.")
