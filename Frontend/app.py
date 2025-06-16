# Frontend/app.py

import os
import sys
import tempfile

import streamlit as st
import numpy as np
import soundfile as sf
import joblib
import av

# ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Models.cnn_model import predict      as predict_cnn
from Models.svm_model import predict      as predict_svm
from Models.knn_model import predict      as predict_knn
# from Models.mlp_model import predict    as predict_mlp  # when it's ready

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ----------------------------------------
# Page config & CSS
# ----------------------------------------
st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="centered")
st.markdown("""
<style>
  .main-header   { color:#2196F3; font-size:2.5em; font-weight:bold; margin-bottom:0; }
  .subheader     { color:#555;   font-size:1.1em; margin-top:0; margin-bottom:1.5em; }
  .section-header{ color:#2196F3; font-size:1.6em; margin-top:1em; }
  .prediction    { font-size:1.8em; color:#FF5722; font-weight:bold; margin-top:0.5em; }
  .record-info   { color:#009688; font-size:1.1em; }
  .stTabs [role="tab"] { font-size:1.1em; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🎙️ Emotion Recognition from Voice</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload or record audio and choose a model to see the prediction.</div>", unsafe_allow_html=True)

# ----------------------------------------
# Load label encoder
# ----------------------------------------
@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(PROJECT_ROOT, "Trained_Models", "label_encoder.pkl"))

le = load_label_encoder()

# ----------------------------------------
# Tabs: Upload / Record
# ----------------------------------------
tab_upload, tab_record = st.tabs(["Upload Audio", "Record Audio"])

# --- Upload Audio ---
with tab_upload:
    st.markdown("<div class='section-header'>📁 Upload Audio</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])

    with col1:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(uploaded.read())
            audio_path = tmp.name
            st.audio(audio_path, format="audio/wav")

    with col2:
        choice = st.selectbox("🎛️ Model", ["CNN","SVM","KNN"], key="upload_model")
        if st.button("Predict Emotion", key="predict_upload"):
            if uploaded:
                with st.spinner("Predicting…"):
                    if choice=="CNN":
                        idx = predict_cnn(audio_path)
                    elif choice=="SVM":
                        idx = predict_svm(audio_path)
                    else:  # KNN
                        idx = predict_knn(audio_path)
                    label = le.inverse_transform([idx])[0]
                st.markdown(f"<div class='prediction'>Predicted: {label}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please upload a WAV file first.")

# --- Record Audio ---
with tab_record:
    st.markdown("<div class='section-header'>🎤 Record Your Own Voice</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])

    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.rate   = None
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame)
            if self.rate is None:
                self.rate = frame.sample_rate
            return frame

    with col1:
        ctx = webrtc_streamer(
            key="recorder",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=AudioRecorder,
            media_stream_constraints={"audio":True,"video":False},
            async_processing=True,
        )
        if ctx.state.playing:
            st.markdown("<div class='record-info'>Recording… speak now.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='record-info'>▶️ Click Start to record.</div>", unsafe_allow_html=True)

    with col2:
        choice_r = st.selectbox("🎛️ Model", ["CNN","SVM","KNN"], key="record_model")
        if st.button("Save & Predict"):
            rec = ctx.audio_processor
            if not rec or not rec.frames:
                st.warning("No audio captured. Please grant mic access and record.")
            else:
                data = np.concatenate([f.to_ndarray().mean(axis=0) for f in rec.frames])
                sr   = rec.rate or 48000
                tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, data, sr)
                st.success("Saved recording!")
                st.audio(tmp.name, format="audio/wav")

                with st.spinner("Predicting…"):
                    if choice_r=="CNN":
                        idx = predict_cnn(tmp.name)
                    elif choice_r=="SVM":
                        idx = predict_svm(tmp.name)
                    else:
                        idx = predict_knn(tmp.name)
                    label = le.inverse_transform([idx])[0]
                st.markdown(f"<div class='prediction'>Predicted: {label}</div>", unsafe_allow_html=True)
