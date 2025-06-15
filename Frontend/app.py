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
# Page configuration and custom CSS
# ----------------------------------------
st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="centered")
st.markdown("""
<style>
  /* Headers */
  .main-header { color: #2196F3; font-size: 2.5em; font-weight: bold; margin-bottom: 0; }
  .subheader  { color: #555555; font-size: 1.1em; margin-top: 0; margin-bottom: 1.5em; }

  /* Section titles */
  .section-header { color: #2196F3; font-size: 1.6em; margin-top: 1em; }

  /* Prediction text */
  .prediction { font-size: 1.8em; color: #FF5722; font-weight: bold; margin-top: 0.5em; }

  /* Recording info */
  .record-info { color: #009688; font-size: 1.1em; }

  /* Tabs styling override */
  .stTabs [role="tab"] { font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🎙️ Emotion Recognition from Voice</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload or record audio and choose a model to see the predicted emotion.</div>", unsafe_allow_html=True)

# ----------------------------------------
# Load Label Encoder once
# ----------------------------------------
@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(PROJECT_ROOT, "Trained_Models", "label_encoder.pkl"))

le = load_label_encoder()

# ----------------------------------------
# Tabs: Upload vs Record
# ----------------------------------------
tab_upload, tab_record = st.tabs(["Upload Audio", "Record Audio"])

# --- Upload Audio Tab ---
with tab_upload:
    st.markdown("<div class='section-header'>📁 Upload Audio</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])

    with col1:
        uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded_file:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            st.audio(tmp_path, format="audio/wav")
    with col2:
        model_choice = st.selectbox("🎛️ Model", ["CNN", "SVM", "MLP", "KNN"], key="upload_model")
        if st.button("Predict Emotion", key="predict_upload"):
            if uploaded_file and model_choice == "CNN":
                with st.spinner("Predicting..."):
                    idx = predict_cnn(tmp_path)
                    emotion = le.inverse_transform([idx])[0]
                st.markdown(f"<div class='prediction'>Predicted: {emotion}</div>", unsafe_allow_html=True)
            else:
                st.info(f"Model `{model_choice}` not yet integrated.")

# --- Record Audio Tab ---
with tab_record:
    st.markdown("<div class='section-header'>🎤 Record Your Own Voice</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])

    # Audio recorder class
    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.sample_rate = None
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame)
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate
            return frame

    with col1:
        ctx = webrtc_streamer(
            key="record_audio",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=AudioRecorder,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        if ctx.state.playing:
            st.markdown("<div class='record-info'>Recording… speak now.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='record-info'>▶️ Click Start to record.</div>", unsafe_allow_html=True)

    with col2:
        model_choice_record = st.selectbox("🎛️ Model", ["CNN", "SVM", "MLP", "KNN"], key="record_model")
        if st.button("Save & Predict"):
            recorder = ctx.audio_processor
            if not recorder or not recorder.frames:
                st.warning("No audio captured. Please grant microphone access and record.")
            else:
                # Assemble numpy audio
                audio_np = np.concatenate([f.to_ndarray().mean(axis=0) for f in recorder.frames])
                sr = recorder.sample_rate or 48000
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, audio_np, sr)
                st.success("Saved recording!")
                st.audio(tmp.name, format="audio/wav")

                if model_choice_record == "CNN":
                    with st.spinner("Predicting..."):
                        idx = predict_cnn(tmp.name)
                        emotion = le.inverse_transform([idx])[0]
                    st.markdown(f"<div class='prediction'>Predicted: {emotion}</div>", unsafe_allow_html=True)
                else:
                    st.info(f"Model `{model_choice_record}` not yet integrated.")
