# Frontend/app.py

import os
import sys
import io
import tempfile

import streamlit as st
import numpy as np
import soundfile as sf
import joblib
import av

# 1. Ensure Models/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from Models.cnn_model import predict as predict_cnn
from Models.svm_model import predict as predict_svm
from Models.mlp_model import predict as predict_mlp
from Models.knn_model import predict as predict_knn

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ----------------------------------------
# Page Config & Styling
# ----------------------------------------
st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="wide")
st.markdown("""
<style>
  /* Main Headers */
  .main-header {
      font-size: 3.5em;
      font-weight: bold;
      color: #FFC107; /* Amber */
      text-align: center;
      margin-bottom: 10px;
  }
  .subheader {
      font-size: 1.6em;
      color: #795548; /* Brown */
      text-align: center;
      margin-bottom: 40px;
  }
  /* Section Headers */
  .section-header {
      font-size: 2em;
      font-weight: bold;
      color: #5D4037; /* Deep Brown */
      margin: 20px 0;
  }
  /* Recording Status */
  .record-info {
      font-size: 1.4em;
      text-align: center;
      margin-bottom: 20px;
  }
  .record-info.recording { color: #D32F2F; font-weight: bold; }
  .record-info.idle      { color: #5D4037; }
  /* Tabs */
  .stTabs [role="tab"] { font-size: 1.2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def display_prediction_card(emotion: str):
    EMOTION_EMOJI_MAP = {
        "happy": "😄", "sad": "😢", "angry": "😠", "fear": "😨",
        "disgust": "🤢","neutral": "😐", "calm": "😌", "surprised": "😲"
    }
    emoji = EMOTION_EMOJI_MAP.get(emotion, "🤔")
    st.markdown(f"""
    <div style="
        background-color: #FFF8E1;
        border: 2px solid #FFC107;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;">
      <span style="font-size:100px">{emoji}</span>
      <h2 style="
          color: #5D4037;
          font-size:2.5em;
          font-weight:bold;
          margin-top:15px;">
        Predicted Emotion: {emotion.capitalize()}
      </h2>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------
# Load Label Encoder
# ----------------------------------------
@st.cache_resource
def load_label_encoder():
    path = os.path.join(PROJECT_ROOT, "Trained_Models", "label_encoder.pkl")
    if not os.path.exists(path):
        st.error("Missing 'label_encoder.pkl' in Trained_Models/")
        st.stop()
    return joblib.load(path)

le = load_label_encoder()

# ----------------------------------------
# Main UI
# ----------------------------------------
st.markdown("<div class='main-header'>🎙️ Emotion Recognition from Voice</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Analyze emotion from audio files or record live audio.</div>", unsafe_allow_html=True)

tab_upload, tab_record = st.tabs(["Upload Audio File", "Record Live Audio"])

# --- Upload Audio Tab ---
with tab_upload:
    col_left, col_right = st.columns([1,1], gap="large")

    with col_left:
        st.markdown("<h2 class='section-header'>1. Upload Your Audio</h2>", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["wav"], label_visibility="collapsed")
        audio_path = None
        if uploaded:
            raw = uploaded.read()
            data, sr = sf.read(io.BytesIO(raw))
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            st.audio(buf.read(), format="audio/wav")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(buf.getvalue())
                audio_path = tmp.name

    with col_right:
        st.markdown("<h2 class='section-header'>2. Choose Model & Predict</h2>", unsafe_allow_html=True)
        choice = st.selectbox("", ["CNN","SVM","MLP","KNN"], key="upload_model", label_visibility="collapsed")
        if st.button("Analyze Emotion", use_container_width=True, type="primary"):
            if not audio_path:
                st.warning("Please upload a WAV file first.")
            else:
                with st.spinner("Analyzing..."):
                    idx = {
                        "CNN": predict_cnn,
                        "SVM": predict_svm,
                        "MLP": predict_mlp,
                        "KNN": predict_knn
                    }[choice](audio_path)
                    st.session_state.upload_prediction = le.inverse_transform([idx])[0]

        st.markdown("<h2 class='section-header'>Analysis Result</h2>", unsafe_allow_html=True)
        if st.session_state.get("upload_prediction"):
            display_prediction_card(st.session_state.upload_prediction)
        else:
            st.info("Results will appear here after analysis.")

# --- Record Live Audio Tab ---
with tab_record:
    col_rec_left, col_rec_right = st.columns([1,1], gap="large")

    with col_rec_left:
        st.markdown("<h2 class='section-header'>1. Record or Stop</h2>", unsafe_allow_html=True)

        class AudioRecorder(AudioProcessorBase):
            def __init__(self): self.frames = []
            def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                self.frames.append(frame)
                return frame

        ctx = webrtc_streamer(
            key="recorder", mode=WebRtcMode.SENDRECV,
            audio_processor_factory=AudioRecorder,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        if ctx.state.playing:
            st.markdown("<p class='record-info recording'>🔴 Recording... Speak now.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='record-info idle'>▶️ Press Start to begin recording.</p>", unsafe_allow_html=True)

        st.markdown("<h2 class='section-header'>2. Choose Model & Analyze</h2>", unsafe_allow_html=True)
        choice_r = st.selectbox("", ["CNN","SVM","MLP","KNN"], key="record_model", label_visibility="collapsed")
        if st.button("Analyze Recording", use_container_width=True, type="primary"):
            proc = ctx.audio_processor
            if not proc or not proc.frames:
                st.warning("No audio captured. Please record first.")
            else:
                # resample & mono-mix
                resampler = av.AudioResampler(format="s16", layout="mono", rate=44100)
                frames = [f2 for f in proc.frames for f2 in resampler.resample(f)]
                proc.frames.clear()
                audio_np = np.concatenate([f.to_ndarray() for f in frames], axis=1).flatten()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_np, 44100, subtype="PCM_16")
                    st.session_state.recorded_audio_path = tmp.name

                with st.spinner("Analyzing..."):
                    idx = {
                        "CNN": predict_cnn,
                        "SVM": predict_svm,
                        "MLP": predict_mlp,
                        "KNN": predict_knn
                    }[choice_r](st.session_state.recorded_audio_path)
                    st.session_state.record_prediction = le.inverse_transform([idx])[0]

    with col_rec_right:
        st.markdown("<h2 class='section-header'>Analysis Result</h2>", unsafe_allow_html=True)
        if st.session_state.get("record_prediction"):
            st.markdown("### Last Recording:")
            st.audio(st.session_state.recorded_audio_path, format="audio/wav")
            st.divider()
            display_prediction_card(st.session_state.record_prediction)
        else:
            st.info("Your recording and prediction will appear here after analysis.")
