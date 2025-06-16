# Frontend/app.py

import os
import sys
import tempfile

import streamlit as st
import numpy as np
import soundfile as sf
import joblib
import av

# Allow imports from project root if your structure requires it
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, PROJECT_ROOT)
    from Models.cnn_model import predict as predict_cnn
    from Models.svm_model import predict as predict_svm
    from Models.mlp_model import predict as predict_mlp
    from Models.knn_model import predict as predict_knn
except (ImportError, IndexError):
    st.error("Could not import models. Make sure the file is in the 'Frontend' directory and the 'Models' directory is in the parent folder.")
    st.stop()


from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ----------------------------------------
# Page Config & Styling
# ----------------------------------------
st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="wide")

# Enhanced CSS with the "Warm & Energetic" color palette
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
  /* Section Headers within Tabs */
  .section-header {
      font-size: 2em;
      font-weight: bold;
      color: #5D4037; /* Deep Brown */
      margin-top: 20px;
      margin-bottom: 20px;
  }
  /* Recording Status Indicator */
  .record-info {
      font-size: 1.4em;
      text-align: center;
      margin-bottom: 20px;
  }
  .record-info.recording {
      color: #D32F2F; /* Keep red for recording status */
      font-weight: bold;
  }
  .record-info.idle {
      color: #5D4037; /* Deep Brown */
  }
  /* General text size for labels */
  .st-emotion-cache-16idsys p {
      font-size: 1.1em;
  }
  /* Tab styling */
  .stTabs [role="tab"] {
      font-size: 1.2em;
      font-weight: bold;
  }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Constants & Helper Functions
# ----------------------------------------
TARGET_SAMPLE_RATE = 44100

EMOTION_EMOJI_MAP = {
    "happy": "😄", "sad": "😢", "angry": "😠", "fear": "😨",
    "disgust": "🤢", "neutral": "😐", "calm": "😌", "surprised": "😲"
}

def display_prediction_card(emotion):
    """Displays the predicted emotion in a styled card."""
    emoji = EMOTION_EMOJI_MAP.get(emotion, "🤔")
    card_html = f"""
    <div style="background-color: #FFF8E1; border: 2px solid #FFC107; padding: 30px; border-radius: 15px; text-align: center; margin-top: 20px;">
        <span style="font-size: 100px;">{emoji}</span>
        <h2 style="color: #5D4037; font-size: 2.5em; font-weight: bold; margin-top: 15px;">
            Predicted Emotion: {emotion.capitalize()}
        </h2>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

@st.cache_resource
def load_label_encoder():
    try:
        model_path = os.path.join(PROJECT_ROOT, "Trained_Models", "label_encoder.pkl")
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Label encoder not found. Make sure 'label_encoder.pkl' is in the 'Trained_Models' directory.")
        st.stop()

le = load_label_encoder()

# ----------------------------------------
# Main UI
# ----------------------------------------
st.markdown("<div class='main-header'>🎙️ Emotion Recognition from Voice</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Analyze emotion from uploaded audio files or by recording your own voice in real-time.</div>", unsafe_allow_html=True)

tab_upload, tab_record = st.tabs(["Upload Audio File", "Record Live Audio"])

# --- Upload Audio Tab ---
with tab_upload:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("<h2 class='section-header'>1. Upload Your Audio</h2>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload a WAV file:", type=["wav"], label_visibility="collapsed")
            audio_path = None
            if uploaded:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    audio_path = tmp.name
                st.audio(audio_path, format="audio/wav")

        with st.container(border=True):
             st.markdown("<h2 class='section-header'>2. Choose a Model & Predict</h2>", unsafe_allow_html=True)
             choice = st.selectbox("Select Model:", ["CNN","SVM","MLP","KNN"], key="upload_model", label_visibility="collapsed")
             if st.button("Analyze Emotion", key="predict_upload", use_container_width=True, type="primary"):
                 st.session_state.upload_prediction = None
                 if not audio_path:
                     st.warning("Please upload a WAV file first.")
                 else:
                     with st.spinner("Analyzing..."):
                         if choice == "CNN": idx = predict_cnn(audio_path)
                         elif choice == "SVM": idx = predict_svm(audio_path)
                         elif choice == "MLP": idx = predict_mlp(audio_path)
                         else: idx = predict_knn(audio_path)
                         st.session_state.upload_prediction = le.inverse_transform([idx])[0]

    with col_right:
        with st.container(border=True, height=500):
            st.markdown("<h2 class='section-header'>Analysis Result</h2>", unsafe_allow_html=True)
            if 'upload_prediction' in st.session_state and st.session_state.upload_prediction:
                display_prediction_card(st.session_state.upload_prediction)
            else:
                st.info("The predicted emotion will be displayed here after analysis.")

# --- Record Audio Tab ---
with tab_record:
    col_rec_left, col_rec_right = st.columns([1, 1], gap="large")

    with col_rec_left:
        with st.container(border=True):
            st.markdown("<h2 class='section-header'>1. Record or Stop</h2>", unsafe_allow_html=True)
            class AudioRecorder(AudioProcessorBase):
                def __init__(self): self.frames = []
                def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                    self.frames.append(frame); return frame

            webrtc_ctx = webrtc_streamer(
                key="recorder", mode=WebRtcMode.SENDRECV, audio_processor_factory=AudioRecorder,
                media_stream_constraints={"audio": True, "video": False}, async_processing=True,
            )

            if webrtc_ctx.state.playing:
                st.markdown("<p class='record-info recording'>🔴 Recording... Speak now.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='record-info idle'>▶️ Press Start to begin recording.</p>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<h2 class='section-header'>2. Choose Model & Analyze</h2>", unsafe_allow_html=True)
            choice_r = st.selectbox("Select Model:", ["CNN", "SVM", "MLP", "KNN"], key="record_model", label_visibility="collapsed")
            if st.button("Analyze Recording", key="predict_record", use_container_width=True, type="primary"):
                rec = webrtc_ctx.audio_processor
                st.session_state.record_prediction = None
                if not rec or not rec.frames:
                    st.warning("No audio captured. Please grant mic access and record.")
                else:
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
                    processed_frames = [f for frame in rec.frames for f in resampler.resample(frame)]
                    rec.frames.clear()
                    if processed_frames:
                        audio_np = np.concatenate([p.to_ndarray() for p in processed_frames], axis=1).flatten()
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            sf.write(tmp.name, audio_np, TARGET_SAMPLE_RATE, subtype="PCM_16")
                            st.session_state.recorded_audio_path = tmp.name
                        with st.spinner("Analyzing..."):
                            if choice_r == "CNN": idx = predict_cnn(st.session_state.recorded_audio_path)
                            elif choice_r == "SVM": idx = predict_svm(st.session_state.recorded_audio_path)
                            elif choice_r == "MLP": idx = predict_mlp(st.session_state.recorded_audio_path)
                            else: idx = predict_knn(st.session_state.recorded_audio_path)
                            st.session_state.record_prediction = le.inverse_transform([idx])[0]
                    else:
                        st.warning("Audio processing failed. Not enough frames captured.")

    with col_rec_right:
        with st.container(border=True, height=500):
            st.markdown("<h2 class='section-header'>Analysis Result</h2>", unsafe_allow_html=True)
            if 'record_prediction' in st.session_state and st.session_state.record_prediction:
                st.markdown("### Last Recording:")
                st.audio(st.session_state.recorded_audio_path)
                st.divider()
                display_prediction_card(st.session_state.record_prediction)
            else:
                st.info("Your most recent recording and its predicted emotion will appear here after analysis.")
