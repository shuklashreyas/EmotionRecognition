# Frontend/app.py

import os
import sys
import tempfile
import io

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
    st.error(
        "Could not import models. Make sure this file is in 'Frontend/' and "
        "'Models/' is in the parent folder."
    )
    st.stop()

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
      color: #D32F2F; /* Red */
      font-weight: bold;
  }
  .record-info.idle {
      color: #5D4037; /* Deep Brown */
  }
  /* Tab styling */
  .stTabs [role="tab"] {
      font-size: 1.2em;
      font-weight: bold;
  }
</style>
""", unsafe_allow_html=True)

def display_prediction_card(emotion: str):
    EMOTION_EMOJI_MAP = {
        "happy": "😄", "sad": "😢", "angry": "😠", "fear": "😨",
        "disgust": "🤢", "neutral": "😐"
    }
    emoji = EMOTION_EMOJI_MAP.get(emotion, "🤔")
    card_html = f"""
    <div style="
        background-color: #FFF8E1;
        border: 2px solid #FFC107;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    ">
        <span style="font-size: 100px;">{emoji}</span>
        <h2 style="
            color: #5D4037;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 15px;
        ">
            Predicted Emotion: {emotion.capitalize()}
        </h2>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# ----------------------------------------
# Load Label Encoder
# ----------------------------------------
@st.cache_resource
def load_label_encoder():
    try:
        path = os.path.join(PROJECT_ROOT, "Trained_Models", "label_encoder.pkl")
        return joblib.load(path)
    except FileNotFoundError:
        st.error("Missing 'label_encoder.pkl' in Trained_Models/")
        st.stop()

le = load_label_encoder()

# ----------------------------------------
# Main UI
# ----------------------------------------
st.markdown("<div class='main-header'>🎙️ Emotion Recognition from Voice</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subheader'>Analyze emotion from audio files or record live audio.</div>",
    unsafe_allow_html=True
)

tab_upload, tab_record = st.tabs(["Upload Audio File", "Record Live Audio"])

# --- Upload Audio Tab ---
with tab_upload:
    col_left, col_right = st.columns([1, 1], gap="large")

    # Left: Upload + playback
    with col_left:
        with st.container():
            st.markdown("<h2 class='section-header'>1. Upload Your Audio</h2>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload a WAV file:", type=["wav"], label_visibility="collapsed")
            audio_path = None
            if uploaded:
                # --- NEW: decode + re-encode as PCM-16 WAV in-memory ---
                raw_bytes = uploaded.read()
                data, sr = sf.read(io.BytesIO(raw_bytes))
                buf = io.BytesIO()
                sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
                buf.seek(0)
                st.audio(buf.read(), format="audio/wav")
                # save for prediction
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(buf.getvalue())
                    audio_path = tmp.name

    # Right: Model selection + predict
    with col_right:
        with st.container():
            st.markdown("<h2 class='section-header'>2. Choose Model & Predict</h2>", unsafe_allow_html=True)
            choice = st.selectbox("Select Model:", ["CNN", "SVM", "MLP", "KNN"], key="upload_model", label_visibility="collapsed")
            if st.button("Analyze Emotion", key="predict_upload", use_container_width=True, type="primary"):
                st.session_state.upload_prediction = None
                if not audio_path:
                    st.warning("Please upload a WAV file first.")
                else:
                    with st.spinner("Analyzing..."):
                        if choice == "CNN":
                            idx = predict_cnn(audio_path)
                        elif choice == "SVM":
                            idx = predict_svm(audio_path)
                        elif choice == "MLP":
                            idx = predict_mlp(audio_path)
                        else:
                            idx = predict_knn(audio_path)
                        st.session_state.upload_prediction = le.inverse_transform([idx])[0]

        with st.container():
            st.markdown("<h2 class='section-header'>Analysis Result</h2>", unsafe_allow_html=True)
            if 'upload_prediction' in st.session_state and st.session_state.upload_prediction:
                display_prediction_card(st.session_state.upload_prediction)
            else:
                st.info("Results will appear here after analysis.")

# --- Record Live Audio Tab ---
with tab_record:
    col_rec_left, col_rec_right = st.columns([1, 1], gap="large")

    # Left: live recording
    with col_rec_left:
        with st.container():
            st.markdown("<h2 class='section-header'>1. Record or Stop</h2>", unsafe_allow_html=True)

            class AudioRecorder(AudioProcessorBase):
                def __init__(self):
                    self.frames = []
                def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                    self.frames.append(frame)
                    return frame

            webrtc_ctx = webrtc_streamer(
                key="recorder",
                mode=WebRtcMode.SENDRECV,
                audio_processor_factory=AudioRecorder,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
            )

            if webrtc_ctx.state.playing:
                st.markdown("<p class='record-info recording'>🔴 Recording... Speak now.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='record-info idle'>▶️ Press Start to begin recording.</p>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<h2 class='section-header'>2. Choose Model & Analyze</h2>", unsafe_allow_html=True)
            choice_r = st.selectbox("Select Model:", ["CNN", "SVM", "MLP", "KNN"], key="record_model", label_visibility="collapsed")
            if st.button("Analyze Recording", key="predict_record", use_container_width=True, type="primary"):
                rec = webrtc_ctx.audio_processor
                st.session_state.record_prediction = None
                if not rec or not rec.frames:
                    st.warning("No audio captured. Please record first.")
                else:
                    # resample for consistent rate
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=44100)
                    processed = [f for frame in rec.frames for f in resampler.resample(frame)]
                    rec.frames.clear()
                    if processed:
                        audio_np = np.concatenate([p.to_ndarray() for p in processed], axis=1).flatten()
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            sf.write(tmp.name, audio_np, 44100, subtype="PCM_16")
                            st.session_state.recorded_audio_path = tmp.name
                        with st.spinner("Analyzing..."):
                            if choice_r == "CNN":
                                idx = predict_cnn(st.session_state.recorded_audio_path)
                            elif choice_r == "SVM":
                                idx = predict_svm(st.session_state.recorded_audio_path)
                            elif choice_r == "MLP":
                                idx = predict_mlp(st.session_state.recorded_audio_path)
                            else:
                                idx = predict_knn(st.session_state.recorded_audio_path)
                            st.session_state.record_prediction = le.inverse_transform([idx])[0]
                    else:
                        st.warning("Audio processing failed; not enough frames.")

    # Right: display last recording & result
    with col_rec_right:
        with st.container():
            st.markdown("<h2 class='section-header'>Analysis Result</h2>", unsafe_allow_html=True)
            if 'record_prediction' in st.session_state and st.session_state.record_prediction:
                st.markdown("### Last Recording:")
                st.audio(st.session_state.recorded_audio_path, format="audio/wav")
                st.divider()
                display_prediction_card(st.session_state.record_prediction)
            else:
                st.info("Your recording and prediction will appear here after analysis.")
