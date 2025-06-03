import streamlit as st
import numpy as np
import soundfile as sf
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av


st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="centered")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎙️ Emotion Recognition from Voice</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload or record your voice to detect the emotion expressed.</p>", unsafe_allow_html=True)


st.markdown("<h3 style='color:#3F51B5;'>📁 Upload an Audio File</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    predicted_emotion = "Happy" 
    st.markdown(
        f"<p style='font-size:22px;'>Predicted Emotion: <strong style='color:#FF5722;'>{predicted_emotion}</strong></p>",
        unsafe_allow_html=True
    )

# ------------------------------
# Voice Recording Section
# ------------------------------
st.markdown("<h3 style='color:#3F51B5;'>🎤 Record Your Voice</h3>", unsafe_allow_html=True)

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.recorded_frames.append(frame)
        return frame

ctx = webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if ctx.audio_processor:
    if st.button("Save Recording"):
        frames = ctx.audio_processor.recorded_frames
        if frames:
            # Convert frames to numpy array
            audio_np = np.concatenate([frame.to_ndarray().mean(axis=0) for frame in frames])
            out_path = "recorded_audio.wav"
            sf.write(out_path, audio_np, 48000)  # 48kHz is default for WebRTC
            st.success("Recording saved as recorded_audio.wav")
            st.audio(out_path, format='audio/wav')

            predicted_emotion = "Calm"  # TODO: Replace with actual model prediction
            st.markdown(
                f"<p style='font-size:22px;'>Predicted Emotion: <strong style='color:#009688;'>{predicted_emotion}</strong></p>",
                unsafe_allow_html=True
            )
        else:
            st.warning("No audio recorded yet.")
