import streamlit as st
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

# Set page configuration
st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="centered")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎙️ Emotion Recognition from Voice</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload or record your voice to detect the emotion expressed.</p>", unsafe_allow_html=True)


st.markdown("<h3 style='color:#3F51B5;'>📁 Upload an Audio File</h3>",
            unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    model_choice = st.selectbox("🎛️ Choose a model:", [
                                "SVM", "CNN", "MLP", "KNN"], key="upload_model")
    # TODO: Replace with real model predictions
    predicted_emotion = {
        "SVM": "Happy",
        "CNN": "Excited",
        "MLP": "Joyful",
        "KNN": "Content"
    }[model_choice]

    st.markdown(
        f"<p style='font-size:22px;'>Predicted Emotion ({model_choice}): <strong style='color:#FF5722;'>{predicted_emotion}</strong></p>",
        unsafe_allow_html=True
    )

    if st.button("Compare All Models (Upload)", key="compare_upload"):
        comparison_data = {
            "Model": ["SVM", "CNN", "MLP", "KNN"],
            "Predicted Emotion": ["Happy", "Excited", "Joyful", "Content"],
            "Confidence": ["87%", "85%", "82%", "79%"]  # Placeholder values
        }
        st.table(comparison_data)


st.markdown("<h3 style='color:#3F51B5;'>🎤 Record Your Voice</h3>",
            unsafe_allow_html=True)


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
    model_choice_record = st.selectbox(
        "🎛️ Choose a model:", ["SVM", "CNN", "MLP", "KNN"], key="record_model")

    if st.button("Save Recording"):
        frames = ctx.audio_processor.recorded_frames
        if frames:
            # Convert and save
            audio_np = np.concatenate([
                frame.to_ndarray().mean(axis=0)
                for frame in frames
            ])
            out_path = "recorded_audio.wav"
            sf.write(out_path, audio_np, 48000)
            st.success("Recording saved as recorded_audio.wav")
            st.audio(out_path, format='audio/wav')

            predicted_emotion = {
                "SVM": "Calm",
                "CNN": "Relaxed",
                "MLP": "Neutral",
                "KNN": "Peaceful"
            }[model_choice_record]

            st.markdown(
                (
                    f"<p style='font-size:22px;'>Predicted Emotion ({model_choice_record}): "
                    f"<strong style='color:#009688;'>{predicted_emotion}</strong></p>"
                ),
                unsafe_allow_html=True
            )

            if st.button("Compare All Models (Recording)", key="compare_recording"):
                comparison_data = {
                    "Model": ["SVM", "CNN", "MLP", "KNN"],
                    "Predicted Emotion": ["Calm", "Relaxed", "Neutral", "Peaceful"],
                    "Confidence": ["84%", "81%", "80%", "78%"]  # Placeholder
                }
                st.table(comparison_data)
        else:
            st.warning("No audio recorded yet.")
