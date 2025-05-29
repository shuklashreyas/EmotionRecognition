import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="Voice Emotion Detector", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎙️ Emotion Recognition from Voice</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a .wav file to detect the emotion expressed in the speech.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("🎧 Upload an audio file", type=["wav"])

# Main app logic
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    # (Placeholder for your ML model prediction)
    predicted_emotion = "Happy"  # Replace this with actual prediction result

    # Display result
    st.markdown(
        f"<p style='font-size: 22px; color: #333;'>Predicted Emotion: <strong style='color: #FF5722;'>{predicted_emotion}</strong></p>",
        unsafe_allow_html=True
    )
