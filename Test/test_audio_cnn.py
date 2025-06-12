# Test/test_audio_cnn.py

import os
import sys
import tempfile

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib

from Models.cnn_model import predict as predict_cnn

def main(duration=3, sr=48000):
    # 1. Record audio
    print(f"Recording {duration} seconds of audio... Speak now!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # wait until recording is finished
    recording = np.squeeze(recording)

    # 2. Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, sr)
        tmp_path = tmp.name

    print(f"Saved recording to: {tmp_path}")

    # 3. Predict with CNN
    idx = predict_cnn(tmp_path)
    
    # 4. Load label encoder and print emotion
    le = joblib.load(os.path.join("Trained_Models", "label_encoder.pkl"))
    emotion = le.inverse_transform([idx])[0]
    print(f"Predicted emotion: {emotion}")

    # Cleanup (optional)
    # os.remove(tmp_path)

if __name__ == "__main__":
    main()
