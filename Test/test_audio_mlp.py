import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib

from Models.mlp_model import predict as predict_mlp

def main(duration=3, sr=48000):
    print(f"Recording {duration} seconds of audio... Speak now!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    recording = np.squeeze(recording)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, sr)
        tmp_path = tmp.name

    print(f"Saved temporary audio to: {tmp_path}")

    pred_idx = predict_mlp(tmp_path)

    label_encoder_path = os.path.join("Trained_Models", "label_encoder.pkl")
    label_encode = joblib.load(label_encoder_path)
    predicted_emotion = label_encode.inverse_transform([pred_idx])[0]

    print(f" Predicted Emotion: {predicted_emotion}")

if __name__ == "__main__":
    main()
