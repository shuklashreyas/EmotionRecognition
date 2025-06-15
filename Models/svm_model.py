import os
import sys
import joblib
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.utils import extract_spectrogram_from_path  # updated import path

def load_model(model_path: str = os.path.join("Trained_Models", "svm.pth")):
    model = joblib.load(model_path)
    return model


def predict(
    audio_path: str,
    model_path: str = os.path.join("Trained_Models", "svm.pth"),
    sr=48000
) -> int:
    
    model_dir = os.path.dirname(model_path)
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    pca = joblib.load(os.path.join(model_dir, "pca.pkl"))

    svm = load_model(model_path)

    # Extract spectrogram and flatten
    spec = extract_spectrogram_from_path(audio_path, sr=48000, n_mels=128, target_shape=(128, 128))
    x = spec.flatten().reshape(1, -1)

    # Scale and apply PCA
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    # Predict class index
    pred_idx = svm.predict(x_pca)[0]

    return pred_idx


