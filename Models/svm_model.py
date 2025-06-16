# Models/svm_model.py

import os
import joblib
import numpy as np
from Utils.utils import extract_spectrogram_from_path

# Compute the project root (one level up from Models/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "Trained_Models")

# Absolute paths to each artifact
SCALER_FP = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_FP    = os.path.join(MODEL_DIR, "pca.pkl")
SVM_FP    = os.path.join(MODEL_DIR, "svm.pth")  # or .pkl if that's what you saved

# Load everything once at import
if not os.path.exists(SCALER_FP):
    raise FileNotFoundError(f"Could not find scaler at {SCALER_FP}")
if not os.path.exists(PCA_FP):
    raise FileNotFoundError(f"Could not find PCA at {PCA_FP}")
if not os.path.exists(SVM_FP):
    raise FileNotFoundError(f"Could not find SVM at {SVM_FP}")

_scaler = joblib.load(SCALER_FP)
_pca    = joblib.load(PCA_FP)
_svm    = joblib.load(SVM_FP)

def predict(
    audio_path: str,
    sr: int = 48000,
    n_mels: int = 128,
    target_shape: tuple[int,int] = (128, 128)
) -> int:
    """
    1) Load audio → mel‐spectrogram (128×128)
    2) Flatten → scale → PCA → SVM.predict
    3) Return the class index
    """
    spec = extract_spectrogram_from_path(
        audio_path,
        sr=sr,
        n_mels=n_mels,
        duration=None,
        target_shape=target_shape
    )  # shape (128,128)

    x = spec.flatten().reshape(1, -1).astype(np.float32)
    x_scaled = _scaler.transform(x)
    x_pca    = _pca.transform(x_scaled)
    pred_idx = int(_svm.predict(x_pca)[0])
    return pred_idx
