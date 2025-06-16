# Models/knn_model.py

import os
import joblib
import numpy as np
import functools
from Utils.utils import extract_spectrogram_from_path
from huggingface_hub import hf_hub_download

HF_REPO_ID = "shreyasshukla/emotion-cnn-single"
HF_FILENAME = "knn.pkl"

@functools.lru_cache(maxsize=1)
def load_knn_model(model_path: str = None):
    if model_path is None or not os.path.exists(model_path):
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model",
            token=True
        )
    return joblib.load(model_path)

def predict(audio_path: str, model_path: str = None) -> int:
    knn = load_knn_model(model_path)
    spec = extract_spectrogram_from_path(
        audio_path,
        sr=48000,
        n_mels=128,
        duration=None,
        target_shape=(128, 128)
    )
    features = spec.flatten().reshape(1, -1)
    return int(knn.predict(features)[0])
