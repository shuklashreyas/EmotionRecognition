import os
import joblib
import numpy as np
from Utils.utils import extract_spectrogram_from_path

def load_knn_model(model_path="Trained_Models/knn.pkl"):
    return joblib.load(model_path)

def predict_knn(audio_path, model_path="Trained_Models/knn.pkl"):
    knn = load_knn_model(model_path)
    spec = extract_spectrogram_from_path(audio_path, sr=48000, n_mels=128, target_shape=(128, 128))
    x = spec.flatten().reshape(1, -1)
    pred = knn.predict(x)[0]
    return int(pred)