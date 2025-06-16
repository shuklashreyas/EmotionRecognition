# Models/cnn_model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.utils import extract_spectrogram_from_path

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base directory of the project
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR  = os.path.join(BASE_DIR, "Trained_Models")
DEFAULT_FP = os.path.join(MODEL_DIR, "cnn.pth")

class CNNModel(nn.Module):
    """
    Simple CNN for emotion classification from 128x128 spectrograms.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 32 * 32, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # -> (B,16,64,64)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B,32,32,32)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_model(
    model_path: str = DEFAULT_FP,
    device=DEVICE
) -> CNNModel:
    """
    Instantiate a CNNModel, load weights from Trained_Models/cnn.pth,
    and return it in eval mode.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find CNN weights at {model_path}")
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(
    audio_path: str,
    model_path: str = DEFAULT_FP,
    device=DEVICE
) -> int:
    """
    Preprocess the audio file into a spectrogram, load the model,
    and return the predicted class index.
    """
    # 1. Load model
    model = load_model(model_path, device)

    # 2. Preprocess to (128×128) mel-spectrogram
    spec = extract_spectrogram_from_path(
        audio_path,
        sr=48000,
        n_mels=128,
        duration=None,
        target_shape=(128, 128)
    )
    tensor = (
        torch.tensor(spec, dtype=torch.float32)
             .unsqueeze(0).unsqueeze(0)
             .to(device)
    )

    # 3. Predict
    with torch.no_grad():
        logits = model(tensor)
        return int(torch.argmax(logits, dim=1).item())
