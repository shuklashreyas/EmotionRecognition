import os
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import joblib 

# locally import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.utils import extract_mfcc_from_path2

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_units1: int,
        hidden_units2: int,
        hidden_units3: int,
        num_emotions: int = 6
    ):
        super(MLP, self).__init__()
        # first hidden
        self.hidden_layer1 = nn.Linear(input_features, hidden_units1)
        # second hidden
        self.hidden_layer2 = nn.Linear(hidden_units1, hidden_units2)
        # third hidden
        self.hidden_layer3 = nn.Linear(hidden_units2, hidden_units3)
        # output
        self.output_layer = nn.Linear(hidden_units3, num_emotions)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        return self.output_layer(x)


def load_model(
    model_path: str = "Trained_Models/mlp.pth",
    hidden_units1: int = 727,
    hidden_units2: int = 147,
    hidden_units3: int = 30,
    num_emotions: int = 6,
    device=DEVICE
) -> MLP:
    # load the feature-length
    feat_path = os.path.join("Trained_Models", "input_features.pkl")
    input_features = joblib.load(feat_path)

    # construct and load
    model = MLP(input_features, hidden_units1, hidden_units2, hidden_units3, num_emotions)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def predict(
    audio_path: str,
    model_path: str = "Trained_Models/mlp.pth",
    hidden_units1: int = 727,
    hidden_units2: int = 147,
    hidden_units3: int = 30,
    num_emotions: int = 6,
    device=DEVICE
) -> int:
    # load feature-length
    feat_path = os.path.join("Trained_Models", "input_features.pkl")
    input_features = joblib.load(feat_path)

    # load model
    model = load_model(
        model_path=model_path,
        hidden_units1=hidden_units1,
        hidden_units2=hidden_units2,
        hidden_units3=hidden_units3,
        num_emotions=num_emotions,
        device=device
    )

    # extract MFCC 
    mfcc = extract_mfcc_from_path2(audio_path, sr=48000, n_mfcc=13)
    x = torch.tensor(mfcc.flatten(), dtype=torch.float32).view(1, input_features).to(device)

    # predict
    with torch.no_grad():
        pred_idx = int(model(x).argmax(dim=1).cpu().item())

    return pred_idx