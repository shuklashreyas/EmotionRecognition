import os
import sys
import numpy as np
import pandas as pd
import librosa
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# Local import support
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.mlp_model import MLP, DEVICE, load_model
from Utils.utils import extract_mfcc_from_path

class CremaDataset(Dataset):
    def __init__(self, dataframe, root_dir="Data"):
        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        full_path = os.path.join(self.root_dir, row["path"])

        # Load and trim
        y, sr = librosa.load(full_path, sr=48000, mono=True)
        max_len = sr * 3
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Normalize MFCCs
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        # Flatten to 1D vector and convert to tensor
        x = torch.tensor(mfcc.flatten(), dtype=torch.float32)
        y = torch.tensor(row["label_idx"], dtype=torch.long)

        return x, y

def main():
    print("Loading test data and label encoder...")

    csv_path = os.path.join("Data", "crema_intended_labels.csv")
    df = pd.read_csv(csv_path)

    # Load encoder
    encoder_path = os.path.join("Trained_Models", "label_encoder.pkl")
    label_encode = joblib.load(encoder_path)
    df["label_idx"] = label_encode.transform(df["emotion"])

    # 10% test set
    _, test_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df["label_idx"],
        random_state=42
    )

    test_dataset = CremaDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load MLP model
    print("Loading trained MLP model...")
    model_path = os.path.join("Trained_Models", "mlp.pth")
    mlp_model = load_model(model_path)
    mlp_model.eval()

    # Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = mlp_model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Decode and print metrics
    y_true = label_encode.inverse_transform(all_labels)
    y_pred = label_encode.inverse_transform(all_preds)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix ")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
