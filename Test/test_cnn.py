# Test/test_cnn.py

import os
import sys

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from Utils.utils import extract_spectrogram_from_path
from Models.cnn_model import load_model, DEVICE

#1. Define the Spectrogram Dataset and load the audio spectrograms from the dataset
class SpectrogramDataset(Dataset):
    def __init__(self, df, root_dir="Data/AudioWAV", sr=48000, target_shape=(128,128)):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.sr = sr
        self.target_shape = target_shape

    # return the length of the dataset
    def __len__(self):
        return len(self.df)

    # Get the label associated with the audio spectrogram from the dataset
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Extract relative path after "AudioWAV/"
        rel_path = row["path"].split("AudioWAV/")[-1]
        audio_path = os.path.join(self.root_dir, rel_path)
        # Compute fixed-size spectrogram
        spec = extract_spectrogram_from_path(
            audio_path,
            sr=self.sr,
            n_mels=128,
            duration=None,
            target_shape=self.target_shape
        )
        # (1, 128, 128)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = int(row["label_idx"])
        return spec_tensor, label

def main():
    # 1. Load labels CSV and encode
    csv_path = os.path.join("Data", "crema_intended_labels.csv")
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["emotion"])

    # 2. Split off a 10% test set
    _, test_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df["label_idx"],
        random_state=42
    )

    # 3. Load saved LabelEncoder 
    encoder_path = os.path.join("Trained_Models", "label_encoder.pkl")
    le = joblib.load(encoder_path)

    # 4. Load the CNN model
    model_path = os.path.join("Trained_Models", "cnn.pth")
    model = load_model(model_path, device=DEVICE)

    # 5. Create DataLoader for test set
    test_ds = SpectrogramDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # 6. Run inference
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(DEVICE)
            outputs = model(specs)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    # 7. Decode labels and print metrics
    y_true = le.inverse_transform(all_labels)
    y_pred = le.inverse_transform(all_preds)

    # Print the classification report and confusion matrix based on the model's predictions
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
