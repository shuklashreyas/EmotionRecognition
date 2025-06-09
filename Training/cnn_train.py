# Training/cnn_train.py

import os, sys
# Allow imports from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from multiprocessing import freeze_support
from tqdm import tqdm

# Make sure these point to your packages
from Utils.utils import extract_spectrogram_from_path
from Models.cnn_model import CNNModel, DEVICE

def main():
    # 1. Load CSV & encode labels
    csv_path = os.path.join("Data", "crema_intended_labels.csv")
    df       = pd.read_csv(csv_path)
    le       = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["emotion"])

    os.makedirs("Trained_Models", exist_ok=True)
    joblib.dump(le, os.path.join("Trained_Models", "label_encoder.pkl"))

    # 2. Train/Val split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_idx"], random_state=42
    )

    # 3. Dataset & DataLoader
    class SpectrogramDataset(Dataset):
        def __init__(self, df, root_dir="Data/AudioWAV", sr=48000):
            self.df   = df.reset_index(drop=True)
            self.root = root_dir
            self.sr   = sr

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row       = self.df.iloc[idx]
            wav_file  = os.path.join(self.root, row["path"].split("AudioWAV/")[-1])
            spec      = extract_spectrogram_from_path(wav_file, sr=self.sr)
            spec_tens = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
            label     = int(row["label_idx"])
            return spec_tens, label

    batch_size = 32
    # For debugging, set num_workers=0; you can bump this up once it’s working
    train_loader = DataLoader(SpectrogramDataset(train_df),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(SpectrogramDataset(val_df),
                              batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # 4. Model setup
    num_classes = len(le.classes_)
    model       = CNNModel(num_classes=num_classes).to(DEVICE)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    model_path   = os.path.join("Trained_Models", "cnn.pth")
    epochs       = 20

    # 5. Training loop
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for specs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(specs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * specs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                preds = model(specs).argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved new best model (Val Acc={val_acc:.4f})")

    print("Training complete. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    freeze_support()  # required on Windows/macOS with spawn
    main()
