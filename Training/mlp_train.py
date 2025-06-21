import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from multiprocessing import freeze_support
import librosa


# Allow imports from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.mlp_model import MLP
from Utils.utils import extract_mfcc_from_path 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Load CSV & encode labels
    csv_path = os.path.join("Data", "crema_intended_labels.csv") 
    df = pd.read_csv(csv_path) 

    # initialize label encoder
    print("Encoding labels")
    label_encode = LabelEncoder()

    # transform emotions into numbers
    df["label_idx"] = label_encode.fit_transform(df["emotion"])

    # save label encoder
    os.makedirs("Trained_Models", exist_ok=True)
    encoder_path = os.path.join("Trained_Models", "label_encoder.pkl")
    joblib.dump(label_encode, encoder_path)

    # 2. Split data based on label_idx
    print("Splitting data")
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,                
        stratify=df["label_idx"],     
        random_state=42               
    )

    # 3. Create CremaDatasets and DataLoaders
    # datasets
    print("Creating datasets and dataloaders")
    train_dataset = CremaDataset(train_df)
    val_dataset   = CremaDataset(val_df)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    sample_x, _ = train_dataset[0]
    input_features = sample_x.numel()

    feat_path = os.path.join("Trained_Models", "input_features.pkl")
    joblib.dump(input_features, feat_path)
    print(f"Saved input_features = {input_features} → {feat_path}")

    # 4. MLP Model setup
    mlp_model = MLP(input_features, hidden_units1= 727,hidden_units2= 147, hidden_units3=30 ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( mlp_model.parameters(),lr=1e-3, weight_decay=1e-5)   
    target_iterations = 3000
    batch_size = 32
    num_epochs = 30
    losses_mlp = []
    best_val_acc = 0.0

    print(f"\nTraining MLP for {num_epochs} epochs...\n")

    # 5. Training loop
    for epoch in range(num_epochs):
        mlp_model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = mlp_model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Training Loss: {running_loss:.4f}")
        losses_mlp.append(running_loss)

        mlp_model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = mlp_model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_loader.dataset)
        print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.4f}")

        # Get the best accuracy and save the model with the best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(mlp_model.state_dict(), os.path.join("Trained_Models", "mlp.pth"))
            print(f"Saved best model so far! (Val Acc: {val_acc:.4f})")

    # Return the best accuracy
    print("\nTraining finished.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Crema Dataset
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


if __name__ == "__main__":
    freeze_support()  
    main()
