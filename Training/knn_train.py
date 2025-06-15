import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Utils.utils import extract_spectrogram_from_path

def main():
    csv_path = os.path.join("Data", "crema_intended_labels.csv")
    df       = pd.read_csv(csv_path)
    le       = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["emotion"])
    
    os.makedirs("Trained_Models", exist_ok=True)
    joblib.dump(le, os.path.join("Trained_Models", "label_encoder.pkl"))
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_idx"], random_state=42)
    
    def extract_features(df_split):
        X, y = [], []
        for _, row in df_split.iterrows():
            wav_file = os.path.join("Data", "AudioWAV", row["path"].split("AudioWAV/")[-1])
            spec     = extract_spectrogram_from_path(wav_file, sr=48000, n_mels=128, target_shape=(128, 128))
            X.append(spec.flatten())
            y.append(row["label_idx"])
        return np.array(X), np.array(y)
    
    X_train, y_train = extract_features(train_df)
    X_val, y_val     = extract_features(val_df)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    val_acc = (knn.predict(X_val) == y_val).mean()
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    joblib.dump(knn, os.path.join("Trained_Models", "knn.pkl"))
    print("kNN model saved.")

if __name__ == "__main__":
    main()
