# Training/svm_train_rbf.py

import os, sys, time, joblib
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

from tqdm import tqdm
from multiprocessing import freeze_support

# Add parent path to import from Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.utils import extract_spectrogram_from_path

def main():
    # 1. Load metadata and encode labels
    csv_path = os.path.join("Data", "crema_intended_labels.csv")
    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["emotion"])
    os.makedirs("Trained_Models", exist_ok=True)
    joblib.dump(le, os.path.join("Trained_Models", "label_encoder.pkl"))

    # 2. Train/Val split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_idx"], random_state=42
    )

    # 3. Feature extraction
    def extract_features(df_split, sr=48000):
        X, y = [], []
        for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
            wav_path = os.path.join("Data", "AudioWAV", row["path"].split("AudioWAV/")[-1])
            spec = extract_spectrogram_from_path(wav_path, sr=sr)
            if spec is not None:
                X.append(spec.flatten())
                y.append(row["label_idx"])
        return np.array(X), np.array(y)

    print("Extracting training features...")
    X_train, y_train = extract_features(train_df)

    print("Extracting validation features...")
    X_val, y_val = extract_features(val_df)

    # 4. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join("Trained_Models", "scaler.pkl"))

    # 5. PCA 
    print("Applying PCA...")
    pca = PCA(n_components=150, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    joblib.dump(pca, os.path.join("Trained_Models", "pca.pkl"))

    print("Training feature shape after PCA:", X_train_pca.shape)
    print("Validation feature shape after PCA:", X_val_pca.shape)

    # 6. Train Non-linear SVM with RBF kernel
    print("Training RBF-kernel SVM...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', verbose=True)
    start_time = time.time()
    svm_clf.fit(X_train_pca, y_train)
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f} seconds.")

    # 7. Evaluate
    y_pred = svm_clf.predict(X_val_pca)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")


    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 8. Save model
    model_path = os.path.join("Trained_Models", "svm.pth")
    joblib.dump(svm_clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    freeze_support()
    main()
