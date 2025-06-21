import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Utils.utils import extract_spectrogram_from_path
from sklearn.preprocessing import StandardScaler


def main():
   # 1. Load CSV & encode labels
   csv_path = os.path.join("Data", "crema_intended_labels.csv")
   df       = pd.read_csv(csv_path)
   le       = LabelEncoder()
   df["label_idx"] = le.fit_transform(df["emotion"])
  
   os.makedirs("Trained_Models", exist_ok=True)
   joblib.dump(le, os.path.join("Trained_Models", "label_encoder.pkl"))
  
   # Drop duplicate paths just in case
   df = df.drop_duplicates(subset="path")

   # 2. Split based on unique file paths
   train_paths, val_paths = train_test_split(
   df["path"].unique(), test_size=0.2, random_state=5
   )

   # 3. Filter the DataFrame based on the split paths
   train_df = df[df["path"].isin(train_paths)].reset_index(drop=True)
   val_df   = df[df["path"].isin(val_paths)].reset_index(drop=True)

   # 4. Extract features from a DataFrame split. If a cache file exists, load from it. 
   #Otherwise, compute features and save them.
   def extract_features(df_split, cache_path):
      
       if os.path.exists(cache_path):
           print(f"🔁 Loading features from cache: {cache_path}")
           data = np.load(cache_path)
           return data["X"], data["y"] # x is the flattened spectrogram features, y is the integer labels
  
       print(f"⚙️  Extracting features and caching to: {cache_path}")
       X, y = [], []
       for _, row in df_split.iterrows():
           wav_file = os.path.join("Data", "AudioWAV", row["path"].split("AudioWAV/")[-1])
           spec     = extract_spectrogram_from_path(
               wav_file,
               sr=48000,
               n_mels=128,
               target_shape=(128, 128)
           )
           X.append(spec.flatten())
           y.append(row["label_idx"])
  
       X = np.array(X)
       y = np.array(y)
       np.savez(cache_path, X=X, y=y)
       return X, y

  
   # 5. Check for overlap 
   overlap = set(train_df["path"]) & set(val_df["path"])
   print(f"Overlapping files: {len(overlap)}")


   # 6. Load or compute features
   X_train, y_train = extract_features(train_df, "features_train.npz")
   X_val, y_val     = extract_features(val_df, "features_val.npz")


   # # Try different combinations of k, metric, and weights
   # for k in [1, 25, 50, 100]:
   #     for metric in ['euclidean', 'manhattan']:
   #         for weight in ['uniform', 'distance']:
   #             knn = KNeighborsClassifier(
   #                 n_neighbors=k,
   #                 weights=weight,
   #                 metric=metric
   #             )


   #             # Scale features
   #             scaler = StandardScaler()
   #             X_train_scaled = scaler.fit_transform(X_train)
   #             X_val_scaled   = scaler.transform(X_val)


   #             # Fit and evaluate
   #             knn.fit(X_train_scaled, y_train)
   #             val_acc = (knn.predict(X_val_scaled) == y_val).mean()
   #             print(f"k={k}, metric={metric}, weights={weight} → val acc: {val_acc:.4f}")


   # Try different combinations of k, metric, and weights


   knn = KNeighborsClassifier(
   n_neighbors=14,
   weights='uniform',
   metric='manhattan'
   )


   # 7. Scale features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_val_scaled   = scaler.transform(X_val)


   # 8. Fit and evaluate
   knn.fit(X_train_scaled, y_train)
   val_acc = (knn.predict(X_val_scaled) == y_val).mean()
   print(f"k=14, metric= manhattan, weights= uniform → val acc: {val_acc:.4f}")

   # 9. Save the KNN model
   joblib.dump(knn, os.path.join("Trained_Models", "knn.pkl"))
   print("kNN model saved.")
   
   # 10. Print number of training and validation files
   print(f"Number of training files: {len(train_df)}")
   print(f"Number of validation files: {len(val_df)}")


if __name__ == "__main__":
   main()



