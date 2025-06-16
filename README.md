## EmotionRecognition

Build a multiclass emotion‐recognition system that classifies six emotional states—anger, disgust, fear, happiness, neutral, and sadness—from short voice recordings by extracting audio features (e.g., MFCCs and mel-spectrograms) and training four separate models (SVM, KNN, MLP, and CNN) to compare their performance.

Our base model has these accuacies for the four models:
SVM:96%
KNN:48%
MLP:
CNN:98%

## Project Structure

EmotionRecognition/
├── Data/ # Raw audio files and CSV label files, CREAMA-D
├── Utils/ # Feature‐extraction and preprocessing helpers
├── Models/ # Model definitions (e.g., cnn_model.py, svm_model.py, etc.)
├── Training/ # Training scripts for each model
├── Trained_Models/ # Saved model weights (.pth, .pkl)
├── Frontend/ # Streamlit app code
├── Test/ # Unit and integration tests
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions

## Setup

# To create a virtual environment copy and paste these commands into the terminal:
conda create -n emotion-voice python=3.10 -y
conda activate emotion-voice
# Next to install dependencies of our program copy and past these commands into the terminal:
pip install -r requirements.txt
pip install streamlit
pip install streamlit-webrtc

## Running the Program
# To run the program copy and paste the two commands into the terminal:
cd Frontend
streamlit run app.py
# This is where we host the website on your computer once the app.py is ran:
http://localhost:8501
