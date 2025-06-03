import librosa
import numpy as np


def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=48000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def extract_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=48000)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(spectrogram, ref=np.max)
