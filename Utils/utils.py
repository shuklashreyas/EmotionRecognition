import librosa
import numpy as np
import cv2

def load_audio(path, sr=48000, duration=None):
    """
    Load an audio file as a mono waveform.
    - path: file path
    - sr: target sampling rate
    - duration: max seconds to load (None = full)
    """
    y, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y, sr

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Given a waveform y and sampling rate sr, compute the mean MFCC vector.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def extract_mfcc_from_path(path, sr=48000, n_mfcc=13, duration=None):
    y, sr = load_audio(path, sr, duration)
    return extract_mfcc(y, sr, n_mfcc)

def extract_mel_spectrogram(y, sr, n_mels=128):
    """
    Convert waveform to log‐scaled Mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)

def extract_spectrogram_from_path(
    path, sr=48000, n_mels=128, duration=None, target_shape=(128,128)
):
    """
    Load an audio file, compute its Mel spectrogram, normalize,
    and resize to target_shape. Returns a 2D numpy array in [0,1].
    """
    y, sr = load_audio(path, sr, duration)

    # (optional) pad/trim y here if you want fixed-length audio

    # compute Mel spectrogram (in dB)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize to [0,1] with safe divide
    min_val = mel_db.min()
    max_val = mel_db.max()
    diff    = max_val - min_val
    if diff < 1e-6:
        # all values are (almost) the same — produce a zero array
        mel_norm = np.zeros_like(mel_db)
    else:
        mel_norm = (mel_db - min_val) / diff

    # resize to fixed dimensions
    mel_resized = cv2.resize(mel_norm, target_shape, interpolation=cv2.INTER_AREA)
    return mel_resized

def pad_or_trim(y, sr, max_seconds=3):
    """
    Ensure waveform is exactly max_seconds long (pad with zeros or trim).
    """
    max_len = int(sr * max_seconds)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    return y

def extract_mfcc_from_path2(path, sr=48000, n_mfcc=13, max_seconds=3):
    """
    Load audio from path, pad/trim it to max_seconds,
    extract MFCC features, normalize them, and return a flattened array.
    """
    y, sr = load_audio(path, sr)
    y = pad_or_trim(y, sr, max_seconds=max_seconds)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Normalize
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)

    # Flatten
    return mfccs.flatten()
