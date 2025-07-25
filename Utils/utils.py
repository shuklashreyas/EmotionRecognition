import io
import numpy as np
import soundfile as sf
import librosa
import cv2


def load_audio(path, sr=48000, duration=None):
    """
    Load an audio file (or file‐like) into a mono float32 array.
    - If `path` is a filename str, reads from disk.
    - If `path` is a file‐like (e.g. BytesIO), soundfile handles it too.
    Resamples to `sr`. If duration is set, clips to duration (in seconds).
    """
    # 1) Read everything
    data, sr_native = sf.read(path, dtype='float32', always_2d=False)
    # 2) To mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # 3) Resample if needed
    if sr_native != sr:
        data = librosa.resample(data, orig_sr=sr_native, target_sr=sr)
    # 4) Trim to duration
    if duration is not None:
        max_samples = int(sr * duration)
        data = data[:max_samples]
    return data, sr

def extract_mfcc_from_path2(path, sr=48000, n_mfcc=13, duration=None, target_length=130):
    """
    Extract MFCCs from an audio file and pad/truncate to `target_length` frames.
    """
    y, _ = load_audio(path, sr=sr, duration=duration)
    # Compute MFCCs: shape (n_mfcc, t)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # pad or truncate time‐axis to target_length
    if mfcc.shape[1] < target_length:
        pad_width = target_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_length]
    return mfcc

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Given a waveform y and sampling rate sr, compute the mean MFCC vector.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def extract_mfcc_from_path(path, sr=48000, n_mfcc=13, duration=None):
    """
    Given a path, load the audio and compute the MFCC vector of it
    """
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
