import os
import gradio as gr
import joblib
import librosa
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

# Load artifacts
def load_assets():
    model = XGBClassifier()
    model.load_model('best_xgb_model.json')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    # Get features from scaler metadata
    features = list(scaler.feature_names_in_)
    return model, scaler, label_encoder, features

# Preprocessing
def denoise_audio(y):
    alpha = 0.97
    return np.append(y[0], y[1:] - alpha * y[:-1])

def normalize_audio(y):
    return librosa.util.normalize(y)

# Feature Extraction (Librosa version for HF compatibility)
def extract_features(y, sr):
    d = {}
    d['rms_mean'] = float(np.mean(librosa.feature.rms(y=y)[0]))
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    d['zcr_mean'], d['zcr_std'] = float(np.mean(zcr)), float(np.std(zcr))
    c = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    d['centroid_mean'], d['centroid_std'] = float(np.mean(c)), float(np.std(c))
    d['bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    d['rolloff_mean'], d['rolloff_std'] = float(np.mean(ro)), float(np.std(ro))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        ix = i + 1
        d[f'mfcc_{ix}_mean'] = float(np.mean(mfcc[i]))
        d[f'mfcc_{ix}_std'] = float(np.std(mfcc[i]))
        d[f'mfcc_{ix}_min'] = float(np.min(mfcc[i]))
        d[f'mfcc_{ix}_max'] = float(np.max(mfcc[i]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(7):
        ix = i + 1
        d[f'contrast_{ix}_mean'] = float(np.mean(contrast[i]))
        d[f'contrast_{ix}_std'] = float(np.std(contrast[i]))
        d[f'contrast_{ix}_min'] = float(np.min(contrast[i]))
    return d

# Try to import Rust extension for speed
FAST_FEAT = None
try:
    import fast_features
    FAST_FEAT = fast_features
    print("Engine: Rust (Fast)")
except ImportError:
    print("Engine: Librosa (Standard fallback)")

# Constants for extraction
N_FFT = 2048
N_MELS = 128

def setup_constants(sr):
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, norm='slaney').astype(np.float32)
    freq = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    fmin, n_bands = 200.0, 6
    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))
    contrast_bins = []
    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        mask = np.logical_and(freq >= f_low, freq <= f_high)
        idx = np.flatnonzero(mask)
        start = (idx[0] - 1 if k > 0 else idx[0])
        start = max(start, 0)
        end = len(freq) if k == n_bands else idx[-1]
        contrast_bins.append(start)
    contrast_bins.append(len(freq))
    return mel_basis, contrast_bins

# Global variables to store assets
MODEL, SCALER, ENCODER, FEATURES = None, None, None, None
MEL_BASIS, CONTRAST_BINS, LAST_SR = None, None, None

def classify_sound(audio_path):
    global MODEL, SCALER, ENCODER, FEATURES, MEL_BASIS, CONTRAST_BINS, LAST_SR
    if MODEL is None:
        MODEL, SCALER, ENCODER, FEATURES = load_assets()
    
    if audio_path is None:
        return "Please upload an audio file."

    # Load and Preprocess
    y, sr = librosa.load(audio_path, sr=None)
    y_clean = denoise_audio(y)
    y_norm = normalize_audio(y_clean)
    
    # Feature Extraction
    if FAST_FEAT:
        if sr != LAST_SR:
            MEL_BASIS, CONTRAST_BINS = setup_constants(sr)
            LAST_SR = sr
        res = FAST_FEAT.extract_features(y_norm.astype(np.float32), float(sr), MEL_BASIS, CONTRAST_BINS)
        raw_features = res.features
    else:
        raw_features = extract_features(y_norm, sr)
    
    # Alignment
    vec = [raw_features.get(c, 0.0) for c in FEATURES]
    
    # Predict
    X_scaled = SCALER.transform([vec])
    pred_idx = MODEL.predict(X_scaled)[0]
    pred_prob = MODEL.predict_proba(X_scaled)[0]
    
    predicted_class = ENCODER.inverse_transform([pred_idx])[0]
    confidence = np.max(pred_prob)
    
    # Prepare result string
    status = "Normal" if "Normal" in predicted_class else "Abnormal"
    machine = predicted_class.split("_")[0]
    
    return {
        "Prediction": predicted_class,
        "Machine": machine,
        "Status": status,
        "Confidence": f"{confidence * 100:.2f}%"
    }

# Create Gradio interface
demo = gr.Interface(
    fn=classify_sound,
    inputs=gr.Audio(type="filepath", label="Upload Machine Sound (.wav)"),
    outputs=gr.JSON(label="Classification Result"),
    title="Acoustic Anomaly Detection System",
    description="Upload a machine sound file to detect anomalies across Machine 1, 2, and 3.",
    examples=[
        # You can add example files here later
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
