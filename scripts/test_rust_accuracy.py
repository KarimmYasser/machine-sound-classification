import numpy as np
import librosa
import fast_features
import joblib
from pathlib import Path
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Constants (matching compare_features.py)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13

FEATURE_COLS_FALLBACK = sorted([
    'bandwidth_mean', 'centroid_mean', 'centroid_std',
    'contrast_1_mean', 'contrast_1_std', 'contrast_2_mean', 'contrast_2_std',
    'contrast_3_mean', 'contrast_4_mean', 'contrast_5_mean', 'contrast_6_mean',
    'contrast_7_mean', 'contrast_7_min', 'mfcc_10_mean', 'mfcc_11_mean',
    'mfcc_12_mean', 'mfcc_13_mean', 'mfcc_13_std', 'mfcc_1_mean', 'mfcc_1_std',
    'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_min', 'mfcc_4_max',
    'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std', 'mfcc_6_max',
    'mfcc_6_mean', 'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std',
    'rms_mean', 'rolloff_mean', 'rolloff_std', 'zcr_mean', 'zcr_std'
])

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

def get_librosa_features(y, sr):
    d = {}
    d['rms_mean'] = float(np.mean(librosa.feature.rms(y=y)[0]))
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    d['zcr_mean'], d['zcr_std'] = float(np.mean(zcr)), float(np.std(zcr))
    c = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    d['centroid_mean'], d['centroid_std'] = float(np.mean(c)), float(np.std(c))
    d['bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    d['rolloff_mean'], d['rolloff_std'] = float(np.mean(ro)), float(np.std(ro))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        ix = i + 1
        d[f'mfcc_{ix}_mean'] = float(np.mean(mfcc[i])); d[f'mfcc_{ix}_std'] = float(np.std(mfcc[i]))
        d[f'mfcc_{ix}_min']  = float(np.min(mfcc[i]));  d[f'mfcc_{ix}_max'] = float(np.max(mfcc[i]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(7):
        ix = i + 1
        d[f'contrast_{ix}_mean'] = float(np.mean(contrast[i]))
        d[f'contrast_{ix}_std']  = float(np.std(contrast[i]))
        d[f'contrast_{ix}_min']  = float(np.min(contrast[i]))
    return d

def denoise_audio(y):
    alpha = 0.97
    return np.append(y[0], y[1:] - alpha * y[:-1])

def normalize_audio(y):
    return librosa.util.normalize(y)

def main():
    # 1. Load Assets
    print("Loading model and assets...")
    model = XGBClassifier()
    model.load_model('outputs/best_xgb_model.json')
    scaler = joblib.load('outputs/scaler.pkl')
    le = joblib.load('outputs/label_encoder.pkl')

    if hasattr(scaler, 'feature_names_in_'):
        feature_cols = list(scaler.feature_names_in_)
        print(f"Using {len(feature_cols)} features from scaler metadata.")
    else:
        feature_cols = FEATURE_COLS_FALLBACK
        print("Warning: Scaler has no metadata. Using default sorted columns.")

    data_dir = Path("data/subset_2500")
    all_wavs = sorted(data_dir.rglob("*.wav"))
    
    # Group and sample (300 per class)
    classes = {"Abnormal": [], "Normal": []}
    for w in all_wavs:
        class_name = w.parent.name
        if class_name in classes:
            classes[class_name].append(w)
    
    import random
    random.seed(42)
    wav_files = []
    for class_name, files in classes.items():
        sampled = random.sample(files, min(len(files), 300))
        wav_files.extend(sampled)
    
    n_files = len(wav_files)
    print(f"Sampling complete: {n_files} files (300 per class).")
    print(f"Target Labels: {list(le.classes_)}")

    # 2. Extract Features
    y_true = []
    features_lib = []
    features_rust = []

    # Pre-setup constants
    y_first, sr_first = librosa.load(wav_files[0], sr=None)
    sr_first = float(sr_first)
    mel_basis, contrast_bins = setup_constants(sr_first)

    print("Starting dual-engine feature extraction...")
    for i, wav in enumerate(tqdm(wav_files)):
        class_label = wav.parent.name
        machine_label = wav.parent.parent.name
        fused_label = f"{machine_label}_{class_label}"
        
        try:
            y, sr = librosa.load(wav, sr=None)
            sr = float(sr)
            if sr != sr_first:
                mel_basis, contrast_bins = setup_constants(sr)
                sr_first = sr
            
            # --- Preprocessing ---
            y_clean = denoise_audio(y)
            y_norm = normalize_audio(y_clean)
            
            # --- Extraction ---
            # Librosa
            lib_f_dict = get_librosa_features(y_norm, sr)
            lib_vec = [lib_f_dict.get(c, 0.0) for c in feature_cols]
            
            # Rust
            rust_res = fast_features.extract_features(y_norm.astype(np.float32), sr, mel_basis, contrast_bins)
            rust_f_dict = rust_res.features
            rust_vec = [rust_f_dict.get(c, 0.0) for c in feature_cols]
            
            features_lib.append(lib_vec)
            features_rust.append(rust_vec)
            y_true.append(fused_label)
        except Exception as e:
            print(f"Error on {wav}: {e}")

    # 3. Predict and Compare
    y_true_enc = le.transform(y_true)
    
    # Librosa Accuracy
    X_lib = scaler.transform(features_lib)
    y_pred_lib = model.predict(X_lib)
    acc_lib = accuracy_score(y_true_enc, y_pred_lib)
    
    # Rust Accuracy
    X_rust = scaler.transform(features_rust)
    y_pred_rust = model.predict(X_rust)
    acc_rust = accuracy_score(y_true_enc, y_pred_rust)
    
    print("\n" + "="*40)
    print("PIPELINE PARITY TEST (LIBROSA VS RUST)")
    print("="*40)
    print(f"Samples Tested:   {len(y_true)}")
    print(f"Librosa Accuracy: {acc_lib:.4f}")
    print(f"Rust Accuracy:    {acc_rust:.4f}")
    print(f"Accuracy Drift:   {abs(acc_lib - acc_rust):.4f}")
    print("="*40)

    # Check if predictions are identical
    mismatch = np.sum(y_pred_lib != y_pred_rust)
    print(f"Prediction Mismatches: {mismatch} / {len(y_true)}")

if __name__ == "__main__":
    main()
