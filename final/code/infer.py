import os
import sys
import time
import joblib
import numpy as np
import librosa
from pathlib import Path
import warnings
from xgboost import XGBClassifier

# Suppress Sklearn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Constants
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13

LABEL_MAP = {
    "Machine 1_Normal": 0,
    "Machine 1_Abnormal": 1,
    "Machine 2_Normal": 2,
    "Machine 2_Abnormal": 3,
    "Machine 3_Normal": 4,
    "Machine 3_Abnormal": 5
}

def denoise_audio(y):
    alpha = 0.97
    return np.append(y[0], y[1:] - alpha * y[:-1])

def normalize_audio(y):
    return librosa.util.normalize(y)

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py <data_directory>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory.")
        sys.exit(1)

    # 1. Load Assets (Relative to script directory)
    script_dir = Path(__file__).parent
    model = XGBClassifier()
    model.load_model(script_dir / 'models/best_xgb_model.json')
    scaler = joblib.load(script_dir / 'models/scaler.pkl')
    label_encoder = joblib.load(script_dir / 'models/label_encoder.pkl')
    
    # Try to import Rust extension for speed
    fast_feat = None
    try:
        import fast_features
        fast_feat = fast_features
        sys.stderr.write("Engine: Rust (Fast)\n")
    except ImportError:
        sys.stderr.write("Engine: Librosa (Standard fallback)\n")
        pass

    # Get feature order
    feature_cols = list(scaler.feature_names_in_)

    # 2. Collect files and sort them (attempt numeric sort, fallback to string)
    def sort_key(x):
        try:
            return (0, int(x.stem))
        except ValueError:
            return (1, x.stem)
            
    wav_files = sorted(data_dir.glob("*.wav"), key=sort_key)
    
    results = []
    times = []

    # Pre-setup constants
    mel_basis = None
    contrast_bins = None
    last_sr = None

    for i, wav_path in enumerate(wav_files):
        # Progress logging to stderr (so it doesn't pollute results.txt)
        if (i + 1) % 50 == 0 or i == 0 or i == len(wav_files) - 1:
            sys.stderr.write(f"\rProgress: {i+1}/{len(wav_files)} files processed...")
            sys.stderr.flush()

        # Load audio file (timer starts AFTER reading as per 9.2.2)
        y, sr = librosa.load(wav_path, sr=None)
        
        start_time = time.perf_counter()
        
        # Preprocessing
        y_clean = denoise_audio(y)
        y_norm = normalize_audio(y_clean)
        
        # Setup constants if SR changed
        if sr != last_sr:
            mel_basis, contrast_bins = setup_constants(sr)
            last_sr = sr

        # Extraction
        if fast_feat:
            res = fast_feat.extract_features(y_norm.astype(np.float32), float(sr), mel_basis, contrast_bins)
            f_dict = res.features
        else:
            f_dict = get_librosa_features(y_norm, sr)
            
        # Alignment
        vec = [f_dict.get(c, 0.0) for c in feature_cols]
        
        # Predict
        X = scaler.transform([vec])
        pred_idx = model.predict(X)[0]
        
        # End timer
        end_time = time.perf_counter()
        
        # Map label
        str_label = label_encoder.inverse_transform([pred_idx])[0]
        int_label = LABEL_MAP.get(str_label, -1)
        
        results.append(int_label)
        times.append(end_time - start_time)
    
    sys.stderr.write("\n")

    # 3. Output files (detect /app/output for Docker volume support)
    out_dir = Path('/app/output') if Path('/app/output').exists() else Path('.')
    
    with open(out_dir / 'results.txt', 'w') as f:
        for r in results:
            f.write(f"{r}\n")
            
    with open(out_dir / 'time.txt', 'w') as f:
        for t in times:
            f.write(f"{t:.3f}\n")

    print(f"Processed {len(wav_files)} files. Outputs generated at {out_dir.absolute()}")

if __name__ == "__main__":
    main()
