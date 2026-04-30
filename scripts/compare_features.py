"""
Three-way feature value comparison:
  A) Librosa       - ground truth
  B) Rust Exact    - current implementation matching Librosa
"""
import numpy as np
import librosa
import scipy.fft
import fast_features
from pathlib import Path
import time

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13

FEATURE_COLS = sorted([
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
    return mel_basis, contrast_bins, freq


def get_librosa_features(y, sr):
    d = {}
    d['rms_mean']       = float(np.mean(librosa.feature.rms(y=y)[0]))
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


if __name__ == "__main__":
    data_dir = Path("data/subset_2500")
    all_wavs = sorted(data_dir.rglob("*.wav"))
    
    # Group by class (Abnormal/Normal)
    classes = {"Abnormal": [], "Normal": []}
    for w in all_wavs:
        class_name = w.parent.name
        if class_name in classes:
            classes[class_name].append(w)
    
    # Sample 300 from each
    import random
    random.seed(42)
    wav_files = []
    for class_name, files in classes.items():
        sampled = random.sample(files, min(len(files), 300))
        wav_files.extend(sampled)
        print(f"Class {class_name}: sampled {len(sampled)}/{len(files)} files.")
    
    if not wav_files:
        print(f"No WAV files found in {data_dir}")
        exit()

    wav_files.sort() # Keep sorted for consistency
    n_files = len(wav_files)
    print(f"Total: {n_files} files. Starting comparison...")

    total_t_lib = 0.0
    total_t_exact = 0.0
    feature_diffs = {col: [] for col in FEATURE_COLS}
    
    # Pre-setup constants using the first file's SR
    y_first, sr_first = librosa.load(wav_files[0], sr=None)
    sr_first = float(sr_first)
    mel_basis, contrast_bins, freq = setup_constants(sr_first)
    
    # Warm up Rust
    _ = fast_features.extract_features(y_first.astype(np.float32), sr_first, mel_basis, contrast_bins)

    try:
        from tqdm import tqdm
        pbar = tqdm(wav_files, desc="Comparing Features")
    except ImportError:
        pbar = wav_files
        print("tqdm not found, falling back to simple loop.")

    for i, wav in enumerate(pbar):
        try:
            y, sr = librosa.load(wav, sr=None)
            sr = float(sr)
            
            if sr != sr_first:
                mel_basis, contrast_bins, freq = setup_constants(sr)
                sr_first = sr

            t0 = time.perf_counter()
            lib_f = get_librosa_features(y, sr)
            total_t_lib += (time.perf_counter() - t0)

            t0 = time.perf_counter()
            exact_result = fast_features.extract_features(y.astype(np.float32), sr, mel_basis, contrast_bins)
            total_t_exact += (time.perf_counter() - t0)
            exact_f = exact_result.features

            for col in FEATURE_COLS:
                lv = lib_f.get(col, 0.0)
                ev = exact_f.get(col, 0.0)
                denom = abs(lv) + 1e-9
                diff_pct = abs(lv - ev) / denom * 100
                feature_diffs[col].append(diff_pct)
        except Exception as e:
            print(f"\nError processing {wav.name}: {e}")

    avg_t_lib = (total_t_lib / n_files) * 1000
    avg_t_exact = (total_t_exact / n_files) * 1000
    speedup = total_t_lib / total_t_exact if total_t_exact > 0 else 0

    print(f"\nSummary for {n_files} files:")
    print(f"Total Librosa Time: {total_t_lib:.2f}s (Avg: {avg_t_lib:.1f}ms)")
    print(f"Total Rust Time:    {total_t_exact:.2f}s (Avg: {avg_t_exact:.1f}ms)")
    print(f"Overall Speedup:    {speedup:.1f}x\n")

    print(f"{'Feature':<25} | {'Mean Exact Δ%':>15} | {'Max Δ%':>10}")
    print("-" * 55)

    for col in FEATURE_COLS:
        mean_err = np.mean(feature_diffs[col])
        max_err = np.max(feature_diffs[col])
        print(f"{col:<25} | {mean_err:>14.4f}% | {max_err:>9.4f}%")
