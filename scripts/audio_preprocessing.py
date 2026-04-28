import librosa
import numpy as np
import os
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

def normalize_audio(y):
    """
    Normalizes the audio to have a consistent volume level (Peak Normalization).
    This handles the issue where sounds from some classes have high volume 
    and others do not, preventing the model from incorrectly learning volume as a classification feature.
    """
    return librosa.util.normalize(y)

def add_noise(y, noise_factor=0.005):
    """
    Data Augmentation / Regularization: Adds white noise to the audio.
    This prevents the model from overfitting to specific clean background conditions,
    acting as a strong regularizer.
    """
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def time_shift(y, shift_max=0.2, sr=22050):
    """
    Data Augmentation: Shifts the audio randomly in time.
    Ensures the model doesn't overfit to the exact temporal placement of sounds.
    """
    shift = np.random.randint(int(sr * shift_max))
    direction = np.random.choice(['right', 'left'])
    if direction == 'right':
        augmented_data = np.roll(y, shift)
    else:
        augmented_data = np.roll(y, -shift)
    return augmented_data

def denoise_audio(y):
    """
    Simple denoising via a Pre-emphasis filter to enhance high frequencies 
    and suppress low-frequency background noise.
    (For advanced spectral gating, consider installing and using the 'noisereduce' python package).
    """
    # Pre-emphasis filter formula: y[t] = x[t] - alpha * x[t-1]
    alpha = 0.97
    y_denoised = np.append(y[0], y[1:] - alpha * y[:-1])
    return y_denoised

def extract_advanced_features(y, sr):
    """
    Extracts advanced audio features for classification.
    This is applied after denoising and normalization.
    """
    features = {}
    try:
        # --- Extract Raw 2D Audio Sequences ---
        # 1. Existing Features
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 2. New Features: Spectral Shape
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # 3. New Feature: Spectral Contrast (very good for machine anomalies)
        # By default, it returns 7 frequency bands.
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # --- Define a Helper for Statistical Expansion ---
        def add_stats(name, array):
            features[f'{name}_mean'] = np.mean(array)
            features[f'{name}_std'] = np.std(array)
            features[f'{name}_max'] = np.max(array)
            features[f'{name}_min'] = np.min(array)
            
        # --- Calculate and Store Stats ---
        add_stats('rms', rms)
        add_stats('zcr', zcr)
        add_stats('centroid', centroid)
        add_stats('rolloff', rolloff)
        add_stats('bandwidth', bandwidth)
        
        for i in range(13):
            add_stats(f'mfcc_{i+1}', mfccs[i])
            
        for i in range(contrast.shape[0]):
            add_stats(f'contrast_{i+1}', contrast[i])
            
        return features
    except Exception as e:
        return {}

def process_file(file_path, apply_augmentation=False):
    """
    Loads a file, applies preprocessing (denoising, normalization),
    optionally creates augmented versions, and extracts features.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Denoising (Suppress background noise)
        y_clean = denoise_audio(y)
        
        # 2. Volume Normalization (Handle varying volumes)
        y_norm = normalize_audio(y_clean)
        
        # 3. Feature Extraction on Clean, Normalized Data
        base_features = extract_advanced_features(y_norm, sr)
        if not base_features:
            return []
            
        # Add metadata
        machine_name = Path(file_path).parent.parent.name
        label = Path(file_path).parent.name
        
        base_features['file_path'] = str(file_path)
        base_features['machine_name'] = machine_name
        base_features['label'] = label
        base_features['is_augmented'] = 0
        
        records = [base_features]
        
        # 4. (Optional) Data Augmentation / Regularization
        # Only augment the training set normally, but here we provide the capability
        if apply_augmentation:
            # Noise augmented
            y_noise = add_noise(y_norm)
            noise_features = extract_advanced_features(y_noise, sr)
            if noise_features:
                noise_features['file_path'] = str(file_path)
                noise_features['machine_name'] = machine_name
                noise_features['label'] = label
                noise_features['is_augmented'] = 1
                records.append(noise_features)
                
            # Time shifted
            y_shifted = time_shift(y_norm, sr=sr)
            shifted_features = extract_advanced_features(y_shifted, sr)
            if shifted_features:
                shifted_features['file_path'] = str(file_path)
                shifted_features['machine_name'] = machine_name
                shifted_features['label'] = label
                shifted_features['is_augmented'] = 2
                records.append(shifted_features)
                
        return records
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    # Target directory to scan
    root_dir = Path("d:/Projects/machine-sound-classification/data/subset_2500")
    
    print("Scanning directory for audio files...")
    wav_files = list(root_dir.rglob("*.wav"))
    
    if not wav_files:
        print("No .wav files found in the specified directory.")
        return
        
    print(f"Found {len(wav_files)} files. Starting processing...")
    
    # Process files in parallel across all CPU cores
    # We set apply_augmentation=True to generate 3x the data (clean, noise-injected, time-shifted)
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(wav_path, apply_augmentation=True) 
        for wav_path in tqdm(wav_files, desc="Processing Audio")
    )
    
    # Flatten the list of lists
    flat_results = [item for sublist in results for item in sublist]
    
    # Convert to DataFrame
    df = pd.DataFrame(flat_results)
    
    # Save the processed features
    output_csv = "d:/Projects/machine-sound-classification/processed_features_augmented.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nProcessing complete. Generated {len(df)} feature records.")
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    main()
