import librosa
import numpy as np
import os
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Import the preprocessing functions we just created
from audio_preprocessing import normalize_audio, add_noise, time_shift, denoise_audio

def main():
    root_dir = Path("d:/Projects/machine-sound-classification/data/subset_2500")
    output_dir = Path("d:/Projects/machine-sound-classification/preprocessed_audio_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Scanning directory for audio files...")
    wav_files = list(root_dir.rglob("*.wav"))
    
    if not wav_files:
        print("No .wav files found in the specified directory.")
        return

    # Group files by their class (e.g., 'Machine 1_Normal')
    class_files = defaultdict(list)
    for wav_path in wav_files:
        machine_name = wav_path.parent.parent.name
        label = wav_path.parent.name
        # Skip if the directory structure doesn't match expected levels
        if not machine_name.startswith("Machine"):
            continue
        class_key = f"{machine_name}_{label}"
        class_files[class_key].append(wav_path)
        
    print(f"Found {len(class_files.keys())} classes. Extracting 10 samples each...")
    
    for class_key, files in class_files.items():
        print(f"Processing {class_key}...")
        # Take the first 10 files for this class
        samples = files[:10]
        
        # Create a subfolder for this class to keep things organized
        class_out_dir = output_dir / class_key
        class_out_dir.mkdir(exist_ok=True)
        
        for i, file_path in enumerate(tqdm(samples, leave=False)):
            try:
                # Load original
                y, sr = librosa.load(file_path, sr=None)
                
                # 1. Apply Denoising and Normalization
                y_clean = denoise_audio(y)
                y_norm = normalize_audio(y_clean)
                
                # 2. Apply Augmentations (Regularization)
                y_noise = add_noise(y_norm)
                y_shifted = time_shift(y_norm, sr=sr)
                
                # --- Save the files so you can listen to the differences ---
                
                # Save original for comparison
                orig_out = class_out_dir / f"sample_{i+1}_1_original.wav"
                sf.write(orig_out, y, sr)
                
                # Save preprocessed (Denoised + Normalized)
                prep_out = class_out_dir / f"sample_{i+1}_2_preprocessed.wav"
                sf.write(prep_out, y_norm, sr)
                
                # Save augmented - Noise
                noise_out = class_out_dir / f"sample_{i+1}_3_noise_added.wav"
                sf.write(noise_out, y_noise, sr)
                
                # Save augmented - Shifted
                shift_out = class_out_dir / f"sample_{i+1}_4_time_shifted.wav"
                sf.write(shift_out, y_shifted, sr)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"\n✅ All samples successfully processed and saved to:")
    print(f"   {output_dir}")
    print("You can now open these folders and play the .wav files to hear the patterns and preprocessing effects!")

if __name__ == "__main__":
    main()
