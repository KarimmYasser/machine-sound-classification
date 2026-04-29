# Phase 1: Data Preprocessing & Augmentation

## 1. Overview
The raw acoustic dataset consists of `.wav` files recorded from different industrial machines operating in various states (Normal and Abnormal). Before any machine learning features can be extracted, the raw audio waveforms must be heavily preprocessed to remove recording inconsistencies, suppress environmental noise, and artificially multiply the dataset to prevent model overfitting.

## 2. Audio Cleaning Pipeline
The cleaning pipeline is executed sequentially on every incoming audio file via the `audio_preprocessing.py` script.

### 2.1. Denoising via Pre-Emphasis Filter
**The Problem:** Factory environments contain pervasive low-frequency background hums. If not removed, the ML models will expend unnecessary capacity analyzing background noise rather than the machine itself.
**The Solution:** A Pre-Emphasis filter is applied directly to the waveform array. 
- **Formula:** `y[t] = x[t] - \alpha * x[t-1]` (where `\alpha = 0.97`)
- **Effect:** This acts as a high-pass filter. It suppresses the low-frequency hums and artificially boosts the high-frequency spectra, which is exactly where subtle mechanical anomalies (like grinding, screeching, or rattling) manifest.

### 2.2. Peak Normalization
**The Problem:** Microphones may have been placed at different distances from the machines, or certain machines simply operate at a higher baseline decibel level. If left unhandled, models will memorize absolute "loudness" as a classification feature, leading to brittle predictions.
**The Solution:** Peak Volume Normalization (`librosa.util.normalize`) scales the entire waveform so that its maximum absolute amplitude is 1.0. This ensures the model learns frequency-based timbral signatures rather than absolute volume.

## 3. Data Augmentation (Regularization)
Deep learning and tree-based models easily overfit to highly consistent datasets. To forcefully stop the model from memorizing the background noise, we synthetically multiply the dataset by injecting targeted variations. This acts as a powerful regularizer.

### 3.1. White Noise Injection
A tiny, randomized amount of Gaussian white noise (`noise_factor=0.005`) is injected across the entire audio file. 
- **Why it works:** By providing the model with "noisy" versions of the exact same audio, the model is penalized for relying on static background conditions. It is forced to look for the loud, distinct mechanical sounds that penetrate the noise.

### 3.2. Time Shifting
The audio array is randomly rolled horizontally (shifted left or right in time) by up to 20% of the sampling rate.
- **Why it works:** Some machines have rhythmic operations (e.g., a pump firing every 1.5 seconds). Time shifting ensures the model does not memorize the *exact* timestamp a sound occurs, but rather learns to identify the sound regardless of its temporal placement in the recording.

## 4. Unified Dataset Consolidation
For production pipelines and lineage tracking, the raw and augmented features are merged into a single structured file.

- **Unified File:** `data/kaggle_unified_features.csv`
- **Lineage Tracking:** Each recording is represented by four versions (`raw`, `denoised`, `added_noise`, `time_shifted`).
- **Detailed Reference:** See [08: Unified Dataset Lineage](08_Unified_Dataset_Lineage.md) for a full breakdown of the unified structure.
