# EDA Insights & Preprocessing Strategy

This document tracks findings from the Phase 1 EDA and translates them into actionable requirements for the Phase 2 Preprocessing and Phase 3 Modeling.

---

## 🟢 1. Class Balance (Finding)
**Data Summary:**
- **Total Audited:** 56,236 files.
- **Normal Samples:** 46,800 files (~83.2%).
  - M1: 16,200 | M2: 16,200 | M3: 14,400
- **Abnormal Samples:** 9,436 files (~16.8%).
  - M1: 3,176 | M2: 3,240 | M3: 3,020
- **Global Imbalance Ratio:** 4.96 (Normal to Abnormal).

**Action for Preprocessing:**
- **Oversampling:** We must implement a PyTorch data loader that oversamples the "Abnormal" class to combat the heavy ~5:1 imbalance.
- **Augmentation:** Apply time-shifting, pitch-shifting, and "mixup" specifically to the Abnormal class to create synthetic samples and reach a 50/50 balance.
- **Weighted Loss:** Use `WeightedCrossEntropyLoss` during training, giving approximately 5x more weight to Abnormal classification errors.

---

## ⚪ 2. Duration & Consistency (Finding)
**Data Summary:**
- **Uniform Channels & Sample Rate:** 100% of the dataset is recorded in Mono (1 channel) at exactly 48,000 Hz. There is zero variance.
- **Discrete Durations:** Audio lengths are completely fixed based on the machine type (Standard deviation is 0.0s within classes). 
    - **Machine 1 & Machine 2:** Exactly 11.0 seconds.
    - **Machine 3:** Exactly 10.0 seconds.

**Action for Preprocessing:**
- **Resampling:** Downsample all audio from 48,000 Hz to 16,000 Hz before generating spectrograms. This drastically reduces memory usage while preserving the necessary frequency bands (up to 8,000 Hz) needed for mechanical fault detection.
- **Length Standardization (Crucial):** CNNs require fixed-size inputs.
    - *Warning:* Because M1 and M2 have critical anomalies at the *very end* of their recordings, **do not truncate** the 11-second files down to 10 seconds. If you do, you will slice off the "shutdown" anomalies.
    - *Solution:* **Pad** the Machine 3 recordings from 10.0s to 11.0s (e.g., by zero-padding/adding silence to the end) to create a uniform temporal dimension for the CNN without destroying evidence of M1/M2 failures.

---

## 🔵 3. Spectral & Spatial Fingerprints (Finding)
**Data Summary:**
- **Machine 1 (Transient Anomalies):** Failures manifest as sharp, short-duration acoustic events (clicks/pops) at the very beginning of the audio, coupled with premature cut-offs at the end.
- **Machine 2 (Phase-Specific Failures):** Heavy acoustic deviations occur exclusively during the machine's spin-up and spin-down phases. The continuous running phase in the middle is acoustically identical to a normal machine.
- **Machine 3 (Harmonic Loss):** M3's abnormal state is characterized by *missing* acoustic energy at specific resonant frequency bands, alongside delayed start/stop timings.

**Action for Preprocessing & Modeling:**
- **Feature Choice:** Time-averaged 1D features (like mean MFCCs) are **strictly prohibited**. Averaging dilutes the phase-specific anomalies. We must use full **2D Mel-Spectrograms** as primary input features.
- **Cropping & Chunking Constraints:** **Blind random cropping must be avoided**. Extracting a chunk from the middle of an abnormal M2 audio file yields a "normal" spectrogram labeled as "abnormal," injecting severe label noise. Chunks must be sequence-aware, or the full 11-second contextual window must be preserved.
- **Model Architecture:** Treat this as a Computer Vision task. Leverage **Convolutional Neural Networks (CNNs)** to efficiently learn the localized 2D spatial structures.

---

## 🟡 4. Energy & Amplitude Variance (Finding)
**Data Summary:**
- **Statistical Significance:** Mann-Whitney U Tests confirm that the difference in RMS Energy between Normal and Abnormal states is highly statistically significant across all machines ($p=0.0000$).
- **Extreme Spikes:** The Abnormal class contains massive RMS energy outliers. Exactly 142 files (0.3% of the dataset) exhibited energy z-scores > 3.

**Action for Preprocessing & Modeling:**
- **Normalization Warning (Critical):** Do **NOT** use per-file Instance Normalization or Min-Max scaling on the spectrograms. Because absolute volume is a proven, statistically significant indicator of a fault, normalizing every file individually to a [0, 1] range will erase the volume differences.
- **Standardization Strategy:** Use **Global Standardization** or **Per-Machine Standardization** (calculating the mean/std across the *entire* normal training set, and applying that to scale the arrays).
- **Outlier Preservation:** Do not clip or remove the 142 extreme high-energy outliers. They represent catastrophic mechanical faults the model needs to learn.

---

## 🟣 5. Frequency Band Profiles (Finding)
**Data Summary:**
- **Machine 1:** Abnormal state exhibits significantly higher energy in the high-frequency range (Mel bands 80–128).
- **Machine 2:** Abnormal state shows a consistent increase in average energy across almost the entire frequency spectrum.
- **Machine 3:** Abnormal state suppresses the sharp, distinct energy peaks (harmonics) normally found in the lower frequency bands (Mel bands 10–60).

**Action for Preprocessing & Modeling:**
- **No Frequency Truncation:** It is absolutely critical to generate and keep the full 128 Mel bands to capture M1's high-frequency anomalies.
- **Model Attention:** A standard 2D CNN with Batch Normalization will naturally learn to weight the high filters for M1 and the low filters for M3.

---

## 🟤 6. Silence Ratio & Background Noise (Finding)
**Data Summary:**
- **Global Silence:** The dataset has a very high overall silence ratio (mean = 0.862), largely driven by Machine 2, which is virtually silent outside of brief spin-up/spin-down phases.
- **Machine 1:** The mechanical fault introduces sustained noise that noticeably reduces the "silent" portion of the audio compared to the Normal state.

**Action for Preprocessing & Modeling:**
- **Silence Trimming Prohibition:** Do **NOT** use automated Voice Activity Detection (VAD) or silence trimming utilities. Trimming silence will compress the audio, misalign the crucial start/end transient anomalies, and destroy the temporal spatial structure.
- **Background Noise Augmentation:** Because files are predominantly silent, the model may overfit to pure digital silence. During training, apply subtle **Gaussian/Pink Noise Injection** to prevent the model from memorizing pristine studio silence.

---

## 🟠 7. Raw Spectrogram Visual Signatures (Finding)
**Data Summary:**
- **Machine 1:** Features a sharp, distinct vertical stripe (a transient pop/impulse) at the onset.
- **Machine 2:** Exhibits highly localized, intense bright "dots" or horizontal dashes (rattles/stutters).
- **Machine 3:** Exhibits a distinct low-frequency **wavy/oscillating pattern** and "smudging" (distortion) of higher harmonic lines.

**Action for Preprocessing & Modeling:**
- **No Image Smoothing:** Do **NOT** apply any form of aggressive 2D smoothing or blurring (e.g., Gaussian Blur) to the spectrogram arrays. Blurring will wash out the sharp rattles and wavy oscillations.
- **Stride/Pooling Strategy:** Ensure initial convolutional layers use small kernel sizes (e.g., 3x3) to capture these micro-textures before applying heavy max-pooling.

---

## 🔴 8. Temporal Envelopes & Amplitude Scaling (Finding)
**Data Summary:**
- **DC Offset Check:** All plotted waveforms show a DC offset of exactly 0.0000. 
- **Extreme M3 Transients:** M3 Abnormal features massive, instantaneous "crack" anomalies (amplitude spiking past 0.04) at the start and end of the cycle, despite the continuous baseline noise being quiet.

**Action for Preprocessing:**
- **Skip DC Offset Removal:** Because the data is naturally centered at zero, safely skip DC offset removal (mean subtraction) in your preprocessing pipeline.
- **Squashing Warning:** Applying standard per-file min-max scaling to M3 will squash the entire baseline signal to near zero just to accommodate the transient spikes. Stick to dataset-level (global) standardization.