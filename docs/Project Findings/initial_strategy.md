# Project Findings & Initial Strategy Ledger

**Project:** Machine Listener System (Fault Recognition)
**Phase:** 1 (Data Preparation & EDA)

## 1. Dataset Characteristics & Splitting Strategy

- **The Data Leakage Threat:** The dataset is not entirely randomized; contiguous audio files belong to specific, continuous recording sessions. A standard random shuffle (`random.shuffle()`) would cause data leakage by placing sequential files (sharing identical background noise) into both Training and Validation sets, resulting in a model that memorizes room noise rather than mechanical faults.
- **The Solution (Block Splitting):** The data must be split chronologically into "blocks" or "chunks." By shuffling blocks instead of individual files, continuous recording sessions are kept isolated.
- **Target Split:** 70% Train, 15% Validation, 15% Test.
- **Structure:** A script has been established to format the Kaggle output directory into `train/`, `val/`, and `test/` folders, each containing subfolders mapped directly to the 6 integer class labels (0 to 5).

## 2. Exploratory Data Analysis (EDA) Protocol

Before any preprocessing logic is written, the raw data must be visually inspected using `librosa`.

- **Waveforms (Time-Domain):** Used to identify leading/trailing silence, detect amplitude clipping, and assess macro-level volume variations.
- **Mel-Spectrograms (Frequency-Domain):** The primary diagnostic tool. Used to identify the "visual fingerprints" of the audio:
  - _Normal Operation:_ Solid horizontal bands and rhythmic symmetry.
  - _Impact Faults:_ Bright vertical spikes spanning multiple frequencies.
  - _Friction/Wear Faults:_ Broadband smearing, "clouds" of static, or shifting high-frequency harmonics.

## 3. Preprocessing Philosophy

- **Order of Operations:** Visualize Raw $\rightarrow$ Apply Cleaning (Silence Trimming/Noise Filtering) $\rightarrow$ Visualize Cleaned.
- **Objective:** Ensure the preprocessing module successfully removes useless variations (like dead air or microphone static) without destroying the critical frequency bands identified during the EDA phase.

## 4. Proposed Architectural Direction

Treating the audio classification task as a Computer Vision problem yields the best results within the project constraints (2 Billion parameters max, balanced scoring for accuracy and inference speed).

- **Feature Representation:** Convert all audio into Mel-Spectrograms (1-channel 2D images).
- **Model Candidates:**
  - **EfficientNet (e.g., B0):** The strongest candidate for the competition, optimizing for both high accuracy and fast running time.
  - **ResNet (18 or 34):** The reliable baseline for establishing initial accuracy metrics.
- **Input Adjustment:** Since pre-trained vision models expect 3 color channels, the 1-channel spectrograms must either be stacked ($1 \times 3$) or the model's initial convolutional layer must be modified to accept `in_channels=1`.

## 5. Next Immediate Step

- Execute the EDA starter code on the newly block-split data in Kaggle to compare the Mel-Spectrogram of a Class 0 (Normal) file against a Class 1 (Abnormal) file to establish a visual baseline.
