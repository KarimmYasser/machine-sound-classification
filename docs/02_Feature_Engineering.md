# Phase 2: Feature Engineering

## 1. Overview
Instead of feeding raw audio directly into Convolutional Neural Networks (which is computationally expensive and slow), we opt for a tabular approach. We extract mathematically dense statistics from the preprocessed audio and feed them into traditional algorithms (XGBoost, Random Forest, SVM). 

The `audio_preprocessing.py` script extracts a rich **104-feature dataset** from every single audio file.

## 2. The Acoustic Features Extracted

### 2.1. Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
MFCCs represent the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. 
- **Purpose:** They capture the "envelope" of the sound, mimicking human auditory perception. They are the most crucial features for identifying the overall "timbre" of the machine.

### 2.2. Spectral Contrast (7 Bands)
Spectral Contrast calculates the decibel difference between peaks (harmonic components) and valleys (non-harmonic/noise components) in the spectrum across 7 frequency bands.
- **Purpose:** Normal machines have steady harmonic peaks (from spinning motors/fans). Anomalies (like grinding or clanking) cause sudden wide-band noise that "flattens" the spectrum. Spectral contrast is incredibly sensitive to this rhythmic flattening.

### 2.3. Spectral Centroid, Rolloff, & Bandwidth
These features describe the overall "shape" of the frequency spectrum.
- **Spectral Centroid:** The "center of mass" of the spectrum (indicates if the sound is "bright" or "dark").
- **Spectral Rolloff:** The frequency bin beneath which 85% of the spectral energy lies.
- **Spectral Bandwidth:** The width of the spectral band.
- **Purpose:** Anomalies often introduce high-frequency screeching, which drastically shifts the centroid higher and pushes the rolloff point up.

### 2.4. Root Mean Square (RMS) & Zero Crossing Rate (ZCR)
- **RMS:** Measures the physical loudness/energy of the audio frame.
- **ZCR:** Measures the rate at which the signal goes from positive to negative (a proxy for noisiness).
- **Purpose:** Because we normalized the audio, the *average* RMS is no longer heavily predictive. However, the standard deviation of the RMS is highly predictive of rhythmic, sudden mechanical knocks.

## 3. Statistical Expansion
Since audio is time-series data, a single `.wav` file produces a 2D matrix for every feature above (e.g., 13 MFCCs calculated over 100 time frames). To flatten this into a 1D tabular row suitable for XGBoost, we calculate four statistical summaries for every feature:
- `Mean`
- `Standard Deviation`
- `Maximum`
- `Minimum`

This flattens the sequence into 104 distinct numeric columns per audio file.
