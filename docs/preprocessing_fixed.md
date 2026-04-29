# 🧪 `preprocessing_fixed.ipynb` — Documentation

**Project:** Machine Fault Recognition System  
**Phase:** Phase 2 — Production-Ready Preprocessing Pipeline  
**Compliance:** Aligns with `pipeline.pdf` (Phases 1–10) with targeted overrides justified by EDA findings in `eda_insights.md`.

---

## 📌 Overview

This notebook builds the complete audio preprocessing pipeline that converts raw `.wav` machine recordings into normalized, 3-channel feature tensors ready for model training and inference.

The core output per audio file is a **`[3, 128, 344]` PyTorch tensor** — representing 3 feature channels (Mel-spectrogram + 1st/2nd-order deltas), 128 Mel frequency bins, and 344 time frames.

---

## ⚠️ Key EDA-Justified Design Decisions

The following decisions **override the original `pipeline.pdf` spec**, backed by Exploratory Data Analysis (EDA):

| Decision | `pipeline.pdf` says | EDA finding | What we use |
|---|---|---|---|
| Audio duration | 10 s | M1/M2 have anomalies at the end of 11 s | **11.0 s** |
| Silence trimming | Apply | PROHIBITED — destroys M2 temporal structure (86% silence) | **Removed** |
| DC removal | Apply | DC offset = 0.000 across all files — no-op | **Removed** |
| Peak normalization | Apply | PROHIBITED — erases the RMS fault signal | **Removed** |
| Instance normalization | Per-file | PROHIBITED — use global norm to preserve inter-file energy | **Global stats** |
| Noise augmentation | Gaussian only | Add Pink noise to prevent overfitting to digital silence | **Both types** |

---

## 📦 Dependencies & Constants

### Libraries
```python
torch, torchaudio, librosa, numpy, pandas, matplotlib, tqdm
```
Installed via: `!pip install -q torchaudio librosa tqdm`

### Global Constants

| Constant | Value | Purpose |
|---|---|---|
| `SEED` | `42` | Reproducibility for PyTorch, NumPy, random |
| `TARGET_SR` | `16,000 Hz` | Target sample rate after resampling |
| `TARGET_SECONDS` | `11.0 s` | Audio clip length (EDA override from 10 s) |
| `TARGET_SAMPLES` | `176,000` | `TARGET_SR × TARGET_SECONDS` |
| `N_MELS` | `128` | Mel frequency bins |
| `N_FFT` | `1024` | FFT window size |
| `HOP_LENGTH` | `512` | Spectrogram hop size |
| `F_MIN` / `F_MAX` | `50 / 8000 Hz` | Frequency range of Mel filterbanks |
| `TOP_DB` | `80` | Dynamic range clamp (dB) |
| `T_FRAMES` | `344` | Time frames = `floor((176000 - 1024) / 512) + 1` |
| `N_CHANNELS` | `3` | `Mel + Δ + ΔΔ` (Mono) |

**Final tensor shape:** `[3, 128, 344]` — 132,096 elements per sample.

> **Note on mono data:** EDA §2 confirms 100% of the dataset is mono. We still duplicate the raw audio to `[2, T]` for API compatibility during loading, but the feature extraction uses only channel 0 to produce a unique `[3, 128, 344]` tensor. Adjust your model's `in_channels` accordingly.

---

## 🔄 Pipeline Stages

The pipeline is orchestrated by the **`process_full()`** master function, which runs these stages in order:

```
.wav file
  │
  ▼  Phase 1 ── Load & stereo-fix (mono duplication)
  ▼  Phase 2 ── Resample to 16 kHz
  ▼  Phase 3 ── Waveform augmentations (training only)
  ▼  Phase 4 ── Pad / truncate to 176,000 samples (11 s)
  ▼  Phase 5-6 ── Mel-spectrogram + dB + delta features → [3, 128, 344]
  ▼  Phase 7 ── Global normalisation (if stats provided)
  ▼  Phase 8 ── SpecAugment (training only)
  │
  ▼  [3, 128, 344] tensor
```

---

### Phase 1 — Load & Ingest (`load_audio`, `get_sorted_files`)

- **`get_sorted_files(data_dir)`**: Collects all `.wav` files and sorts them **numerically** (not alphabetically), so `2.wav` comes before `10.wav`. This is critical for matching `results.txt` ordering.
- **`load_audio(path)`**: Loads audio with `torchaudio.load()` and guarantees exactly 2 channels:
  - Mono (1 ch) → duplicated to stereo `[2, T]`
  - More than 2 channels → trimmed to first two

---

### Phase 2 — Resampling (`resample_waveform`)

- Resamples audio to **16,000 Hz** using the `sinc_interp_hann` method for best anti-aliasing quality.
- Passes through unchanged if the source is already 16 kHz.
- Preserves the 0–8 kHz band sufficient for mechanical fault detection.

> **EDA override:** DC offset removal and peak normalization are **not applied** here. DC offset is confirmed 0.0000 across all files, and peak normalization would destroy M3's transient crack spikes and erase RMS fault signal (statistically significant, p=0.0000).

---

### Phase 3 — Waveform Augmentations (`add_gaussian_noise`, `add_pink_noise`, `random_gain`, `time_shift`)

All waveform augmentations are **training only** — completely disabled during inference.

| Augmentation | Function | Details |
|---|---|---|
| Gaussian noise | `add_gaussian_noise` | Random SNR between 15–40 dB; simulates microphone quality variance |
| Pink (1/f) noise | `add_pink_noise` | SNR = 25 dB; prevents memorization of digital silence (86% of dataset is silence) |
| Random gain | `random_gain` | Gain factor in `[0.8, 1.2]`; simulates recording volume variance |
| Time shift | `time_shift` | Circular shift; **max ±0.5 s** for Normal, **max ±0.1 s** for Abnormal (protects M1 onset & M2 boundary events) |

---

### Phase 4 — Length Standardization (`fix_length`)

- Pads or truncates the waveform to exactly **176,000 samples** (11 s).
- **M3 (10 s):** Zero-padded at the **end** to 11 s — avoids shifting onset events.
- **M1/M2 (11 s):** Pass through unchanged.

> **EDA override:** Silence trimming is prohibited. M2's 86% silence ratio is structural — the timing of spin-up/spin-down **is** the anomaly. Trimming would destroy this temporal structure.

---

### Phases 4–6 — Spectral Feature Extraction (`compute_features`)

Transforms a `[2, 176000]` waveform into a `[3, 128, 344]` feature tensor:

1. **Mel-spectrogram** (`torchaudio.transforms.MelSpectrogram`): computes energy spectrogram (`power=2.0`) → `[1, 128, 344]` (using channel 0)
2. **dB conversion** (`torchaudio.transforms.AmplitudeToDB`): applies log-scale with 80 dB dynamic range clamp
3. **Delta features** (`torchaudio.functional.compute_deltas`):
   - `Mel` (static features)
   - `Δ` (1st-order delta — rate of spectral change)
   - `ΔΔ` (2nd-order delta — spectral acceleration)
   
   Stacking these generates 3 channels: `[Mel_ch0, Δ_ch0, ΔΔ_ch0]`

> **EDA §7 requirement:** no smoothing or blurring of spectrograms. **EDA §5 requirement:** Full 128 Mel bands are preserved to capture M1 high-frequency anomalies. Global normalisation (applied after extracting these features) preserves absolute energy differs established in EDA §4.

---

### Phase 7 — Global Normalization (`compute_global_stats`, `load_global_stats`, `global_normalize`)

**Why global, not per-file?**  
EDA §4 proves that absolute RMS energy is a statistically significant fault indicator (p=0.0000). Per-file instance normalization would erase this cross-file energy difference.

**Workflow:**
1. **`compute_global_stats(df_train_normal, save_path)`** — One-time computation using **Welford's online algorithm** (memory-efficient, no need to store all tensors in RAM):
   - Iterates over Normal-class training files only
   - Computes per-channel mean and std over the `(128, 344)` spatial dimensions
   - Saves stats to `norm_stats.npz`
   - Returns a dict with `mean` and `std` tensors of shape `[3, 1, 1]`

2. **`load_global_stats(path)`** — Loads the saved stats from disk. Must be called at both training and inference time.

3. **`global_normalize(feat, stats)`** — Applies `(feat - mean) / (std + 1e-6)` using the globally computed statistics.

> The exact same stats file **must** be used at training and inference to ensure consistency.

---

### Phase 8 — Spectral Augmentations (`spec_augment`, `mixup`)

These augmentations operate on the frequency-domain feature tensors rather than the time-domain waveforms.

| Augmentation | Function | Details |
|---|---|---|
| SpecAugment | `spec_augment` | 2 frequency masks (up to 24 bins) + 2 time masks (up to 40 frames) |
| Mixup | `mixup` | Optional; blends two samples with a `Beta(0.2, 0.2)` coefficient |

---

## ⚖️ Class Imbalance Handling

The dataset has a **4.96:1 Normal:Abnormal imbalance** (EDA §1). Two complementary strategies are implemented:

### `build_class_weights(df, num_classes=6)`
- Computes per-class weights = `1 / class_count`, normalized so they sum to `num_classes`.
- Abnormal classes receive ~5× higher weight than Normal.

### `build_sampler(df, class_weights)`
- Creates a `WeightedRandomSampler` so the `DataLoader` sees an approximately balanced distribution during training.
- Uses `replacement=True`.

### `build_criterion(class_weights, device)`
- Creates a `nn.CrossEntropyLoss` with the per-class weights applied.
- Abnormal classification errors penalized ~5× more heavily.

**Usage:**
```python
class_weights = build_class_weights(df_train)
sampler       = build_sampler(df_train, class_weights)
criterion     = build_criterion(class_weights, device='cuda')
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## ⚙️ Master Run Loop

The notebook run loop is a **two-stage workflow**:

### Stage A — Compute Global Stats (run once)
```python
# Only runs if norm_stats.npz doesn't exist yet
df_normal_train = df[df['state'] == 'normal']
global_stats = compute_global_stats(df_normal_train, save_path=STATS_PATH, max_files=None)
```
- Reads Normal training files from the audit CSV.
- Saves `norm_stats.npz` (mean + std per channel).
- On subsequent runs, loads the existing file instead.

### Stage B — Preprocess All Samples
```python
N_PER_LABEL = 1500  # cap per label for balance
```
- For each file: runs `process_full()` → saves output tensor as `.npy` to `SAMP_DIR`.
- Uses **float16** precision to keep the total output size ~14 GB (within Kaggle's 20 GB limit).
- Skips files already processed (idempotent — safe to restart).
- Errors are collected and reported at the end without aborting the run.

**Output filename format:** `{machine}_{state}_{filename}.npy`

---

## 🌍 Environment Detection

The notebook auto-detects **local vs. Kaggle** and sets paths accordingly:

| Setting | Local | Kaggle |
|---|---|---|
| CSV path | `outputs/full_eda/dataset_audit.csv` | `/kaggle/input/datat-set/dataset_audit.csv` |
| Audio root | (from CSV paths) | `/kaggle/input/cmp27-machine-fault-recognition-full` |
| Output dir | `data/processed/` | `/kaggle/working/data/processed/` |
| Stats file | `norm_stats.npz` | `/kaggle/working/norm_stats.npz` |

**`rebase_path(old_path, new_root)`**: Converts local Windows paths (e.g., `D:\...`) to Kaggle Linux paths by finding the `machine` or `data` path segment and rebasing from there.

---

## ✅ Sanity Check

After preprocessing, the notebook loads a saved `.npy` file and verifies:
- **Shape:** `[3, 128, 344]`
- **dtype:** `float16` (saved) / `float32` (runtime)
- **Mean ≈ 0** and **std ≈ 1** (confirming global normalization worked)
- Visualizes all 3 channels (`Mel`, `Δ`, `ΔΔ`) as spectrograms using matplotlib with the `magma` colormap.

---

## 🚀 `infer.py` Skeleton

The final cell writes an `infer.py` file, which is the **inference-only pipeline** for graded submission:

```python
# Key compliance requirements (pipeline.pdf §2.1):
# - Files processed in strictly increasing numerical order
# - results.txt: one integer per line, no headers
# - time.txt: one float per line, 3 decimal places, excludes I/O time
```

**Inference pipeline** (no augmentations):
1. Load audio (I/O — **excluded from timer**)
2. Start timer
3. Resample → fix length → compute features → global normalize
4. Run model forward pass (`torch.no_grad()`)
5. Stop timer
6. Append prediction and elapsed time

**Output files:**
- `results.txt` — one class integer per line
- `time.txt` — one elapsed time (seconds, 3 decimal places) per line

---

## 📁 Output Files Summary

| File | Description |
|---|---|
| `data/processed/*.npy` | Preprocessed tensors, shape `[3, 128, 344]`, one per audio file |
| `norm_stats.npz` | Global normalization stats: `mean [3,1,1]` and `std [3,1,1]` |
| `infer.py` | Auto-generated inference script for graded submission |
| `results.txt` | (inference output) Predicted class per file |
| `time.txt` | (inference output) Processing time per file |
