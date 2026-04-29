# 08: Unified Dataset Lineage

## 1. Overview
To ensure robust model training and facilitate complex analysis of audio processing effects, a unified dataset has been generated: `data/kaggle_unified_features.csv`. This dataset consolidates four distinct versions of every audio recording into a single structured file.

## 2. Processing Variations
For each original `.wav` file, the dataset contains the following four variations, identified by the `processing_type` column:

| Processing Type | Lineage | Description |
| :--- | :--- | :--- |
| **`raw`** | Original | The baseline audio features extracted directly from the raw Kaggle recordings. |
| **`denoised`** | Cleaned | Audio processed with a Pre-Emphasis filter ($\alpha=0.97$) and peak normalization. |
| **`added_noise`** | Augmented | Denoised audio with a small amount of Gaussian white noise injected. |
| **`time_shifted`** | Augmented | Denoised audio with a random temporal shift (roll) applied to the waveform. |

## 3. Dataset Structure
The unified CSV contains 224,944 rows (56,236 unique recordings $\times$ 4 variations).

### Key Columns:
- `file_path`: The path to the original Kaggle source file. This serves as the primary key to group related variations.
- `processing_type`: Categorizes the audio version (`raw`, `denoised`, `added_noise`, `time_shifted`).
- `machine_name` & `label`: Metadata inherited from the source directory structure.
- `is_augmented`: Boolean flag (`True` for `added_noise` and `time_shifted`).
- `rms_mean` ... `contrast_7_min`: 104 acoustic features extracted via Librosa.

## 4. Usage in Pipeline
The `scripts/create_unified_dataset.py` utility is used to generate this file by merging `kaggle_extracted_features_raw.csv` and `kaggle_extracted_features.csv`.

### Benefits:
1. **Lineage Tracking:** Researchers can directly compare feature distributions between raw and denoised versions of the same file.
2. **Leakage Prevention:** Grouping by `file_path` ensures that all versions of a single recording are kept together during cross-validation splits.
3. **Simplicity:** A single source of truth for all feature engineering and training tasks.
