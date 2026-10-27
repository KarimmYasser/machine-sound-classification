# Machine Listener System

Machine Listener System is an audio classification project for machine fault recognition.
The target task is 6-class prediction across machine ID and operating condition:

- 0: Machine 1, Normal
- 1: Machine 1, Abnormal
- 2: Machine 2, Normal
- 3: Machine 2, Abnormal
- 4: Machine 3, Normal
- 5: Machine 3, Abnormal

This repository currently focuses on Phase 1: Exploratory Data Analysis (EDA) and visual inspection of raw audio.

## Team 2

- Abdallah Ayman
- Karim Yasser
- Mohammad Ahmad
- Mohammed Abd-Elaziem

## Repository Layout

```text
.
|- data/
|  |- data_manifast/
|  |  |- train_split.csv
|  |  |- val_split.csv
|  |  \- test_split.csv
|  |- Machine 1/machine_data/{Normal,Abnormal}
|  |- Machine 2/machine_data/{Normal,Abnormal}
|  \- Machine 3/machine_data/{Normal,Abnormal}
|- docs/
|  |- PROJECT_CONTEXT.md
|  |- VISUALIZATION_CONTEXT.md
|  \- Project Findings/initial_strategy.md
|- outputs/
|  \- eda/
\- src/
	\- visualization/
		\- eda_baseline.py
```

## Phase 1 Goal

Use side-by-side visualizations to inspect differences between normal and abnormal machine sounds before defining preprocessing and modeling.

Primary visual diagnostics:

- Waveform (time domain): inspect silence, clipping, and amplitude variation.
- Mel-spectrogram (time-frequency): inspect frequency-band patterns and fault signatures.

## Environment Setup (Conda)

This project is intended to run with Conda.

```bash
conda env create -f environment.yml
conda activate machine-sound-eda
```

## Usage

### 1) Interactive EDA explorer

Browse machines, state (Normal/Abnormal), graph type, and sample index in a single UI.

```bash
python .\src\visualization\eda_baseline.py --interactive
```

### 2) Save baseline comparison figures

Generates waveform and mel-spectrogram comparison images for one machine.

```bash
python .\src\visualization\eda_baseline.py --machine "Machine 1"
```

Optional arguments:

- `--data-root` dataset root directory (default: `data`)
- `--normal-file` specific normal WAV filename/path
- `--abnormal-file` specific abnormal WAV filename/path
- `--sr` target sampling rate (`None` keeps original)
- `--n-mels` mel bins (default: `128`)
- `--hop-length` hop length (default: `512`)
- `--out-dir` output directory (default: `outputs/eda`)

### 3) Prepare Unified Dataset

Before training, consolidate raw and augmented features into a single structured file to ensure lineage tracking and proper grouping:

```bash
python .\scripts\create_unified_dataset.py
```

This generates `data/kaggle_unified_features.csv` which contains 4 versions of every recording: `raw`, `denoised`, `added_noise`, and `time_shifted`.

### 4) Run Full Training Pipeline

To execute the complete end-to-end model training pipeline:

```bash
# Run with full augmentations (default)
python .\scripts\run_training_pipeline.py

# Run without synthetic augmentations (only Raw + Denoised)
python .\scripts\run_training_pipeline.py --no-augmentation
```

This master script sequentially executes:
1. `train_01_feature_engineering.py`: Prunes features to the top 40 and assigns clustering IDs to prevent data leakage.
2. `train_02_model_training.py`: Splits the data securely using `GroupKFold`, fits a `StandardScaler`, and trains a GPU-accelerated `XGBClassifier`.

The best model, scaler, and label encoder are saved to the `outputs/` directory.

## Documentation Index

Detailed technical documentation is available in the `docs/` folder:

- **[01. Data Preprocessing & Augmentation](docs/01_Data_Preprocessing_and_Augmentation.md)**
- **[02. Feature Engineering](docs/02_Feature_Engineering.md)**
- **[06. Clustering & Overfitting Analysis](docs/06_Clustering_and_Overfitting_Analysis.md)**
- **[07. Model Performance Report](docs/07_Model_Performance_Report.md)**
- **[08. Unified Dataset Lineage](docs/08_Unified_Dataset_Lineage.md)**

## EDA Outputs

By default, saved figures are written to `outputs/eda`:

- `machine_1_waveforms.png`
- `machine_1_melspec.png`

## Notes from Current Findings

- Avoid random per-file splitting due to leakage risk from contiguous recording sessions.
- Prefer block/chunk-based splitting to keep adjacent samples in the same split.
- EDA should compare raw vs cleaned audio to verify preprocessing preserves diagnostic frequency content.

## Next Planned Stages

- Robust preprocessing pipeline (silence trimming, denoising, normalization checks).
- Feature pipeline refinement based on EDA evidence.
- Model training/evaluation under course constraints.

## License

No license file has been added yet.
