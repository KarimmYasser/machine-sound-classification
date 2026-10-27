# Machine Listener System

Machine Listener System is an audio classification project for machine fault recognition.

The target task is 6-class prediction across machine ID and operating condition:

- 0: Machine 1, Normal
- 1: Machine 1, Abnormal
- 2: Machine 2, Normal
- 3: Machine 2, Abnormal
- 4: Machine 3, Normal
- 5: Machine 3, Abnormal

The project combines exploratory audio analysis, preprocessing, feature engineering, clustering-based data splitting, and machine learning and deep learning approaches for machine sound classification.

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
|- scripts/
|  |- create_unified_dataset.py
|  |- run_training_pipeline.py
|  |- train_01_feature_engineering.py
|  \- train_02_model_training.py
\- src/
   \- visualization/
      \- eda_baseline.py
```

## Phase 1: Exploratory Data Analysis (EDA)

Use side-by-side visualizations to inspect differences between normal and abnormal machine sounds before defining preprocessing and modeling.

Primary visual diagnostics:

- **Waveform (time domain):** Inspect silence, clipping, and amplitude variation.
- **Mel-spectrogram (time-frequency):** Inspect frequency-band patterns and fault signatures.

## Environment Setup (Conda)

This project is intended to run with Conda.

```bash
conda env create -f environment.yml
conda activate machine-sound-eda
```

## Usage

### 1) Interactive EDA Explorer

Browse machines, state (Normal/Abnormal), graph type, and sample index in a single UI.

```bash
python .\src\visualization\eda_baseline.py --interactive
```

### 2) Save Baseline Comparison Figures

Generate waveform and mel-spectrogram comparison images for one machine.

```bash
python .\src\visualization\eda_baseline.py --machine "Machine 1"
```

Optional arguments:

- `--data-root`: Dataset root directory (default: `data`)
- `--normal-file`: Specific normal WAV filename/path
- `--abnormal-file`: Specific abnormal WAV filename/path
- `--sr`: Target sampling rate (`None` keeps original)
- `--n-mels`: Mel bins (default: `128`)
- `--hop-length`: Hop length (default: `512`)
- `--out-dir`: Output directory (default: `outputs/eda`)

### 3) Prepare Unified Dataset

Before training, consolidate raw and augmented features into a single structured file to ensure lineage tracking and proper grouping.

```bash
python .\scripts\create_unified_dataset.py
```

This generates `data/kaggle_unified_features.csv`, containing four versions of each recording:

- `raw`
- `denoised`
- `added_noise`
- `time_shifted`

### 4) Run XGBoost Training Pipeline

Execute the end-to-end feature-based machine learning pipeline.

```bash
# Run with full augmentations (default)
python .\scripts\run_training_pipeline.py

# Run without synthetic augmentations (only Raw + Denoised)
python .\scripts\run_training_pipeline.py --no-augmentation
```

The master script sequentially executes:

1. `train_01_feature_engineering.py`: Prunes features to the top 40 and assigns clustering IDs to prevent data leakage.
2. `train_02_model_training.py`: Splits the data using `GroupKFold`, fits a `StandardScaler`, and trains a GPU-accelerated `XGBClassifier`.

The best model, scaler, and label encoder are saved to the `outputs/` directory.

### 5) EfficientNet-B0 Audio Classification

EfficientNet-B0 is also used as a deep learning architecture for the six-class machine sound classification task.

The approach uses mel-spectrogram representations of audio recordings as input to the convolutional neural network, allowing the model to learn time-frequency patterns associated with different machines and their operating conditions.

The classification task covers:

- Machine identification across three machines.
- Normal and abnormal operating condition recognition.
- Six-class prediction combining machine identity and condition.

**Data Leakage Prevention**

A major challenge is the high similarity between recordings, particularly samples originating from related recording sessions or produced through augmentation.

Random splitting can place highly similar samples into training and evaluation sets, producing overly optimistic results.

To address this:

1. Apply clustering-based grouping to identify related audio samples.
2. Assign group identifiers to recordings with similar characteristics.
3. Use group-aware splitting, including `GroupKFold`, to keep related samples together.
4. Evaluate model performance on separated groups to obtain a more reliable estimate of generalization.

This splitting strategy is important for both the feature-based XGBoost approach and the EfficientNet-B0 approach.

## Modeling Approaches

| Model | Input Representation | Purpose |
|---|---|---|
| XGBoost | Engineered audio features | Feature-based classification baseline |
| EfficientNet-B0 | Mel-spectrograms | Deep learning-based sound classification |

Both approaches target the same six machine-condition classes.

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

## Key Challenges and Findings

- **Data leakage:** Highly similar and augmented audio recordings can leak information between training and evaluation sets.
- **Clustering-based splitting:** Group related samples before splitting to reduce leakage and improve evaluation reliability.
- **Audio preprocessing:** Compare raw and cleaned audio to ensure preprocessing preserves diagnostic frequency content.
- **Feature representation:** Explore both engineered audio features and mel-spectrogram representations.
- **Machine fault recognition:** Distinguish machine identity and normal/abnormal operating conditions within a single six-class classification task.

## Next Planned Stages

- Robust preprocessing pipeline (silence trimming, denoising, normalization checks).
- Feature pipeline refinement based on EDA evidence.
- Comparative evaluation of XGBoost and EfficientNet-B0 under consistent data-splitting conditions.
- Model training and evaluation under course constraints.

## License

No license file has been added yet.
