# Machine Listener System - Final Submission

This folder contains the complete machine sound classification system optimized for performance and accuracy.

## Contents
- `infer.py`: The main inference script meeting the submission requirements.
- `models/`: Contains the trained XGBoost model, scaler, and label encoder.
- `fast_features/`: High-performance Rust extension source for feature extraction.
- `Dockerfile`: Replicates the environment for evaluation.
- `requirements.txt`: Python dependencies.

## How to Run (Local)
1. Ensure Python 3.11+ is installed.
2. Install dependencies: `pip install -r requirements.txt`.
3. (Optional) Build Rust extension for 60x speedup: `cd fast_features && pip install .`.
4. Run inference:
   ```bash
   python infer.py /path/to/data_directory
   ```
5. Check `results.txt` and `time.txt` for outputs.

## How to Run (Docker)
1. Build the image:
   ```bash
   docker build -t machine-listener .
   ```
2. Run container:
   ```bash
   docker run -v /absolute/path/to/data:/data machine-listener /data
   ```

## Label Mapping
| Class | ID |
| :--- | :--- |
| Machine 1, Normal | 0 |
| Machine 1, Abnormal | 1 |
| Machine 2, Normal | 2 |
| Machine 2, Abnormal | 3 |
| Machine 3, Normal | 4 |
| Machine 3, Abnormal | 5 |
