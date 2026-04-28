# Phase 5: Model Optimization & Productionization

## 1. Overview
The final phase bridges the gap between Exploratory Data Analysis (EDA) and a production-ready software system. We transitioned the baseline modeling to an optimized, strict pipeline utilizing **XGBoost**, implemented heavy feature pruning for latency optimization, resolved data leakage, and developed a real-time FastAPI inference endpoint.

## 2. Fixing Data Leakage (Advanced Intra-Class Clustering)
**The Problem:** During Phase 1, we artificially multiplied our dataset by injecting noise and time-shifting the audio. While standard grouping handles these augmented copies, the *original* files themselves might still suffer from non-sequential leakage. If a machine produces the exact same rhythmic anomaly continuously over several contiguous recordings, a standard split could place some of those highly correlated recordings in the Training Set and others in the Testing Set. This creates severe non-sequential data leakage, allowing the model to simply memorize the exact sound profile of that specific recording session.

**The Solution:** We implemented an advanced Intra-Class Clustering strategy coupled with Stratified Group Splitting in our training pipeline scripts:
1. We iterate through every class (e.g., `Machine 1_Abnormal`).
2. We run K-Means Clustering (`n_clusters=100` or up to the max available) *within* that specific class on the 40 extracted features. This forms around 300 total highly specific acoustic clusters across the dataset.
3. This guarantees highly correlated and sonically identical audio files are assigned the same globally unique Cluster ID.
4. **Stratified Splitting:** Instead of a global split, we perform a `GroupShuffleSplit` *per class*. This guarantees that exactly 80% of the clusters for every individual class go into Training, and 20% go into Testing, eliminating the risk of a minority class ending up with zero test support.

This guarantees an impermeable barrier: an entire cluster of highly similar sounds goes *entirely* into the Training Set, or *entirely* into the Testing Set. The model is forced to generalize rather than memorizing a specific clustered sound profile.

## 3. Feature Pruning for Inference Speed
Our EDA discovered 104 acoustic features. While highly predictive, computing all 104 features on an edge device (e.g., a low-power IoT sensor on a factory floor) consumes unnecessary CPU cycles and battery power.
- Using a fast Random Forest pass, we evaluated Gini Importances and **pruned the bottom 64 features**.
- We retained only the **Top 40 most critical features** (heavily favoring `mfcc_3`, `mfcc_10`, and `spectral_contrast`). This drastically speeds up the inference latency with minimal loss in predictive power. The remaining features are saved natively to `top_40_features.joblib`.

## 4. XGBoost Migration & Hyperparameter Tuning
To finalize the classifier, we migrated from Random Forest to **XGBoost** due to its superior inference speed and handling of tabular data.
- **Tuning:** A `GridSearchCV` was executed across `max_depth`, `learning_rate`, and `n_estimators`.
- **Handling Class Imbalance:** The "Normal" classes drastically outnumbered the "Abnormal" classes (~10k vs ~2k samples), causing poor recall on anomalous sounds. We resolved this by computing balanced sample weights (`compute_sample_weight('balanced')`) and passing them directly into XGBoost's `fit()` method. This mathematically penalizes mistakes on the minority classes heavily, forcing the model to learn the fault signatures.
- **Optimization Metric:** We optimized strictly for `Macro F1-Score` to ensure the model focuses equally on all classes. 
- **Serialization:** The champion XGBoost model is saved to `champion_xgboost.json` alongside the `StandardScaler` and `LabelEncoder`.

## 5. FastAPI Inference Pipeline (`inference_api.py`)
To deploy this model into a real-world software architecture, we drafted a minimal, async HTTP API using `FastAPI`. 

**The Pipeline Flow:**
1. A client `POST`s a raw `.wav` file to the `/predict` endpoint.
2. The server temporarily saves and loads the audio into memory.
3. **Preprocessing:** It applies the pre-emphasis filter (denoising) and peak volume normalization.
4. **Extraction:** It extracts the raw 104 features and immediately prunes them down to the serialized Top 40.
5. **Scaling & Prediction:** The features are standardized, passed to the XGBoost champion model, and a human-readable prediction (e.g., `Machine 2_Abnormal` with `98.5%` confidence) is returned as a JSON response.

## 6. How to Run the Training Pipeline
To retrain the models on the extracted Kaggle dataset locally, run the master orchestrator script. This guarantees that all steps (from feature pruning to model saving) occur in the correct, leak-proof sequence.

```bash
python .\scripts\run_training_pipeline.py
```

This runs two sub-scripts sequentially:
- `train_01_feature_engineering.py`: Processes `data/kaggle_extracted_features.csv` to prune features and generate clustering IDs.
- `train_02_model_training.py`: Handles model splitting, grid searching, and saving final assets to the `outputs/` folder (`best_xgb_model.json`, `scaler.pkl`, `label_encoder.pkl`).
