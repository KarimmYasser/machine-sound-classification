# Phase 5: Model Optimization & Productionization

## 1. Overview
The final phase bridges the gap between Exploratory Data Analysis (EDA) and a production-ready software system. We transitioned the baseline modeling to an optimized, strict pipeline utilizing **XGBoost**, implemented heavy feature pruning for latency optimization, resolved data leakage, and developed a real-time FastAPI inference endpoint.

## 2. Fixing Data Leakage (Advanced Intra-Class Clustering)
**The Problem:** During Phase 1, we artificially multiplied our dataset by injecting noise and time-shifting the audio. While standard grouping handles these augmented copies, the *original* files themselves might still suffer from non-sequential leakage. If a machine produces the exact same rhythmic anomaly continuously over several contiguous recordings, a standard split could place some of those highly correlated recordings in the Training Set and others in the Testing Set. This creates severe non-sequential data leakage, allowing the model to simply memorize the exact sound profile of that specific recording session.

**The Solution:** We implemented an advanced Intra-Class Clustering strategy coupled with Stratified Group Splitting in our training pipeline scripts:
1. We iterate through every class (e.g., `Machine 1_Abnormal`).
2. **Density-Based Dynamic Scaling:** Instead of a hardcoded cluster count, the pipeline dynamically targets a goldilocks density of **~60 samples per cluster**. This algorithmically ensures that whether the dataset is augmented (168k samples) or raw (56k samples), the cluster sizes remain perfectly granular. Higher densities (e.g., 500) were found to be too broad for optimal CV, while lower densities (e.g., 5) resulted in data leakage.
3. **Robustness via Dynamic Seeding:** To ensure our model's performance isn't just "getting lucky" on a specific random split, the pipeline generates a dynamic time-based seed (`int(time.time()) % 100`) on every run. This seed is globally applied to NumPy, Random Forest, and K-Means, ensuring a rigorous, non-deterministic evaluation.
4. **Stratified Splitting:** We perform a `GroupShuffleSplit` (or `GroupKFold`) *per class*. This guarantees that exactly 80% of the dynamically sized clusters for every individual class go into Training, and 20% go into Testing.

This guarantees an impermeable barrier: an entire cluster of highly similar sounds goes *entirely* into the Training Set, or *entirely* into the Testing Set. The model is forced to generalize robustly across randomized initializations rather than memorizing a specific clustered sound profile.

## 3. Feature Pruning for Inference Speed
Our EDA discovered 104 acoustic features. While highly predictive, computing all 104 features on an edge device (e.g., a low-power IoT sensor on a factory floor) consumes unnecessary CPU cycles and battery power.
- Using a fast Random Forest pass, we evaluated Gini Importances and **pruned the bottom 64 features**.
- We retained only the **Top 40 most critical features** (heavily favoring `mfcc_3`, `mfcc_10`, and `spectral_contrast`). This drastically speeds up the inference latency with minimal loss in predictive power. The remaining features are saved natively to `top_40_features.joblib`.

## 4. XGBoost Migration & Hyperparameter Tuning
To finalize the classifier, we migrated from Random Forest to **XGBoost** due to its superior inference speed and handling of tabular data.
- **Tuning:** A `GridSearchCV` was executed across `max_depth`, `learning_rate`, and `n_estimators`.
- **Handling Class Imbalance:** The "Normal" classes drastically outnumbered the "Abnormal" classes (~10k vs ~2k samples), causing poor recall on anomalous sounds. We resolved this by computing balanced sample weights (`compute_sample_weight('balanced')`) and passing them directly into XGBoost's `fit()` method. This mathematically penalizes mistakes on the minority classes heavily, forcing the model to learn the fault signatures.
- **Optimization Metric:** We optimized strictly for `Macro F1-Score` to ensure the model focuses equally on all classes.
- **Serialization:** The champion XGBoost model is saved to `best_xgb_model.json` alongside the `StandardScaler` and `LabelEncoder`.

### Performance Summary (Optimized Run)
Using the optimized clustering density (~60 samples/cluster), the pipeline achieved:
- **Best CV Macro F1:** 0.9634
- **Final Test Macro F1:** 0.9793

This indicates extremely robust generalization across machine types and acoustic environments.

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
# Run the pipeline excluding augmented data for the purest baseline evaluation
python .\scripts\run_training_pipeline.py --no-augmentation
```

This runs two sub-scripts sequentially:
- `train_01_feature_engineering.py`: Processes `data/kaggle_extracted_features.csv` to prune features and generate clustering IDs.
- `train_02_model_training.py`: Handles model splitting, grid searching, and saving final assets to the `outputs/` folder (`best_xgb_model.json`, `scaler.pkl`, `label_encoder.pkl`).
