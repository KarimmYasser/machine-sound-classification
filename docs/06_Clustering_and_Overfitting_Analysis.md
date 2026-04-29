# 06. Clustering and Overfitting Analysis

## Overview
In the Machine Sound Classification pipeline, we utilize **Intra-Class K-Means Clustering** to prevent data leakage during cross-validation. This document analyzes the critical relationship between clustering granularity (the number of clusters) and the model's ability to generalize to unseen data.

---

## 1. The Clustering Strategy
The pipeline groups audio samples within each class into acoustic clusters. These clusters serve as the `groups` for `GroupKFold` cross-validation and `GroupShuffleSplit`.

**Goal:** Ensure that segments from the same recording session or machine instance are kept together in either the training set or the validation set, preventing the model from "cheating" by memorizing file-specific noise signatures.

---

## 2. Granularity Scenarios

### Case A: Under-Granular (Too Few Clusters)
*   **Result:** Pessimistic Evaluation.
*   **Observation:** High Test Score, Low CV Score.
*   **Reasoning:** Large clusters remove significant chunks of the acoustic space from training. The validation set becomes "too different" from the training set, causing the CV score to drop even if the model is learning correctly.

### Case B: The "Goldilocks" Zone (Optimal)
*   **Result:** Robust Generalization.
*   **Observation:** High CV Score (~0.95+), High Test Score (~0.95+), close alignment between the two.
*   **Reasoning:** Clusters are small enough to cover the acoustic diversity well but large enough to encompass all segments from a single recording. This is the target for a production-ready model.

### Case C: Over-Granular (Too Many Clusters)
*   **Result:** Data Leakage / Overfitting.
*   **Observation:** Near-perfect scores (~0.99+) on both CV and Test.
*   **Reasoning:** If the number of clusters approaches the number of samples, individual segments from the same file are split into different clusters. The model memorizes the "background noise" of a specific file and identifies it in the test set, creating an illusion of high performance.

---

## 3. Empirical Findings (Run Comparison)

| Metric | Run 1 (High) | Run 2 (Medium) | Run 3 (Extreme) | **Run 4 (Stricter/Current)** |
| :--- | :--- | :--- | :--- | :--- |
| **Samples/Cluster** | ~1000 | ~60 | ~10 | **300** |
| **CV Macro F1** | 0.49 | 0.96 | 0.98 | **0.89** |
| **Test Macro F1** | 0.76 | 0.98 | 0.99 | **0.92** |
| **Interpretation** | Under-granular | Optimized | **Leaking** | **Most Robust / Least Leakage** |

---

## 4. Recommendations for Maintenance
To maintain model integrity as the dataset grows:
1.  **Monitor the CV-Test Gap:** A wide gap suggests clusters are too large; near-perfection suggests they are too small.
2.  **Anchor to File Count:** The number of clusters should ideally not exceed the number of unique source audio files.
3.  **Use Density-Based Scaling:** Continue using the `target_samples_per_cluster` logic. **Target 300 samples per cluster** to ensure absolute minimum leakage between training and testing sets.
