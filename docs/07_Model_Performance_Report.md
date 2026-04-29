# 07. Model Performance Report

## Training Run Summary
**Date:** 2026-04-29  
**Goal:** Maximize model honesty and minimize leakage by using stricter clustering granularity (300 samples/cluster).

---

## 1. Execution Details
*   **Feature Engineering:** Pruned to Top 40 features using Random Forest importance.
*   **Clustering:** ~750 unique acoustic clusters formed within classes (300 samples per cluster).
*   **Algorithm:** XGBoost (Tree Method: `hist`, Device: `cuda`).
*   **Best Params:** `{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 1000}`
*   **Evaluation:** 5-Fold GroupKFold (Cross-Validation) + 20% Unseen Hold-out Test Set.

---

## 2. Performance Metrics

### Cross-Validation (Training Set)
The 5-fold CV scores reflect a more realistic generalization challenge due to the larger cluster sizes.

| Metric | Value |
| :--- | :--- |
| **Best CV Macro F1-Score** | **0.8977** |

### Final Hold-out Test Set
The model achieved a **Final Macro F1 of 0.9250** on completely unseen clusters.

#### Detailed Classification Report:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Machine 1_Abnormal | 0.98 | 0.83 | 0.90 | 2166 |
| Machine 1_Normal | 0.97 | 1.00 | 0.99 | 13851 |
| Machine 2_Abnormal | 0.66 | 0.91 | 0.76 | 2213 |
| Machine 2_Normal | 0.98 | 0.91 | 0.94 | 11119 |
| Machine 3_Abnormal | 0.99 | 0.95 | 0.97 | 3070 |
| Machine 3_Normal | 0.98 | 1.00 | 0.99 | 10207 |
| **Accuracy** | | | **0.96** | **42626** |
| **Macro Avg** | **0.93** | **0.93** | **0.92** | **42626** |

---

## 3. Analysis & Conclusions
By increasing the samples per cluster to **300**, we have significantly reduced the risk of "temporal leakage" (where nearly identical adjacent audio frames are split between training and testing). 

**Key Insights:**
- **Machine 2 Abnormalities:** Precision dropped to 0.66, indicating that the model is now being challenged by harder-to-distinguish samples that were previously "leaked" during over-granular clustering.
- **Robustness:** While the F1-score is lower than the previous 0.98, this model is **far more robust** and less likely to fail when exposed to truly new recording sessions.
- **Final Decision:** We will proceed with this configuration (300 samples/cluster) as it provides the best balance between performance and real-world reliability.
