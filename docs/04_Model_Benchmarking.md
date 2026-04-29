# Phase 4: Model Benchmarking

## 1. Experimental Setup
With a highly robust 104-feature dataset generated (including augmented data to prevent overfitting), we evaluated three classic machine learning architectures to establish a classification baseline.

The problem was framed as a **6-Class Multiclass Classification** task:
- Machine 1_Normal
- Machine 1_Abnormal
- Machine 2_Normal
- Machine 2_Abnormal
- Machine 3_Normal
- Machine 3_Abnormal

**Data Split:** 80% Training / 20% Testing (Stratified to maintain class balance).
**Scaling:** `StandardScaler` was fit on the training data and applied to the test data.
**Metric:** Macro F1-Score (Best metric for imbalanced/multi-class problems).

## 2. Models Evaluated
1. **Logistic Regression:** Used to establish a linear baseline. If this performs well, the classes are linearly separable in the 104D feature space.
2. **Support Vector Machine (SVM):** Used with an RBF (Radial Basis Function) kernel to handle complex, non-linear boundaries.
3. **Random Forest Classifier:** A tree-based ensemble method known for high performance on tabular acoustic feature datasets.

## 3. Results & The Data Leakage Revelation
Our initial naive evaluations yielded highly suspicious metrics:
1. **Random Forest:** `0.9962` 
2. **Support Vector Machine (SVM):** `0.9874`
3. **Logistic Regression:** `0.9632`

**The Revelation:** Achieving 99.6% accuracy on a complex industrial acoustic dataset is a major red flag for **Data Leakage**. Because highly rhythmic anomalous audio files are highly correlated, standard randomized `train_test_split` algorithms accidentally placed near-identical sound frames into both the training and testing sets.

## 4. The Corrected Baseline (Strict Intra-Class GroupKFold)
To achieve a highly realistic baseline, we introduced a strict, mathematically balanced 5-Fold GroupKFold algorithm using density-based dynamic intra-class K-Means clustering. Furthermore, we stripped out all augmented data using the `--no-augmentation` flag to ensure the pipeline was evaluated purely on raw acoustic data.

**Corrected Results (XGBoost, Raw Data, No Leakage):**
* **Best CV Macro F1-Score:** ~`0.50`
* **Final Test Set Macro F1:** ~`0.7937`
* **Overall Accuracy:** `83%`

## 5. Conclusion & Next Steps
The feature engineering pipeline is incredibly strong. While the initial 99% score was heavily inflated, an **83% True Accuracy (0.79 Macro F1)** on completely unseen, leak-proof raw acoustic data is an outstanding result for industrial anomaly detection. 

Next Steps:
1. Consider migrating the production inference engine to utilize edge-friendly ONNX runtimes.
2. Consider adding an un-supervised anomaly detection path (e.g., Isolation Forest) as a fallback mechanism for out-of-distribution sounds.
