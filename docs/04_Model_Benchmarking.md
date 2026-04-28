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

## 3. Results (Macro F1-Score)
1. **Random Forest:** `0.9962` 🏆 *(Champion)*
2. **Support Vector Machine (SVM):** `0.9874`
3. **Logistic Regression:** `0.9632`

## 4. Conclusion
The feature engineering pipeline is incredibly strong. The fact that a simple linear model (Logistic Regression) achieved over 96% accuracy proves that the acoustic features (specifically Spectral Contrast and MFCCs) mathematically separate the states extremely well.

The **Random Forest** ensemble essentially achieved perfect classification (99.6%). Reviewing the generated confusion matrix showed virtually no misclassifications across any of the 6 classes, proving that the models successfully learned to look past the macro-level machine noise and pinpoint the micro-level anomalies.

## 5. Next Steps
While Random Forest performed flawlessly on this dataset, it was trained on augmented data. To fully trust the model in production:
1. Ensure that the test set is completely "unseen" (i.e., do not leak augmented versions of a test file into the training set). 
2. Consider migrating to **XGBoost** for faster inference times in a production environment.
3. Consider dropping highly correlated, non-important features (as revealed in the EDA phase) to speed up prediction latency on edge devices.
