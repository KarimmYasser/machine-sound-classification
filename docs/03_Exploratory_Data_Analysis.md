# Phase 3: Exploratory Data Analysis & Unsupervised Learning

## 1. Dimensionality Reduction (PCA & t-SNE)
We compressed the 104 dimensions into 2D projections using PCA and t-SNE to observe how the data naturally clusters without providing any labels.

**Findings:**
1. **Machine-Specific Grouping:** The feature vectors distinctly separate by **Machine ID**. Machine 1, Machine 2, and Machine 3 form entirely separate dense clusters in both PCA and t-SNE space.
2. **Subtle Anomaly Boundaries:** The differences between "Normal" and "Abnormal" within a specific machine are much less pronounced than the differences between the machines themselves. The acoustic signature of the specific machine heavily dominates the variance.

## 2. K-Means Unsupervised Clustering
When a K-Means algorithm was asked to find `K=6` natural clusters, it perfectly grouped the Machine types but struggled slightly with the condition labels.

**Takeaway on ML Approach:** 
K-Means successfully isolated the macro-level machines from each other (e.g., Machine 2 dominates Clusters 0 and 3). However, unsupervised algorithms struggle to perfectly isolate "Normal" from "Abnormal" because the anomaly features are a very subtle acoustic variance compared to the overall operating sound of the machine. This confirms that **supervised learning algorithms** (like XGBoost or SVM) are strictly required. They can be trained to look past the macro-level machine noise and prioritize the micro-level anomaly indicators.

## 3. Feature Importance (Random Forest)
To find exactly *what* separates a Normal machine from an Abnormal one, we trained a Random Forest classifier across the entire augmented dataset to rank all 104 acoustic features by Gini Importance.

**Key EDA Insights:**
1. **The Power of MFCCs:** `mfcc_3_mean` and `mfcc_10_mean` emerged as the absolute most critical features for detecting anomalies. This confirms that the mechanical anomalies are manifesting as specific resonant frequencies or timbral shifts in the mid-to-high frequency ranges.
2. **Spectral Contrast Validated:** The newly added `contrast_2_mean` and `contrast_1_mean` features cracked the top 15. The decibel difference between peak noise and valley noise in the lower-mid frequency bands strongly correlates with mechanical failure (e.g., rhythmic clanking flattening the spectrum).
3. **RMS Still Holds Value (Temporally):** While `rms_mean` dropped completely out of importance due to our peak volume normalization, `rms_min` and `rms_std` remain highly predictive. This proves it is the *fluctuation* and *rhythm* of the volume (like a sudden loud knock, causing a high standard deviation) rather than the absolute volume that gives away an anomaly.
