import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import time

def main():
    current_seed = int(time.time()) % 100
    print(f"\n[INFO] Using dynamic time-based random seed: {current_seed}")
    np.random.seed(current_seed) # Set global random seed for numpy operations
    
    parser = argparse.ArgumentParser(description='Feature Engineering Script')
    parser.add_argument('--no-augmentation', action='store_true', help='Exclude augmented data')
    args = parser.parse_args()

    # Define paths
    input_csv = os.path.join('data', 'kaggle_unified_features.csv')
    output_csv = os.path.join('data', 'processed_features.csv')
    
    print(f"Loading features from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if args.no_augmentation:
        print("Filtering out augmented data (--no-augmentation flag used)...")
        if 'is_augmented' in df.columns:
            df = df[df['is_augmented'] == False].reset_index(drop=True)
            print(f"Remaining samples after removing augmentations: {len(df)}")
        else:
            print("Warning: 'is_augmented' column not found, proceeding with full dataset.")

    # 1. Feature Pruning
    exclude_cols = ['file_path', 'machine_name', 'label', 'fused_label', 'is_augmented', 'processing_type']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_full = df[feature_cols].fillna(0)
    y_full = df['fused_label']

    print("Running Random Forest to prune features...")
    rf = RandomForestClassifier(n_estimators=50, random_state=current_seed, n_jobs=-1)
    rf.fit(X_full, y_full)
    importances = pd.DataFrame({'Feature': feature_cols, 'Importance': rf.feature_importances_})
    top_40_features = importances.sort_values(by='Importance', ascending=False)['Feature'].head(40).tolist()
    X_pruned = df[top_40_features].fillna(0)

    # 2. Intra-Class Clustering for Leakage Prevention
    print("Clustering within classes for strict grouping...")
    df = df.copy()
    df['leakage_group_id'] = -1
    group_id_counter = 0

    scaler_cluster = StandardScaler()
    X_scaled_for_clustering = scaler_cluster.fit_transform(X_pruned)

    total_samples = len(df)
    
    # Dynamically calculate the target total clusters based on dataset size!
    # This automatically scales down if you use --no-augmentation
    target_samples_per_cluster = 300
    target_total_clusters = max(6, int(total_samples / target_samples_per_cluster))

    for cls in df['fused_label'].unique():
        cls_mask = df['fused_label'] == cls
        X_cls = X_scaled_for_clustering[cls_mask]
        
        # Calculate proportional cluster count
        proportion = len(X_cls) / total_samples
        n_clusters = max(2, int(np.round(proportion * target_total_clusters)))
        n_clusters = min(n_clusters, len(X_cls)) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=current_seed)
        cls_clusters = kmeans.fit_predict(X_cls)
        
        df.loc[cls_mask, 'leakage_group_id'] = cls_clusters + group_id_counter
        group_id_counter += n_clusters

    print(f"Total unique acoustic clusters formed: {group_id_counter}")

    # Save the processed data for the next script
    out_cols = top_40_features + ['fused_label', 'leakage_group_id']
    print(f"Saving processed features to {output_csv}...")
    df[out_cols].to_csv(output_csv, index=False)
    print("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
