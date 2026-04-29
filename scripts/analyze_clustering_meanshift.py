import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import os

def analyze_optimal_clusters():
    input_csv = os.path.join('data', 'kaggle_unified_features.csv')
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run scripts/create_unified_dataset.py first.")
        return

    print(f"Loading features from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Exclude metadata for clustering analysis
    exclude_cols = ['file_path', 'machine_name', 'label', 'fused_label', 'is_augmented', 'processing_type']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # To "push it to the limit", we'll analyze multiple classes with higher sample counts
    # and extremely small quantiles to force discovery of tiny acoustic variations.
    all_classes = df['fused_label'].unique()
    
    quantiles = [0.05, 0.01, 0.005, 0.001]
    results = []

    for target_class in all_classes:
        print(f"\n>>> PROCESSING CLASS: {target_class}")
        df_cls = df[df['fused_label'] == target_class].copy()
        
        # Increased limit for "pushing it"
        sample_limit = 10000
        if len(df_cls) > sample_limit:
            print(f"Sampling {sample_limit} rows from {len(df_cls)}...")
            df_cls = df_cls.sample(n=sample_limit, random_state=42)
        
        X = df_cls[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"{'Quantile':<10} | {'Bandwidth':<10} | {'Clusters':<10} | {'Avg Size':<10}")
        print("-" * 50)
        
        for q in quantiles:
            try:
                # n_samples for estimate_bandwidth also increased for accuracy
                bw = estimate_bandwidth(X_scaled, quantile=q, n_samples=1000)
                if bw <= 0: bw = 0.05 # Extreme lower bound
                
                # bin_seeding=True is critical for speed at high N
                ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
                ms.fit(X_scaled)
                
                n_clusters = len(np.unique(ms.labels_))
                avg_size = len(X_scaled) / n_clusters
                print(f"{q:<10} | {bw:<10.4f} | {n_clusters:<10} | {avg_size:<10.2f}")
                
                results.append({
                    'class': target_class,
                    'quantile': q,
                    'bandwidth': bw,
                    'clusters': n_clusters,
                    'avg_size': avg_size
                })
            except Exception as e:
                print(f"Failed at quantile {q}: {e}")

    print("\n--- AGGREGATED SUMMARY ---")
    res_df = pd.DataFrame(results)
    avg_across_classes = res_df.groupby('quantile')['clusters'].mean()
    print("\nAverage clusters discovered per quantile (across all machines):")
    print(avg_across_classes)

if __name__ == "__main__":
    analyze_optimal_clusters()
