import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import time

warnings.filterwarnings('ignore')

def main():
    current_seed = int(time.time()) % 100
    print(f"\n[INFO] Using dynamic time-based random seed: {current_seed}")
    np.random.seed(current_seed)
    
    print("Loading dataset...")
    try:
        df = pd.read_csv('data/kaggle_extracted_features.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/processed_features.csv')
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Identify label columns
    df['fused_label'] = df['machine_name'] + '_' + df['label']
    exclude_cols = ['file_path', 'machine_name', 'label', 'fused_label', 'is_augmented']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0) # Ensure no NaNs
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ---------------------------------------------------------
    # 1. PCA Analysis (Principal Component Analysis)
    # ---------------------------------------------------------
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='fused_label', data=df, alpha=0.6, palette='tab10')
    plt.title('PCA 2D Projection of Audio Features')
    plt.tight_layout()
    plt.savefig('pca_plot.png', dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Intra-Class K-Means Clustering (10 clusters per class)
    # ---------------------------------------------------------
    print("\nRunning Intra-Class K-Means Clustering (Proportional)...")
    df['Cluster'] = -1
    group_id_counter = 0
    
    cluster_stats = []
    total_samples = len(df)
    
    # Dynamically calculate the target total clusters based on dataset size!
    # We aim for roughly 1,500 samples per cluster to ensure granular train/test splits.
    target_samples_per_cluster = 500
    target_total_clusters = max(6, int(total_samples / target_samples_per_cluster))
    print(f"Dynamically targeting {target_total_clusters} total clusters based on {total_samples} dataset size...")

    for cls in df['fused_label'].unique():
        cls_mask = df['fused_label'] == cls
        X_cls = X_scaled[cls_mask]
        
        # Determine number of clusters based on class imbalance
        proportion = len(X_cls) / total_samples
        n_clusters = max(2, int(np.round(proportion * target_total_clusters)))
        n_clusters = min(n_clusters, len(X_cls)) # Safety check
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=current_seed)
        cls_clusters = kmeans.fit_predict(X_cls)
        
        df.loc[cls_mask, 'Cluster'] = cls_clusters + group_id_counter
        
        print(f"\nCluster sizes for {cls}:")
        unique, counts = np.unique(cls_clusters, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  - Local Cluster {cluster_id + 1}: ~{count} samples")
            cluster_stats.append({
                'Class': cls,
                'Local_Cluster_ID': cluster_id + 1,
                'Global_Cluster_ID': cluster_id + group_id_counter,
                'Sample_Count': count
            })
            
        group_id_counter += n_clusters

    pd.DataFrame(cluster_stats).to_csv('kmeans_cluster_sizes.csv', index=False)
    print("\nCluster size statistics saved to kmeans_cluster_sizes.csv")
    
    # ---------------------------------------------------------
    # 3. t-SNE Analysis (on a subset to save time)
    # ---------------------------------------------------------
    print("Running t-SNE on a subset...")
    subset_size = min(5000, len(df))
    df_sub = df.sample(n=subset_size, random_state=current_seed).copy()
    X_sub_scaled = scaler.transform(df_sub[feature_cols].fillna(0))
    
    tsne = TSNE(n_components=2, random_state=current_seed, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sub_scaled)
    df_sub['tSNE1'] = X_tsne[:, 0]
    df_sub['tSNE2'] = X_tsne[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='tSNE1', y='tSNE2', hue='fused_label', data=df_sub, alpha=0.7, palette='tab10')
    plt.title(f't-SNE 2D Projection of Audio Features (Subset of {subset_size} samples)')
    plt.tight_layout()
    plt.savefig('tsne_plot.png', dpi=300)
    plt.close()
    
    print("Analysis complete! Visualizations saved as pca_plot.png and tsne_plot.png.")

if __name__ == "__main__":
    main()
