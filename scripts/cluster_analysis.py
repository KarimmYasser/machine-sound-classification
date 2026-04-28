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

warnings.filterwarnings('ignore')

def main():
    print("Loading dataset...")
    df = pd.read_csv('processed_features_augmented.csv')
    
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
    # 2. K-Means Clustering
    # ---------------------------------------------------------
    print("Running K-Means Clustering...")
    # Trying K=6 since there are 3 machines x 2 states (Normal/Abnormal)
    kmeans = KMeans(n_clusters=6, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Create a cross-tabulation to see how unsupervised clusters align with true labels
    cross_tab = pd.crosstab(df['fused_label'], df['Cluster'])
    cross_tab.to_csv('kmeans_crosstab.csv')
    print("K-Means Cross-Tabulation saved to kmeans_crosstab.csv")
    
    # ---------------------------------------------------------
    # 3. t-SNE Analysis (on a subset to save time)
    # ---------------------------------------------------------
    print("Running t-SNE on a subset...")
    subset_size = min(5000, len(df))
    df_sub = df.sample(n=subset_size, random_state=42).copy()
    X_sub_scaled = scaler.transform(df_sub[feature_cols].fillna(0))
    
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
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
