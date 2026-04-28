import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def main():
    input_csv = os.path.join('data', 'processed_features.csv')
    output_dir = os.path.join('visualizations', 'clusters')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading processed features from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please run 'train_01_feature_engineering.py' first.")
        return
        
    classes = df['fused_label'].unique()
    feature_cols = [c for c in df.columns if c not in ['fused_label', 'leakage_group_id']]
    
    print(f"Found {len(classes)} classes. Beginning PCA and plotting...")
    
    for cls in classes:
        print(f"Generating PCA cluster graph for: {cls}...")
        cls_df = df[df['fused_label'] == cls]
        
        X = cls_df[feature_cols]
        y_cluster = cls_df['leakage_group_id'].values
        
        # Standardize features before applying PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reduce to 2 dimensions for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot colored by the K-Means cluster ID (leakage_group_id)
        scatter = plt.scatter(
            X_pca[:, 0], 
            X_pca[:, 1], 
            c=y_cluster, 
            cmap='nipy_spectral', # High contrast colormap for many clusters
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        
        plt.title(f"Intra-Class Clusters (Leakage Groups): {cls}\n(Points with the same color are grouped tightly to prevent leakage)")
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
        
        # Save the figure
        safe_cls_name = str(cls).replace(' ', '_').replace('/', '_')
        out_path = os.path.join(output_dir, f"cluster_{safe_cls_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" -> Saved to {out_path}")

    print("\nAll cluster visualizations have been successfully saved!")

if __name__ == "__main__":
    main()
