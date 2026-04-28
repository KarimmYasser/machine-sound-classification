import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading dataset...")
    df = pd.read_csv('processed_features_augmented.csv')
    
    # Identify label columns
    df['fused_label'] = df['machine_name'] + '_' + df['label']
    exclude_cols = ['file_path', 'machine_name', 'label', 'fused_label', 'is_augmented']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['label'] # We want to find features that predict Normal vs Abnormal
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # ---------------------------------------------------------
    # 1. Feature Importance using Random Forest
    # ---------------------------------------------------------
    print("Computing Feature Importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled_df, y)
    
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    # Save top 15 features to CSV
    top_15_features = feat_imp.head(15)
    top_15_features.to_csv('top_15_features.csv', index=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_15_features, palette='viridis')
    plt.title('Top 15 Most Important Features for Anomaly Detection (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Boxplots of Top 4 Features by Machine and Condition
    # ---------------------------------------------------------
    print("Generating Boxplots...")
    top_4_cols = top_15_features['Feature'].values[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_4_cols):
        sns.boxplot(x='machine_name', y=col, hue='label', data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_ylabel('Feature Value')
        axes[i].set_xlabel('Machine')
        
    plt.tight_layout()
    plt.savefig('top_features_boxplot.png', dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Correlation Heatmap of Top 15 Features
    # ---------------------------------------------------------
    print("Generating Correlation Heatmap...")
    top_15_cols = top_15_features['Feature'].values
    corr_matrix = df[top_15_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Top 15 Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()
    
    print("Extensive analysis complete! Generated feature_importance.png, top_features_boxplot.png, and correlation_heatmap.png")

if __name__ == "__main__":
    main()
