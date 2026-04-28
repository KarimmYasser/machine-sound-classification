import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('processed_features_augmented.csv')
    df['fused_label'] = df['machine_name'] + '_' + df['label']
    
    # --- 1. Feature Pruning ---
    exclude_cols = ['file_path', 'machine_name', 'label', 'fused_label', 'is_augmented', 'Cluster', 'PCA1', 'PCA2', 'tSNE1', 'tSNE2']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_full = df[feature_cols].fillna(0)
    y_full = df['fused_label']
    
    print("Pruning features to optimize inference latency...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_full, y_full)
    
    importances = pd.DataFrame({'Feature': feature_cols, 'Importance': rf.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False)
    
    # Keep Top 40 features for edge device inference optimization
    top_40_features = importances['Feature'].head(40).tolist()
    print(f"Kept Top 40 features. Dropped {len(feature_cols) - 40} redundant/low-importance features.")
    
    X_pruned = df[top_40_features].fillna(0)
    
    # Encode target labels to integers for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_full)
    joblib.dump(le, 'label_encoder.joblib')
    
    # --- 2. Fix Data Leakage via Intra-Class Clustering ---
    print("Clustering within each class to identify and group correlated samples...")
    # We assign a unique group ID to each cluster within each class
    df['leakage_group_id'] = -1
    group_id_counter = 0
    
    from sklearn.cluster import KMeans
    # Standardize data strictly for the clustering phase
    scaler_cluster = StandardScaler()
    X_scaled_for_clustering = scaler_cluster.fit_transform(X_pruned)
    
    for cls in df['fused_label'].unique():
        cls_mask = df['fused_label'] == cls
        X_cls = X_scaled_for_clustering[cls_mask]
        
        # Assume 10 clusters per class to group highly similar recordings
        n_clusters = min(10, len(X_cls)) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cls_clusters = kmeans.fit_predict(X_cls)
        
        # Assign globally unique group IDs
        df.loc[cls_mask, 'leakage_group_id'] = cls_clusters + group_id_counter
        group_id_counter += n_clusters

    print(f"Created {group_id_counter} unique clusters across all classes for grouping.")

    print("Splitting data (Strict GroupShuffleSplit on Cluster IDs to prevent Data Leakage)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df['leakage_group_id']
    
    train_idx, test_idx = next(gss.split(X_pruned, y_encoded, groups=groups))
    
    X_train = X_pruned.iloc[train_idx]
    X_test = X_pruned.iloc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')
    
    # --- 3. XGBoost Migration & Tuning ---
    print("Tuning XGBoost Hyperparameters via GridSearchCV...")
    xgb = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1]
    }
    
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    
    # Evaluate strictly
    preds = best_xgb.predict(X_test_scaled)
    macro_f1 = f1_score(y_test, preds, average='macro')
    
    print(f"\nSTRICT Test Set Macro F1 (No Leakage): {macro_f1:.4f}")
    print("\nClassification Report:")
    target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, preds, target_names=target_names))
    
    # Save the pruned feature list and the final champion model
    joblib.dump(top_40_features, 'top_40_features.joblib')
    best_xgb.save_model('champion_xgboost.json')
    print("Model, scaler, encoder, and feature list successfully saved for production!")

if __name__ == "__main__":
    main()
