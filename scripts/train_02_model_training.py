import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def main():
    input_csv = os.path.join('data', 'processed_features.csv')
    
    print(f"Loading processed features from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Exclude target and groups to get feature columns
    exclude_cols = ['fused_label', 'leakage_group_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_pruned = df[feature_cols]
    y_full = df['fused_label']
    groups = df['leakage_group_id']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_full)

    # 1. Stratified 80/20 Train-Test Split (Strictly Grouped per Class)
    print("Splitting data into 80% Train and 20% Unseen Test sets (stratified by class)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_idx_list = []
    test_idx_list = []
    
    for cls in df['fused_label'].unique():
        cls_mask = df['fused_label'] == cls
        cls_idx = np.where(cls_mask)[0]
        
        cls_X = X_pruned.iloc[cls_idx]
        cls_y = y_encoded[cls_idx]
        cls_groups = groups.iloc[cls_idx]
        
        print(f"[DEBUG] Class: {cls}, Samples: {len(cls_idx)}, Groups: {cls_groups.nunique()}")
        tr, te = next(gss.split(cls_X, cls_y, groups=cls_groups))
        train_idx_list.extend(cls_idx[tr])
        test_idx_list.extend(cls_idx[te])
        
    train_idx = np.array(train_idx_list)
    test_idx = np.array(test_idx_list)

    X_train = X_pruned.iloc[train_idx]
    y_train = y_encoded[train_idx]
    groups_train = groups.iloc[train_idx]

    X_test = X_pruned.iloc[test_idx]
    y_test = y_encoded[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. 5-Fold Grouped Cross Validation on Training Data
    print("\nRunning 5-Fold GroupKFold Cross Validation on the Training Set...")
    gkf = GroupKFold(n_splits=5)
    
    # Using GPU for faster training. Note: using n_jobs=1 in GridSearchCV prevents GPU context switching overhead
    xgb = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=42, tree_method='hist', device='cuda')

    param_grid = {
    'n_estimators': [500,1000],
    'max_depth': [3,5],
    'learning_rate': [0.1,0.2]
    }

    grid_search = GridSearchCV(xgb, param_grid, cv=gkf, scoring='f1_macro', n_jobs=1, verbose=3)

    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    
    # Compute class weights for highly imbalanced dataset
    sample_weights = compute_sample_weight('balanced', y_train)
    
    grid_search.fit(X_train_scaled, y_train, groups=groups_train, sample_weight=sample_weights)

    best_xgb = grid_search.best_estimator_
    print(f"\nBest XGBoost Parameters from CV: {grid_search.best_params_}")
    print(f"Best CV Macro F1-Score: {grid_search.best_score_:.4f}")

    # 3. Final Evaluation on Unseen Test Data
    print("\n--- FINAL EVALUATION ON UNSEEN HOLD-OUT TEST SET ---")
    preds = best_xgb.predict(X_test_scaled)
    macro_f1 = f1_score(y_test, preds, average='macro')
    print(f"Final Test Set Macro F1: {macro_f1:.4f}\n")

    target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, preds, target_names=target_names, zero_division=0.0))
    
    # 4. Confusion Matrix Generation
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (Macro F1: {macro_f1:.4f})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = os.path.join('outputs', 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Optional: Save the best model and scaler to the outputs directory
    os.makedirs('outputs', exist_ok=True)
    best_xgb.save_model(os.path.join('outputs', 'best_xgb_model.json'))
    joblib.dump(scaler, os.path.join('outputs', 'scaler.pkl'))
    joblib.dump(le, os.path.join('outputs', 'label_encoder.pkl'))
    print("\nSaved the best model, scaler, and label encoder to the 'outputs' directory.")

if __name__ == "__main__":
    main()
