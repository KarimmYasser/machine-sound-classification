import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
import time
import argparse
import os

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Model Benchmarking Script')
    parser.add_argument('--input', type=str, default='data/processed_features.csv', help='Input CSV file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for metrics and plots')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Dataset loaded with {len(df)} samples across {df['fused_label'].nunique()} classes.")
    
    # Exclude metadata for features
    exclude_cols = ['fused_label', 'leakage_group_id', 'file_path', 'machine_name', 'label', 'is_augmented', 'processing_type']
    feature_cols = [c for c in df.columns if c in df.columns and c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    
    # Handle grouping for leakage prevention
    if 'leakage_group_id' in df.columns:
        groups = df['leakage_group_id']
    elif 'file_path' in df.columns:
        print("Using 'file_path' for strict grouping (leakage prevention)...")
        groups = df['file_path']
    else:
        print("Warning: No grouping column found (leakage_group_id or file_path). Using individual samples.")
        groups = np.arange(len(df))
    
    # Encode categorical target for XGBoost compatibility
    le = LabelEncoder()
    y = le.fit_transform(df['fused_label'])
    
    print("Splitting dataset into 80% Train, 20% Test (Strict Grouped Split)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define baseline algorithms to test against each other (High Performance Settings)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.2, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
    }
    
    results = []
    best_model_name = None
    best_f1 = -1
    best_preds = None
    best_model = None
    
    print("Benchmarking Models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Handle SVM separately to avoid O(n^2) hang on large dataset
        curr_X_train, curr_y_train = X_train_scaled, y_train
        curr_X_test, curr_y_test = X_test_scaled, y_test
        
        if name == 'Support Vector Machine (SVM)' and len(X_train_scaled) > 15000:
            print(f"  [NOTE] Sampling 15,000 rows for SVM to prevent training hang...")
            # Use deterministic seed for sampling to ensure reproducible benchmarks
            np.random.seed(42)
            idx = np.random.choice(len(X_train_scaled), 15000, replace=False)
            curr_X_train, curr_y_train = X_train_scaled[idx], y_train[idx]
            # Also sample test set for consistent latency measurement
            test_idx_samp = np.random.choice(len(X_test_scaled), min(3000, len(X_test_scaled)), replace=False)
            curr_X_test, curr_y_test = X_test_scaled[test_idx_samp], y_test[test_idx_samp]

        # Train model
        model.fit(curr_X_train, curr_y_train)
        
        # Measure Inference Latency
        start_time = time.perf_counter()
        preds = model.predict(curr_X_test)
        end_time = time.perf_counter()
        
        latency_ms = ((end_time - start_time) / len(curr_X_test)) * 1000
        
        # Metrics
        macro_f1 = f1_score(curr_y_test, preds, average='macro')
        acc = accuracy_score(curr_y_test, preds)
        
        results.append({
            'Model': name, 
            'Macro F1-Score': macro_f1, 
            'Accuracy': acc, 
            'Latency (ms)': latency_ms
        })
        
        print(f"  -> {name} | F1: {macro_f1:.4f} | Acc: {acc:.4f} | Latency: {latency_ms:.6f}ms")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            # If champion, compute preds on FULL test set for final visuals
            best_preds = model.predict(X_test_scaled)
            best_model = model
            
    # Save Model Benchmark Bar Plot
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Macro F1-Score', y='Model', data=res_df, palette='magma')
    plt.title('Model Benchmark Comparison (Macro F1-Score)')
    # Zoom in X-axis if scores are all above 0.8
    if res_df['Macro F1-Score'].min() > 0.8:
        plt.xlim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_benchmark.png'), dpi=300)
    plt.close()
    
    # Generate Confusion Matrix for Champion Model
    print(f"Generating Confusion Matrix for Champion: {best_model_name}")
    class_names = le.classes_
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.ylabel('Actual Condition')
    plt.xlabel('Predicted Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'champion_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Save the classification report text
    report = classification_report(y_test, best_preds, target_names=le.classes_, labels=range(len(le.classes_)))
    report_path = os.path.join(args.output_dir, "champion_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Champion Model: {best_model_name}\n")
        f.write("="*60 + "\n")
        f.write(report)
        f.write("\n\nDetailed Benchmark Results:\n")
        f.write(pd.DataFrame(results).to_string(index=False))
        f.write("\n")
        
    print(f"Benchmarking complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()

