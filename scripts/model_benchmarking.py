import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading dataset...")
    df = pd.read_csv('processed_features_augmented.csv')
    
    # 1. Target Fusion (Machine + Condition)
    df['fused_label'] = df['machine_name'] + '_' + df['label']
    
    exclude_cols = ['file_path', 'machine_name', 'label', 'fused_label', 'is_augmented']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['fused_label']
    
    print("Splitting dataset into 80% Train, 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define baseline algorithms to test against each other
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = []
    best_model_name = None
    best_f1 = -1
    best_preds = None
    best_model = None
    
    print("Benchmarking Models...")
    for name, model in models.items():
        print(f"Training {name}...")
        # Train model
        model.fit(X_train_scaled, y_train)
        # Test model
        preds = model.predict(X_test_scaled)
        
        # Macro F1 is the best metric for multi-class classification
        macro_f1 = f1_score(y_test, preds, average='macro')
        results.append({'Model': name, 'Macro F1-Score': macro_f1})
        print(f"  -> {name} Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            best_preds = preds
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
    plt.savefig('model_benchmark.png', dpi=300)
    plt.close()
    
    # Generate Confusion Matrix for Champion Model
    print(f"Generating Confusion Matrix for Champion: {best_model_name}")
    cm = confusion_matrix(y_test, best_preds, labels=best_model.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.ylabel('Actual Condition')
    plt.xlabel('Predicted Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('champion_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Save the classification report text
    report = classification_report(y_test, best_preds)
    with open("champion_classification_report.txt", "w") as f:
        f.write(f"Champion Model: {best_model_name}\\n")
        f.write("="*60 + "\\n")
        f.write(report)
        f.write("\\n")
        
    print("Benchmarking complete! Saved model_benchmark.png, champion_confusion_matrix.png, and champion_classification_report.txt")

if __name__ == "__main__":
    main()
