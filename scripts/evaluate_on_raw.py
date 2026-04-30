import pandas as pd
import numpy as np
import joblib
import os
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate champion model on raw dataset')
    parser.add_argument('--model', type=str, default='outputs/best_xgb_model.json', help='Path to XGBoost model')
    parser.add_argument('--scaler', type=str, default='outputs/scaler.pkl', help='Path to scaler')
    parser.add_argument('--encoder', type=str, default='outputs/label_encoder.pkl', help='Path to label encoder')
    parser.add_argument('--input', type=str, default='data/kaggle_extracted_features.csv', help='Path to raw dataset')
    args = parser.parse_args()

    # 1. Load Artifacts
    print("Loading model and assets...")
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    model = XGBClassifier()
    model.load_model(args.model)
    scaler = joblib.load(args.scaler)
    le = joblib.load(args.encoder)
    
    # 2. Load Raw Dataset
    print(f"Loading raw dataset from {args.input}...")
    df = pd.read_csv(args.input)
    
    # 3. Prepare Features
    # Retrieve feature names directly from the scaler to ensure perfect alignment
    if hasattr(scaler, 'feature_names_in_'):
        feature_cols = list(scaler.feature_names_in_)
        print(f"Using {len(feature_cols)} features identified from scaler.")
    else:
        # Fallback to the corrected list if feature_names_in_ is missing
        print("Warning: Scaler does not contain feature names. Using manual list.")
        feature_cols = [
            'bandwidth_mean', 'centroid_mean', 'centroid_std',
            'contrast_1_mean', 'contrast_1_std', 'contrast_2_mean', 'contrast_2_std', 
            'contrast_3_mean', 'contrast_4_mean', 'contrast_5_mean', 'contrast_6_mean', 
            'contrast_7_mean', 'contrast_7_min', 'mfcc_10_mean', 'mfcc_11_mean', 
            'mfcc_12_mean', 'mfcc_13_mean', 'mfcc_13_std', 'mfcc_1_mean', 'mfcc_1_std', 
            'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_min', 'mfcc_4_max', 
            'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std', 'mfcc_6_max', 
            'mfcc_6_mean', 'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std', 
            'rms_mean', 'rolloff_mean', 'rolloff_std', 'zcr_mean', 'zcr_std'
        ]
    
    # Check if all feature columns exist in the input DF
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in input dataset: {missing_cols}")
        # Try to find them (maybe they have slightly different names or we need to use all columns)
        # Actually, let's just use the columns that are there and see.
        return

    X = df[feature_cols].fillna(0)
    y_true = le.transform(df['fused_label'])
    
    # 4. Run Inference
    print(f"Running inference on {len(X)} samples...")
    
    X_scaled = scaler.transform(X)
    
    start_time = time.perf_counter()
    y_pred = model.predict(X_scaled)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    latency_ms = (total_time / len(X)) * 1000
    
    # 5. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS ON RAW DATASET")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Latency:   {latency_ms:.6f} ms/sample")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    # Save results to a file as requested in previous turn (good practice)
    output_file = "outputs/raw_dataset_evaluation.txt"
    with open(output_file, "w") as f:
        f.write("EVALUATION RESULTS ON RAW DATASET\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Macro F1:  {macro_f1:.4f}\n")
        f.write(f"Latency:   {latency_ms:.6f} ms/sample\n")
        f.write("="*40 + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=le.classes_))
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
