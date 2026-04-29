import argparse
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
import librosa
import sys

# Import custom preprocessing logic from the same directory
from audio_preprocessing import denoise_audio, normalize_audio, extract_advanced_features

def main():
    parser = argparse.ArgumentParser(description="Machine Sound Classification - CLI Inference")
    parser.add_argument("--file", type=str, required=True, help="Path to the .wav file to classify")
    parser.add_argument("--model", type=str, default="outputs/best_xgb_model.json", help="Path to the trained XGBoost model")
    parser.add_argument("--scaler", type=str, default="outputs/scaler.pkl", help="Path to the saved scaler")
    parser.add_argument("--encoder", type=str, default="outputs/label_encoder.pkl", help="Path to the saved label encoder")
    parser.add_argument("--features", type=str, default="outputs/top_40_features.pkl", help="Path to the top 40 features list")
    args = parser.parse_args()

    # 1. Validate paths
    if not os.path.exists(args.file):
        print(f"Error: Audio file not found at {args.file}")
        sys.exit(1)
    
    for path in [args.model, args.scaler, args.encoder, args.features]:
        if not os.path.exists(path):
            print(f"Error: Production artifact not found at {path}. Run the training pipeline first.")
            sys.exit(1)

    try:
        print(f"--- Classifying: {os.path.basename(args.file)} ---")
        
        # 2. Load Assets
        model = XGBClassifier()
        model.load_model(args.model)
        scaler = joblib.load(args.scaler)
        label_encoder = joblib.load(args.encoder)
        top_40_features = joblib.load(args.features)

        # 3. Load and Preprocess Audio
        print("Loading and denoising audio...")
        y, sr = librosa.load(args.file, sr=None)
        y_clean = denoise_audio(y)
        y_norm = normalize_audio(y_clean)

        # 4. Feature Extraction
        print("Extracting acoustic features...")
        raw_features = extract_advanced_features(y_norm, sr)
        if not raw_features:
            print("Error: Feature extraction failed.")
            sys.exit(1)

        # 5. Prepare for Prediction
        # Convert to DataFrame to ensure consistency with training
        df_features = pd.DataFrame([raw_features])
        
        # Prune and align with the top 40 features used during training
        X_final = df_features[top_40_features].fillna(0)
        
        # 6. Scale and Predict
        X_scaled = scaler.transform(X_final)
        pred_idx = model.predict(X_scaled)[0]
        pred_prob = model.predict_proba(X_scaled)[0]
        confidence = np.max(pred_prob)

        predicted_class = label_encoder.inverse_transform([pred_idx])[0]

        print("\n" + "="*40)
        print(f"RESULT: {predicted_class}")
        print(f"CONFIDENCE: {confidence * 100:.2f}%")
        print("="*40 + "\n")

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
