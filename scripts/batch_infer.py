import argparse
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
import librosa
from pathlib import Path
from tqdm import tqdm
import sys

# Import custom preprocessing logic
from audio_preprocessing import denoise_audio, normalize_audio, extract_advanced_features

def main():
    parser = argparse.ArgumentParser(description="Machine Sound Classification - Batch Inference")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .wav files")
    parser.add_argument("--output", type=str, default="outputs/batch_predictions.csv", help="Path to save results")
    parser.add_argument("--model", type=str, default="outputs/best_xgb_model.json", help="Path to model")
    parser.add_argument("--scaler", type=str, default="outputs/scaler.pkl", help="Path to scaler")
    parser.add_argument("--encoder", type=str, default="outputs/label_encoder.pkl", help="Path to encoder")
    parser.add_argument("--features", type=str, default="outputs/top_40_features.pkl", help="Path to features list")
    args = parser.parse_args()

    # 1. Validate Artifacts
    for path in [args.model, args.scaler, args.encoder, args.features]:
        if not os.path.exists(path):
            print(f"Error: Production artifact not found at {path}.")
            sys.exit(1)

    # 2. Load Assets
    print("Loading models and assets...")
    model = XGBClassifier()
    model.load_model(args.model)
    scaler = joblib.load(args.scaler)
    label_encoder = joblib.load(args.encoder)
    top_40_features = joblib.load(args.features)

    # 3. Collect Files
    input_path = Path(args.dir)
    wav_files = list(input_path.rglob("*.wav"))
    print(f"Found {len(wav_files)} audio files for processing.")

    results = []

    # 4. Batch Process
    for wav_file in tqdm(wav_files, desc="Inference"):
        try:
            # Load and Preprocess
            y, sr = librosa.load(wav_file, sr=None)
            y_clean = denoise_audio(y)
            y_norm = normalize_audio(y_clean)

            # Feature Extraction
            raw_features = extract_advanced_features(y_norm, sr)
            if not raw_features:
                continue

            # Align and Predict
            df_features = pd.DataFrame([raw_features])
            X_final = df_features[top_40_features].fillna(0)
            X_scaled = scaler.transform(X_final)

            pred_idx = model.predict(X_scaled)[0]
            pred_prob = model.predict_proba(X_scaled)[0]
            confidence = np.max(pred_prob)
            predicted_class = label_encoder.inverse_transform([pred_idx])[0]

            results.append({
                "file_path": str(wav_file),
                "prediction": predicted_class,
                "confidence": confidence
            })

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

    # 5. Save Results
    if results:
        df_results = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df_results.to_csv(args.output, index=False)
        print(f"\n✅ Batch inference complete. Results saved to: {args.output}")
    else:
        print("\n❌ No predictions were generated.")

if __name__ == "__main__":
    main()
