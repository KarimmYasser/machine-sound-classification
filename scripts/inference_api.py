from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import tempfile
import os
import librosa

# Import our custom preprocessing logic
from audio_preprocessing import denoise_audio, normalize_audio, extract_advanced_features

app = FastAPI(title="Acoustic Anomaly Detection API")

# --- Load Production Assets on Startup ---
# Note: These paths assume you are running from the project root or have moved assets to the app directory.
try:
    print("Loading production models and scalers from outputs/...")
    model = XGBClassifier()
    model.load_model('outputs/best_xgb_model.json')
    
    scaler = joblib.load('outputs/scaler.pkl')
    label_encoder = joblib.load('outputs/label_encoder.pkl')
    top_40_features = joblib.load('outputs/top_40_features.pkl')
    print("Assets loaded successfully.")
except Exception as e:
    print(f"Warning: Production assets not found ({e}). Ensure you run run_training_pipeline.py first.")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Endpoint that takes a raw .wav file, preprocesses it, and returns the machine state classification.
    """
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
    # Save uploaded file to a temporary location for Librosa to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        content = await file.read()
        temp_wav.write(content)
        temp_path = temp_wav.name
        
    try:
        # 1. Load Audio
        y, sr = librosa.load(temp_path, sr=None)
        
        # 2. Preprocess (Denoise & Peak Normalize)
        y_clean = denoise_audio(y)
        y_norm = normalize_audio(y_clean)
        
        # 3. Extract Full Feature Set
        raw_features = extract_advanced_features(y_norm, sr)
        if not raw_features:
            raise HTTPException(status_code=500, detail="Feature extraction failed due to corrupted audio.")
            
        # 4. Prune and Align Features
        df_features = pd.DataFrame([raw_features])
        X_final = df_features[top_40_features].fillna(0)
        
        # 5. Scale Features
        X_scaled = scaler.transform(X_final)
        
        # 6. Predict using XGBoost
        pred_idx = model.predict(X_scaled)[0]
        pred_prob = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(pred_prob))
        
        predicted_class = label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup the temp file to save disk space
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # To run locally: python inference_api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
