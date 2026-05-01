from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

LABEL_MAP = {
    "Machine 1_Normal": 0,
    "Machine 1_Abnormal": 1,
    "Machine 2_Normal": 2,
    "Machine 2_Abnormal": 3,
    "Machine 3_Normal": 4,
    "Machine 3_Abnormal": 5
}

def main():
    results_file = Path("results.txt")
    data_dir = Path("data/subset_2500/data")
    
    if not results_file.exists():
        print("Error: results.txt not found.")
        return
        
    # 1. Read predictions
    with open(results_file, 'r') as f:
        y_pred = [int(line.strip()) for line in f if line.strip()]
        
    # 2. Get true labels from filenames (using same sort logic as infer.py)
    def sort_key(x):
        try:
            return (0, int(x.stem))
        except ValueError:
            return (1, x.stem)
            
    wav_files = sorted(data_dir.glob("*.wav"), key=sort_key)
    
    if len(wav_files) != len(y_pred):
        print(f"Error: Count mismatch! Found {len(wav_files)} files but {len(y_pred)} predictions.")
        return
        
    y_true = []
    for f in wav_files:
        # Filename format: MachineX_ID_Class.wav
        parts = f.stem.split('_')
        # Reconstruct "Machine X_Class" (with space)
        # Note: My move script used .replace(" ", "") so "Machine1"
        # Let's handle both.
        machine_raw = parts[0] # "Machine1"
        # Add space back if missing for map lookup
        if machine_raw.startswith("Machine") and " " not in machine_raw:
            machine_name = f"Machine {machine_raw[7:]}"
        else:
            machine_name = machine_raw
            
        class_label = parts[2]
        fused = f"{machine_name}_{class_label}"
        y_true.append(LABEL_MAP.get(fused, -1))

    # 3. Calculate Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)
    print(f"Samples Evaluated: {len(y_true)}")
    print(f"Overall Accuracy:  {acc:.4f}")
    print("="*40)
    
    # Reverse the map for the report
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    target_names = [inv_map[i] for i in range(6)]
    
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
