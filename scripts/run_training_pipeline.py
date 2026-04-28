import subprocess
import sys
import os

# Suppress Intel/LLVM OpenMP collision warning caused by threadpoolctl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"Running {os.path.basename(script_path)}...")
    print(f"{'='*60}\n")
    
    # Run the script and stream the output
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(os.path.dirname(script_path)))
    
    if result.returncode != 0:
        print(f"\n[ERROR] An error occurred while running {os.path.basename(script_path)}. Exiting pipeline.")
        sys.exit(result.returncode)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the sequence of scripts to run
    pipeline_scripts = [
        "train_01_feature_engineering.py",
        "train_02_model_training.py"
    ]
    
    print("Starting the Machine Sound Classification Training Pipeline...")
    
    for script_name in pipeline_scripts:
        script_path = os.path.join(script_dir, script_name)
        run_script(script_path)
        
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
