import argparse
import subprocess
import sys
import os

# Suppress Intel/LLVM OpenMP collision warning caused by threadpoolctl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_script(script_path, args_list=[]):
    print(f"\n{'='*60}")
    print(f"Running {os.path.basename(script_path)} {' '.join(args_list)}...")
    print(f"{'='*60}\n")
    
    # Run the script and stream the output
    cmd = [sys.executable, script_path] + args_list
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(script_path)))
    
    if result.returncode != 0:
        print(f"\n[ERROR] An error occurred while running {os.path.basename(script_path)}. Exiting pipeline.")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description='Training Pipeline Orchestrator')
    parser.add_argument('--no-augmentation', action='store_true', help='Exclude augmented data from training')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the sequence of scripts to run
    pipeline_scripts = [
        ("train_01_feature_engineering.py", ["--no-augmentation"] if args.no_augmentation else []),
        ("train_02_model_training.py", [])
    ]
    
    print("Starting the Machine Sound Classification Training Pipeline...")
    
    for script_name, script_args in pipeline_scripts:
        script_path = os.path.join(script_dir, script_name)
        run_script(script_path, script_args)
        
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
