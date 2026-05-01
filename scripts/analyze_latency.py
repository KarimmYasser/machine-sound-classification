from pathlib import Path
import numpy as np

def main():
    time_file = Path("time.txt")
    
    if not time_file.exists():
        print("Error: time.txt not found.")
        return
        
    try:
        with open(time_file, 'r') as f:
            # Skip empty lines and convert to float
            times = [float(line.strip()) for line in f if line.strip()]
            
        if not times:
            print("Error: time.txt is empty.")
            return
            
        avg_time = np.mean(times)
        total_time = np.sum(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        print("\n" + "="*40)
        print("LATENCY PERFORMANCE REPORT")
        print("="*40)
        print(f"Total Samples:    {len(times)}")
        print(f"Average Latency:  {avg_time:.4f}s")
        print(f"Total Pipeline:   {total_time:.2f}s")
        print(f"Min / Max:        {min_time:.4f}s / {max_time:.4f}s")
        print("="*40 + "\n")
        
    except ValueError as e:
        print(f"Error parsing time.txt: {e}")

if __name__ == "__main__":
    main()
