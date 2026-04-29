import pandas as pd
import os

def create_unified_dataset():
    raw_path = r"d:\Projects\machine-sound-classification\data\kaggle_extracted_features_raw.csv"
    processed_path = r"d:\Projects\machine-sound-classification\data\kaggle_extracted_features.csv"
    output_path = r"d:\Projects\machine-sound-classification\data\kaggle_unified_features.csv"

    print(f"Loading {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    # Add processing type for raw data
    df_raw['processing_type'] = 'raw'
    
    print(f"Loading {processed_path}...")
    df_processed = pd.read_csv(processed_path)
    
    # We need to assign 'denoised', 'added_noise', 'time_shifted' to the triplets
    # The user says they come in order: False (denoised), True (added), True (shifted)
    
    print("Labeling processing types in augmented dataset...")
    # Group by file_path to ensure we are looking at the same audio file's versions
    # We'll use a cumulative count within each group to distinguish the versions
    df_processed['version_idx'] = df_processed.groupby('file_path').cumcount()
    
    def map_version(idx):
        if idx == 0:
            return 'denoised'
        elif idx == 1:
            return 'added_noise'
        elif idx == 2:
            return 'time_shifted'
        else:
            return f'extra_{idx}'

    df_processed['processing_type'] = df_processed['version_idx'].apply(map_version)
    df_processed.drop(columns=['version_idx'], inplace=True)
    
    print("Combining datasets...")
    # Combine the two dataframes
    df_unified = pd.concat([df_raw, df_processed], ignore_index=True)
    
    # Sort to keep versions of the same file together
    # Order: raw, denoised, added_noise, time_shifted
    type_order = {'raw': 0, 'denoised': 1, 'added_noise': 2, 'time_shifted': 3}
    df_unified['type_rank'] = df_unified['processing_type'].map(lambda x: type_order.get(x, 4))
    
    print("Sorting by file path and processing type...")
    df_unified.sort_values(by=['file_path', 'type_rank'], inplace=True)
    df_unified.drop(columns=['type_rank'], inplace=True)
    
    print(f"Saving unified dataset to {output_path}...")
    df_unified.to_csv(output_path, index=False)
    
    print("Summary:")
    print(df_unified['processing_type'].value_counts())
    print(f"Total rows: {len(df_unified)}")
    print("Done!")

if __name__ == "__main__":
    create_unified_dataset()
