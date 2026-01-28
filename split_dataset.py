import pandas as pd
import numpy as np
import os

def split_dataset(input_csv, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    # Verify ratios sum to 1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Shuffle the dataset
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Total samples: {n}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits to {output_dir}")

if __name__ == "__main__":
    input_csv = os.path.join('dataset', 'train_prototype.csv')
    output_dir = 'dataset'
    split_dataset(input_csv, output_dir)
