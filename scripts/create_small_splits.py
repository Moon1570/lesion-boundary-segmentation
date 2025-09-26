#!/usr/bin/env python3
"""
Create small split files for quick training and testing.
"""

import os
import random
from pathlib import Path

def create_small_splits(base_splits_dir: str = "splits", output_dir: str = "splits_small", 
                       train_size: int = 200, val_size: int = 50):
    """
    Create smaller split files from existing splits.
    
    Args:
        base_splits_dir: Directory with original split files
        output_dir: Directory to save small split files
        train_size: Number of training samples
        val_size: Number of validation samples
    """
    
    base_path = Path(base_splits_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📁 Creating small splits in {output_dir}/")
    print(f"   Train size: {train_size}, Val size: {val_size}")
    
    # Read original splits
    original_train = []
    original_val = []
    
    train_file = base_path / "isic2018_train.txt"
    val_file = base_path / "isic2018_val.txt"
    
    if train_file.exists():
        with open(train_file, 'r') as f:
            original_train = [line.strip() for line in f if line.strip()]
    
    if val_file.exists():
        with open(val_file, 'r') as f:
            original_val = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Original splits: Train={len(original_train)}, Val={len(original_val)}")
    
    # Create smaller subsets
    random.seed(42)  # For reproducibility
    
    # Sample smaller subsets
    small_train = random.sample(original_train, min(train_size, len(original_train)))
    small_val = random.sample(original_val, min(val_size, len(original_val)))
    
    # Write small split files
    with open(output_path / "train.txt", 'w') as f:
        for item in small_train:
            f.write(f"{item}\n")
    
    with open(output_path / "val.txt", 'w') as f:
        for item in small_val:
            f.write(f"{item}\n")
    
    # Copy test split as-is (usually smaller anyway)
    test_file = base_path / "test.txt"
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_data = f.read()
        
        with open(output_path / "test.txt", 'w') as f:
            f.write(test_data)
        
        test_count = len([line for line in test_data.split('\n') if line.strip()])
        print(f"📋 Copied test split: {test_count} samples")
    
    print(f"✅ Small splits created:")
    print(f"   📝 {output_path}/train.txt: {len(small_train)} samples")
    print(f"   📝 {output_path}/val.txt: {len(small_val)} samples")
    
    return len(small_train), len(small_val)

if __name__ == "__main__":
    create_small_splits(
        base_splits_dir="splits",
        output_dir="splits_small",
        train_size=200,
        val_size=50
    )
