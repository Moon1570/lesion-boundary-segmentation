#!/usr/bin/env python3
"""
Test preprocessing script that processes only a small subset of images for validation.
"""

import sys
import os
import shutil
from pathlib import Path
import subprocess

def create_test_subset(n_images=50):
    """Create a test subset with n_images for testing preprocessing."""
    
    source_dir = Path("data/ISIC2018")
    test_dir = Path("data/ISIC2018_test")
    
    # Remove existing test directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create test directory structure
    for split in ["train_images", "train_masks", "val_images", "test_images"]:
        (test_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Copy subset of training images and masks
    train_images = list((source_dir / "train_images").glob("*.jpg"))[:n_images]
    
    for img_path in train_images:
        # Copy image
        shutil.copy2(img_path, test_dir / "train_images")
        
        # Copy corresponding mask
        mask_path = source_dir / "train_masks" / f"{img_path.stem}_segmentation.png"
        if mask_path.exists():
            shutil.copy2(mask_path, test_dir / "train_masks")
    
    # Copy subset of validation images
    val_images = list((source_dir / "val_images").glob("*.jpg"))[:10]
    for img_path in val_images:
        shutil.copy2(img_path, test_dir / "val_images")
    
    # Copy subset of test images
    test_images = list((source_dir / "test_images").glob("*.jpg"))[:10]
    for img_path in test_images:
        shutil.copy2(img_path, test_dir / "test_images")
    
    print(f"Created test subset with:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Test images: {len(test_images)}")
    
    return test_dir

def test_preprocessing(hair_removal=False):
    """Test preprocessing on a small subset."""
    
    # Create test subset
    test_dir = create_test_subset(50)
    
    # Base command
    cmd = [
        sys.executable,
        "scripts/preprocess.py",
        "--input_dir", str(test_dir),
        "--output_dir", "data/ISIC2018_test_proc",
        "--target_size", "384"
    ]
    
    # Add hair removal if specified
    if hair_removal:
        cmd.extend(["--hair-removal", "dullrazor"])
        print("Testing preprocessing WITH DullRazor hair removal...")
    else:
        print("Testing preprocessing WITHOUT hair removal...")
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Test preprocessing completed successfully!")
        
        # Check output
        output_dir = Path("data/ISIC2018_test_proc")
        if output_dir.exists():
            train_processed = len(list((output_dir / "train_images").glob("*.png")))
            masks_processed = len(list((output_dir / "train_masks").glob("*.png")))
            print(f"  Processed {train_processed} training images")
            print(f"  Processed {masks_processed} training masks")
            
            # Check if stats file was created
            stats_file = output_dir / "dataset_stats.json"
            if stats_file.exists():
                print(f"  Dataset statistics saved to {stats_file}")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test preprocessing failed with error code {e.returncode}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ISIC2018 preprocessing on subset")
    parser.add_argument("--hair-removal", action="store_true",
                       help="Apply DullRazor hair removal")
    
    args = parser.parse_args()
    
    success = test_preprocessing(hair_removal=args.hair_removal)
    sys.exit(0 if success else 1)
