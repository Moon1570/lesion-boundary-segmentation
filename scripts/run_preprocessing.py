#!/usr/bin/env python3
"""
Script to run the full preprocessing pipeline on the ISIC2018 dataset.
"""

import subprocess
import sys
from pathlib import Path

def run_preprocessing(hair_removal=False):
    """Run the preprocessing pipeline."""
    
    # Base command
    cmd = [
        sys.executable,
        "scripts/preprocess.py",
        "--input_dir", "data/ISIC2018",
        "--output_dir", "data/ISIC2018_proc",
        "--target_size", "384"
    ]
    
    # Add hair removal if specified
    if hair_removal:
        cmd.extend(["--hair-removal", "dullrazor"])
        print("Running preprocessing WITH DullRazor hair removal...")
    else:
        print("Running preprocessing WITHOUT hair removal...")
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Preprocessing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed with error code {e.returncode}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ISIC2018 preprocessing")
    parser.add_argument("--hair-removal", action="store_true",
                       help="Apply DullRazor hair removal")
    
    args = parser.parse_args()
    
    success = run_preprocessing(hair_removal=args.hair_removal)
    sys.exit(0 if success else 1)
