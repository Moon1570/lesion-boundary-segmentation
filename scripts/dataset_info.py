#!/usr/bin/env python3
"""
Dataset information and statistics utility for ISIC2018 preprocessing.
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict


def analyze_dataset(data_dir: Path, processed: bool = False):
    """Analyze dataset and print comprehensive statistics."""
    
    print(f"Dataset Analysis: {data_dir}")
    print("=" * 50)
    
    # Check directory structure
    expected_dirs = ["train_images", "val_images", "test_images"]
    if not processed:
        expected_dirs.append("train_masks")
    
    stats = defaultdict(int)
    
    for split_dir in expected_dirs:
        dir_path = data_dir / split_dir
        if dir_path.exists():
            if "mask" in split_dir:
                files = list(dir_path.glob("*.png"))
            else:
                files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            stats[split_dir] = len(files)
            print(f"{split_dir:15}: {len(files):5d} files")
        else:
            print(f"{split_dir:15}: NOT FOUND")
    
    print()
    
    # Analyze image dimensions if raw dataset
    if not processed and (data_dir / "train_images").exists():
        print("Image Dimension Analysis:")
        print("-" * 30)
        
        train_images = list((data_dir / "train_images").glob("*.jpg"))[:100]  # Sample first 100
        dimensions = []
        
        for img_path in train_images:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                dimensions.append((h, w))
        
        if dimensions:
            heights, widths = zip(*dimensions)
            print(f"Sample size: {len(dimensions)} images")
            print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.1f}")
            print(f"Width  - Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.1f}")
            
            # Most common dimensions
            dim_counts = defaultdict(int)
            for dim in dimensions:
                dim_counts[dim] += 1
            
            print("\nMost common dimensions:")
            for dim, count in sorted(dim_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {dim[0]}×{dim[1]}: {count} images")
    
    # Check for dataset statistics file
    stats_file = data_dir / "dataset_stats.json"
    if stats_file.exists():
        print(f"\nDataset Statistics ({stats_file}):")
        print("-" * 30)
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        for key, value in stats_data.items():
            print(f"{key}: {value:.4f}")
    
    print()
    
    return stats


def compare_datasets(raw_dir: Path, processed_dir: Path):
    """Compare raw and processed datasets."""
    
    print("Dataset Comparison")
    print("=" * 50)
    
    print("Raw Dataset:")
    raw_stats = analyze_dataset(raw_dir, processed=False)
    
    print("\nProcessed Dataset:")
    processed_stats = analyze_dataset(processed_dir, processed=True)
    
    print("\nComparison Summary:")
    print("-" * 30)
    
    for split in ["train_images", "val_images", "test_images"]:
        raw_count = raw_stats.get(split, 0)
        proc_count = processed_stats.get(split, 0)
        
        if raw_count > 0:
            match_pct = (proc_count / raw_count) * 100
            status = "✓" if proc_count == raw_count else "⚠"
            print(f"{split:15}: {raw_count:5d} → {proc_count:5d} ({match_pct:5.1f}%) {status}")
        else:
            print(f"{split:15}: N/A")
    
    # Check masks
    raw_masks = raw_stats.get("train_masks", 0)
    proc_masks = processed_stats.get("train_masks", 0)
    if raw_masks > 0:
        match_pct = (proc_masks / raw_masks) * 100
        status = "✓" if proc_masks == raw_masks else "⚠"
        print(f"{'train_masks':15}: {raw_masks:5d} → {proc_masks:5d} ({match_pct:5.1f}%) {status}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ISIC2018 dataset")
    parser.add_argument("--raw_dir", type=str, default="data/ISIC2018",
                       help="Raw dataset directory")
    parser.add_argument("--processed_dir", type=str, default="data/ISIC2018_proc",
                       help="Processed dataset directory")
    parser.add_argument("--compare", action="store_true",
                       help="Compare raw and processed datasets")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    
    if args.compare:
        if raw_dir.exists() and processed_dir.exists():
            compare_datasets(raw_dir, processed_dir)
        else:
            print("Error: Both raw and processed directories must exist for comparison")
    else:
        # Analyze individual datasets
        if raw_dir.exists():
            analyze_dataset(raw_dir, processed=False)
        
        if processed_dir.exists():
            print("\n")
            analyze_dataset(processed_dir, processed=True)


if __name__ == "__main__":
    main()
