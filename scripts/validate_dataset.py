#!/usr/bin/env python3
"""
Dataset validation and analysis utility for ISIC2018.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')

from dataset import ISICDataModule, binarize_predictions


def validate_splits():
    """Validate the train/val splits."""
    
    splits_dir = Path("splits")
    
    # Load splits
    with open(splits_dir / "isic2018_train.txt", 'r') as f:
        train_ids = set(line.strip() for line in f)
    
    with open(splits_dir / "isic2018_val.txt", 'r') as f:
        val_ids = set(line.strip() for line in f)
    
    print("Split Validation:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val: {len(val_ids)} samples")
    print(f"  Total: {len(train_ids) + len(val_ids)} samples")
    print(f"  Ratio: {len(train_ids)/(len(train_ids)+len(val_ids))*100:.1f}% train, {len(val_ids)/(len(train_ids)+len(val_ids))*100:.1f}% val")
    
    # Check for overlap
    overlap = train_ids.intersection(val_ids)
    if overlap:
        print(f"  ❌ OVERLAP DETECTED: {len(overlap)} samples in both splits!")
        for sample in list(overlap)[:5]:
            print(f"    {sample}")
    else:
        print("  ✅ No overlap between train and val splits")
    
    return len(train_ids), len(val_ids)


def analyze_mask_distribution(data_module):
    """Analyze the distribution of mask values."""
    
    print("\nMask Distribution Analysis:")
    
    # Setup datasets
    data_module.setup("fit")
    
    # Analyze training masks
    train_loader = data_module.train_dataloader()
    
    mask_stats = {
        'num_samples': 0,
        'foreground_pixels': 0,
        'total_pixels': 0,
        'min_foreground': float('inf'),
        'max_foreground': 0,
    }
    
    print("  Analyzing training masks...")
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Analyze first 10 batches
            break
        
        masks = batch['mask']
        batch_size = masks.shape[0]
        
        mask_stats['num_samples'] += batch_size
        
        for j in range(batch_size):
            mask = masks[j]
            total_pixels = mask.numel()
            foreground_pixels = mask.sum().item()
            
            mask_stats['total_pixels'] += total_pixels
            mask_stats['foreground_pixels'] += foreground_pixels
            mask_stats['min_foreground'] = min(mask_stats['min_foreground'], foreground_pixels)
            mask_stats['max_foreground'] = max(mask_stats['max_foreground'], foreground_pixels)
    
    # Calculate statistics
    foreground_ratio = mask_stats['foreground_pixels'] / mask_stats['total_pixels']
    
    print(f"  Samples analyzed: {mask_stats['num_samples']}")
    print(f"  Foreground ratio: {foreground_ratio:.4f} ({foreground_ratio*100:.2f}%)")
    print(f"  Min foreground pixels: {mask_stats['min_foreground']}")
    print(f"  Max foreground pixels: {mask_stats['max_foreground']}")
    
    return mask_stats


def test_augmentations():
    """Test augmentation pipeline with visualizations."""
    
    print("\nTesting Augmentations:")
    
    # Initialize data module
    data_module = ISICDataModule(
        data_dir="data/ISIC2018_proc",
        splits_dir="splits",
        batch_size=1
    )
    data_module.setup("fit")
    
    # Get a sample
    sample_idx = 0
    original_sample = data_module.train_dataset[sample_idx]
    
    # Create multiple augmented versions
    augmented_samples = []
    for i in range(4):
        augmented_sample = data_module.train_dataset[sample_idx]
        augmented_samples.append(augmented_sample)
    
    # Visualize augmentations
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Denormalization function
    def denormalize(tensor, mean, std):
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)
    
    mean, std = data_module.train_dataset.mean, data_module.train_dataset.std
    
    # Plot original and augmented versions
    for i, sample in enumerate([original_sample] + augmented_samples):
        # Image
        image = denormalize(sample['image'], mean, std)
        image = image.permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Image {i if i > 0 else 'Original'}")
        axes[0, i].axis('off')
        
        # Mask
        mask = sample['mask'].numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Mask {i if i > 0 else 'Original'}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_test.png", dpi=150, bbox_inches='tight')
    print("  Augmentation visualization saved to: augmentation_test.png")
    plt.close()


def test_binarization():
    """Test prediction binarization."""
    
    print("\nTesting Binarization:")
    
    # Create mock predictions
    mock_predictions = torch.tensor([
        [0.1, 0.3, 0.6, 0.8],
        [0.2, 0.4, 0.7, 0.9],
        [0.05, 0.45, 0.55, 0.95]
    ])
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        binary_preds = binarize_predictions(mock_predictions, threshold)
        print(f"  Threshold {threshold}: {binary_preds.tolist()}")


def main():
    """Main validation function."""
    
    print("🔍 ISIC2018 Dataset Validation")
    print("=" * 50)
    
    # 1. Validate splits
    train_count, val_count = validate_splits()
    
    # 2. Initialize data module
    data_module = ISICDataModule(
        data_dir="data/ISIC2018_proc",
        splits_dir="splits",
        batch_size=8
    )
    
    # 3. Analyze mask distribution
    mask_stats = analyze_mask_distribution(data_module)
    
    # 4. Test augmentations
    test_augmentations()
    
    # 5. Test binarization
    test_binarization()
    
    print("\n✅ Dataset validation completed successfully!")
    print(f"   Ready for training with {train_count} train, {val_count} val samples")


if __name__ == "__main__":
    main()
