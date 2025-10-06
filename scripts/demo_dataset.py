#!/usr/bin/env python3
"""
Demonstration script for ISIC2018 dataset augmentations.
Shows the effect of different augmentations on images and masks.
"""

import sys
sys.path.append('scripts')

import numpy as np
import matplotlib.pyplot as plt
from dataset import ISIC2018Dataset
import torch

def demonstrate_augmentations():
    """Demonstrate the effect of augmentations on the same image multiple times."""
    
    print("🎨 Demonstrating Dataset Augmentations")
    print("=" * 50)
    
    # Create datasets
    train_dataset = ISIC2018Dataset(split='train', augment=True)
    val_dataset = ISIC2018Dataset(split='val', augment=False)
    
    print(f"Training dataset: {len(train_dataset)} samples (with augmentations)")
    print(f"Validation dataset: {len(val_dataset)} samples (no augmentations)")
    
    # Get the same image multiple times to show different augmentations
    image_idx = 0
    num_augmentations = 8
    
    # Create figure
    fig, axes = plt.subplots(3, num_augmentations, figsize=(20, 12))
    
    # Get original (validation - no augmentation)
    val_sample = val_dataset[image_idx]
    original_image = val_sample['image']
    original_mask = val_sample['mask']
    
    # Denormalize for visualization
    mean = train_dataset.stats['mean']
    std = train_dataset.stats['std']
    
    def denormalize_image(img_tensor):
        """Denormalize image for visualization."""
        img = img_tensor.clone()
        for c in range(3):
            img[c] = img[c] * std + mean
        return torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    
    # Show original
    orig_img_np = denormalize_image(original_image)
    orig_mask_np = original_mask.squeeze().numpy()
    
    axes[0, 0].imshow(orig_img_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(orig_mask_np, cmap='gray')
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis('off')
    
    axes[2, 0].imshow(orig_img_np)
    axes[2, 0].imshow(orig_mask_np, alpha=0.3, cmap='Reds')
    axes[2, 0].set_title("Original Overlay")
    axes[2, 0].axis('off')
    
    # Show augmented versions
    for i in range(1, num_augmentations):
        # Get augmented sample (same image, different augmentation)
        aug_sample = train_dataset[image_idx]
        aug_image = aug_sample['image']
        aug_mask = aug_sample['mask']
        
        # Denormalize
        aug_img_np = denormalize_image(aug_image)
        aug_mask_np = aug_mask.squeeze().numpy()
        
        # Show augmented image
        axes[0, i].imshow(aug_img_np)
        axes[0, i].set_title(f"Augmented {i}")
        axes[0, i].axis('off')
        
        # Show augmented mask
        axes[1, i].imshow(aug_mask_np, cmap='gray')
        axes[1, i].set_title(f"Aug Mask {i}")
        axes[1, i].axis('off')
        
        # Show overlay
        axes[2, i].imshow(aug_img_np)
        axes[2, i].imshow(aug_mask_np, alpha=0.3, cmap='Reds')
        axes[2, i].set_title(f"Aug Overlay {i}")
        axes[2, i].axis('off')
    
    plt.suptitle(f"Augmentation Effects on Image: {val_sample['image_id']}", fontsize=20)
    plt.tight_layout()
    
    # Ensure runs/figs directory exists
    from pathlib import Path
    figs_dir = Path("runs/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = figs_dir / "augmentation_demonstration.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Augmentation demonstration saved as '{save_path}'")

def analyze_dataset_statistics():
    """Analyze dataset statistics and class distribution."""
    
    print("\n📊 Dataset Statistics Analysis")
    print("=" * 50)
    
    # Load datasets
    train_dataset = ISIC2018Dataset(split='train', augment=False)
    val_dataset = ISIC2018Dataset(split='val', augment=False)
    
    def analyze_masks(dataset, split_name):
        """Analyze mask statistics for a dataset split."""
        print(f"\n{split_name} Split Analysis:")
        
        mask_areas = []
        total_pixels = 384 * 384
        
        for i in range(min(100, len(dataset))):  # Sample first 100 for speed
            sample = dataset[i]
            mask = sample['mask'].squeeze().numpy()
            
            # Calculate mask area (percentage of lesion pixels)
            lesion_pixels = np.sum(mask > 0.5)
            mask_area = lesion_pixels / total_pixels
            mask_areas.append(mask_area)
        
        mask_areas = np.array(mask_areas)
        
        print(f"  Samples analyzed: {len(mask_areas)}")
        print(f"  Lesion area (% of image):")
        print(f"    Mean: {mask_areas.mean()*100:.2f}%")
        print(f"    Median: {np.median(mask_areas)*100:.2f}%")
        print(f"    Min: {mask_areas.min()*100:.2f}%")
        print(f"    Max: {mask_areas.max()*100:.2f}%")
        print(f"    Std: {mask_areas.std()*100:.2f}%")
        
        return mask_areas
    
    # Analyze both splits
    train_areas = analyze_masks(train_dataset, "Training")
    val_areas = analyze_masks(val_dataset, "Validation")
    
    # Create histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_areas * 100, bins=20, alpha=0.7, label='Train', color='blue')
    plt.hist(val_areas * 100, bins=20, alpha=0.7, label='Val', color='orange')
    plt.xlabel('Lesion Area (% of image)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lesion Areas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([train_areas * 100, val_areas * 100], labels=['Train', 'Val'])
    plt.ylabel('Lesion Area (% of image)')
    plt.title('Lesion Area Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure runs/figs directory exists
    from pathlib import Path
    figs_dir = Path("runs/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = figs_dir / "dataset_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Dataset statistics saved as '{save_path}'")

def test_data_loading_speed():
    """Test data loading performance."""
    
    print("\n⚡ Data Loading Performance Test")
    print("=" * 50)
    
    from torch.utils.data import DataLoader
    import time
    
    # Create datasets
    train_dataset = ISIC2018Dataset(split='train', augment=True)
    val_dataset = ISIC2018Dataset(split='val', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    def time_loading(loader, name, max_batches=10):
        """Time the loading of batches."""
        print(f"\nTiming {name} loader...")
        
        start_time = time.time()
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            
            # Simulate some processing
            images = batch['image']
            masks = batch['mask']
            
            if i == 0:
                print(f"  Batch shape: {images.shape}")
                print(f"  Mask shape: {masks.shape}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"  Loaded {min(max_batches, len(loader))} batches in {elapsed:.2f}s")
        print(f"  Average time per batch: {elapsed/min(max_batches, len(loader)):.3f}s")
        
        return elapsed
    
    # Time both loaders
    train_time = time_loading(train_loader, "Training", max_batches=10)
    val_time = time_loading(val_loader, "Validation", max_batches=10)
    
    print(f"\nPerformance Summary:")
    print(f"  Training loader: {train_time:.2f}s for 10 batches")
    print(f"  Validation loader: {val_time:.2f}s for 10 batches")

def main():
    """Main demonstration function."""
    
    print("🚀 ISIC2018 Dataset Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Test basic functionality
        print("Testing basic dataset functionality...")
        train_dataset = ISIC2018Dataset(split='train', augment=True)
        val_dataset = ISIC2018Dataset(split='val', augment=False)
        
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Val dataset: {len(val_dataset)} samples")
        
        # Demonstrate augmentations
        demonstrate_augmentations()
        
        # Analyze statistics
        analyze_dataset_statistics()
        
        # Test loading speed
        test_data_loading_speed()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
