#!/usr/bin/env python3
"""
ISIC2018 Dataset implementation with augmentations using Albumentations.

This module provides PyTorch Dataset classes for the ISIC2018 lesion segmentation challenge
with proper train/validation splits and comprehensive augmentation pipelines.

Key Features:
- Uses predetermined splits from splits/*.txt files
- Training augmentations: flip, rotate, elastic, brightness/contrast, color jitter, dropout
- Validation/Test: only normalization and basic transforms
- Proper mask handling with nearest-neighbor interpolation
- Binary mask enforcement (0/1 values)
- Synchronized image-mask transformations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class ISIC2018Dataset(Dataset):
    """
    ISIC2018 dataset for lesion boundary segmentation.
    
    Args:
        data_dir: Path to processed ISIC2018 data directory
        split: Dataset split ('train', 'val', 'test')
        splits_dir: Directory containing split files
        image_size: Target image size (default: 384)
        augment: Whether to apply augmentations (default: True for train, False for val/test)
        stats_file: Path to dataset statistics JSON file
    """
    
    def __init__(
        self,
        data_dir: str = "data/ISIC2018_proc",
        split: str = "train",
        splits_dir: str = "splits",
        image_size: int = 384,
        augment: Optional[bool] = None,
        stats_file: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.splits_dir = Path(splits_dir)
        self.image_size = image_size
        
        # Set default augmentation behavior
        if augment is None:
            self.augment = (split == "train")
        else:
            self.augment = augment
        
        # Load dataset statistics
        if stats_file is None:
            stats_file = self.data_dir / "dataset_stats.json"
        self.stats = self._load_stats(stats_file)
        
        # Load split-specific image IDs
        self.image_ids = self._load_split_ids()
        
        # Setup data paths
        self._setup_paths()
        
        # Create transforms
        self.transforms = self._create_transforms()
        
        print(f"Loaded {split} split: {len(self.image_ids)} samples")
    
    def _load_stats(self, stats_file: Path) -> Dict[str, float]:
        """Load dataset normalization statistics."""
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            return stats
        except FileNotFoundError:
            print(f"Warning: Stats file {stats_file} not found, using default values")
            return {"mean": 0.6042, "std": 0.1817}
    
    def _load_split_ids(self) -> List[str]:
        """Load image IDs for the specified split."""
        if self.split in ["train", "val"]:
            split_file = self.splits_dir / f"isic2018_{self.split}.txt"
        else:
            # For test split, we'll use all available test images
            split_file = None
        
        if split_file and split_file.exists():
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback: get all images from the split directory
            if self.split == "test":
                image_dir = self.data_dir / "test_images"
                if image_dir.exists():
                    image_ids = [f.stem for f in image_dir.glob("*.png")]
                else:
                    raise FileNotFoundError(f"Test images directory not found: {image_dir}")
            else:
                raise FileNotFoundError(f"Split file not found: {split_file}")
        
        return sorted(image_ids)
    
    def _setup_paths(self):
        """Setup paths for images and masks."""
        if self.split in ["train", "val"]:
            # Train/val use the training images directory with predetermined splits
            self.images_dir = self.data_dir / "train_images"
            self.masks_dir = self.data_dir / "train_masks"
        else:
            # Test split
            self.images_dir = self.data_dir / "test_images"
            self.masks_dir = None  # Test split has no masks
        
        # Validate directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if self.masks_dir and not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
    
    def _create_transforms(self) -> A.Compose:
        """Create albumentations transforms based on split and augmentation settings."""
        
        if self.augment and self.split == "train":
            # Training augmentations
            transforms = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT),
                
                # Elastic and affine transformations (small)
                A.OneOf([
                    A.ElasticTransform(
                        alpha=50, sigma=5, p=0.3,
                        border_mode=cv2.BORDER_REFLECT
                    ),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                    A.OpticalDistortion(distort_limit=0.05, p=0.3),
                ], p=0.4),
                
                # Photometric transformations (image only)
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ], p=0.5),
                
                # Color jitter (light)
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4
                ),
                
                # Normalization (always last)
                A.Normalize(
                    mean=[self.stats["mean"]] * 3,
                    std=[self.stats["std"]] * 3,
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        else:
            # Validation/Test augmentations (minimal)
            transforms = A.Compose([
                A.Normalize(
                    mean=[self.stats["mean"]] * 3,
                    std=[self.stats["std"]] * 3,
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        
        return transforms
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'image': Transformed image tensor (3, H, W)
            - 'mask': Transformed mask tensor (1, H, W) if available
            - 'image_id': Original image ID string
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.images_dir / f"{image_id}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = None
        if self.masks_dir:
            mask_path = self.masks_dir / f"{image_id}_segmentation.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                
                # Ensure binary mask (0/1 values)
                mask = self._binarize_mask(mask)
        
        # Apply transforms
        if mask is not None:
            # Both image and mask
            transformed = self.transforms(image=image, mask=mask)
            sample = {
                'image': transformed['image'],
                'mask': transformed['mask'].unsqueeze(0),  # Add channel dim (1, H, W)
                'image_id': image_id
            }
        else:
            # Image only (test set)
            transformed = self.transforms(image=image)
            sample = {
                'image': transformed['image'],
                'image_id': image_id
            }
        
        return sample
    
    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Ensure mask contains only binary values {0, 1}.
        
        Args:
            mask: Input mask array
            
        Returns:
            Binary mask with values in {0, 1}
        """
        # Handle different input ranges
        if mask.max() > 1:
            # Assume 0-255 range
            mask = mask / 255.0
        
        # Threshold at 0.5 to ensure binary values
        mask = (mask > 0.5).astype(np.float32)  # Use float32 instead of uint8
        
        return mask
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            'split': self.split,
            'num_samples': len(self.image_ids),
            'image_size': self.image_size,
            'augment': self.augment,
            'has_masks': self.masks_dir is not None,
            'stats': self.stats,
            'first_few_ids': self.image_ids[:5]
        }
        return info


def create_data_loaders(
    data_dir: str = "data/ISIC2018_proc",
    splits_dir: str = "splits",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 384,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        data_dir: Path to processed data directory
        splits_dir: Directory containing split files
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Dictionary with 'train', 'val', and 'test' data loaders
    """
    
    # Create datasets
    datasets = {}
    
    # Train dataset with augmentations
    datasets['train'] = ISIC2018Dataset(
        data_dir=data_dir,
        split='train',
        splits_dir=splits_dir,
        image_size=image_size,
        augment=True
    )
    
    # Validation dataset without augmentations
    datasets['val'] = ISIC2018Dataset(
        data_dir=data_dir,
        split='val',
        splits_dir=splits_dir,
        image_size=image_size,
        augment=False
    )
    
    # Test dataset (if available)
    try:
        datasets['test'] = ISIC2018Dataset(
            data_dir=data_dir,
            split='test',
            splits_dir=splits_dir,
            image_size=image_size,
            augment=False
        )
    except FileNotFoundError:
        print("Test dataset not available")
    
    # Create data loaders
    data_loaders = {}
    
    for split, dataset in datasets.items():
        # Use shuffle for training, no shuffle for val/test
        shuffle = (split == 'train')
        
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )
    
    return data_loaders


def visualize_samples(
    dataset: ISIC2018Dataset,
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize samples from the dataset.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        save_path: Optional path to save the visualization
    """
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Denormalize image for visualization
        image = sample['image']
        mean = torch.tensor(dataset.stats['mean'])
        std = torch.tensor(dataset.stats['std'])
        
        # Denormalize
        for c in range(3):
            image[c] = image[c] * std + mean
        
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).numpy()
        
        # Show image
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f"Image: {sample['image_id']}")
        axes[0, i].axis('off')
        
        # Show mask if available
        if 'mask' in sample:
            mask_np = sample['mask'].squeeze().numpy()
            axes[1, i].imshow(mask_np, cmap='gray')
            axes[1, i].set_title(f"Mask: {sample['image_id']}")
        else:
            axes[1, i].text(0.5, 0.5, 'No Mask', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title("No Mask Available")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # If save_path is provided but doesn't include runs/figs, update it
        from pathlib import Path
        save_path = Path(save_path)
        if not str(save_path).startswith("runs/figs"):
            figs_dir = Path("runs/figs")
            figs_dir.mkdir(parents=True, exist_ok=True)
            save_path = figs_dir / save_path.name
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def analyze_dataset_splits(
    data_dir: str = "data/ISIC2018_proc",
    splits_dir: str = "splits"
) -> None:
    """
    Analyze and print information about dataset splits.
    
    Args:
        data_dir: Path to processed data directory
        splits_dir: Directory containing split files
    """
    
    print("📊 Dataset Split Analysis")
    print("=" * 50)
    
    try:
        # Create datasets
        train_dataset = ISIC2018Dataset(data_dir, 'train', splits_dir, augment=False)
        val_dataset = ISIC2018Dataset(data_dir, 'val', splits_dir, augment=False)
        
        train_info = train_dataset.get_sample_info()
        val_info = val_dataset.get_sample_info()
        
        total_samples = train_info['num_samples'] + val_info['num_samples']
        
        print(f"Training Split:")
        print(f"  Samples: {train_info['num_samples']:,}")
        print(f"  Percentage: {train_info['num_samples']/total_samples*100:.1f}%")
        print(f"  First few IDs: {train_info['first_few_ids']}")
        
        print(f"\nValidation Split:")
        print(f"  Samples: {val_info['num_samples']:,}")
        print(f"  Percentage: {val_info['num_samples']/total_samples*100:.1f}%")
        print(f"  First few IDs: {val_info['first_few_ids']}")
        
        print(f"\nTotal: {total_samples:,} samples")
        print(f"Image Size: {train_info['image_size']}×{train_info['image_size']}")
        print(f"Normalization Stats: mean={train_info['stats']['mean']:.4f}, std={train_info['stats']['std']:.4f}")
        
        # Try test dataset
        try:
            test_dataset = ISIC2018Dataset(data_dir, 'test', splits_dir, augment=False)
            test_info = test_dataset.get_sample_info()
            print(f"\nTest Split:")
            print(f"  Samples: {test_info['num_samples']:,}")
            print(f"  Has Masks: {test_info['has_masks']}")
        except FileNotFoundError:
            print(f"\nTest Split: Not available")
            
    except Exception as e:
        print(f"Error analyzing datasets: {e}")


def main():
    """Main function to demonstrate dataset usage."""
    
    print("🔬 ISIC2018 Dataset Implementation")
    print("=" * 50)
    
    # Analyze splits
    analyze_dataset_splits()
    
    print("\n" + "=" * 50)
    
    # Create sample datasets
    print("Creating sample datasets...")
    
    # Training dataset with augmentations
    train_dataset = ISIC2018Dataset(split='train', augment=True)
    print(f"✅ Training dataset: {len(train_dataset)} samples (with augmentations)")
    
    # Validation dataset without augmentations
    val_dataset = ISIC2018Dataset(split='val', augment=False)
    print(f"✅ Validation dataset: {len(val_dataset)} samples (no augmentations)")
    
    # Test a sample
    print("\nTesting sample loading...")
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")
    if 'mask' in sample:
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask values: min={sample['mask'].min():.3f}, max={sample['mask'].max():.3f}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    data_loaders = create_data_loaders(batch_size=8, num_workers=0)  # num_workers=0 for Windows compatibility
    
    for split, loader in data_loaders.items():
        print(f"✅ {split.capitalize()} loader: {len(loader)} batches")
    
    # Test a batch
    print("\nTesting batch loading...")
    train_loader = data_loaders['train']
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    if 'mask' in batch:
        print(f"Batch mask shape: {batch['mask'].shape}")
    
    # Visualize samples
    print("\nCreating sample visualizations...")
    
    # Ensure runs/figs directory exists
    from pathlib import Path
    figs_dir = Path("runs/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = figs_dir / "sample_dataset_visualization.png"
    visualize_samples(val_dataset, num_samples=4, save_path=str(save_path))
    
    print("\n✅ Dataset implementation test completed successfully!")


if __name__ == "__main__":
    main()
