#!/usr/bin/env python3
"""
Advanced Data Augmentation Script for ISIC2018 Dataset.

This script creates augmented versions of the training data to increase dataset size
and improve model generalization. It applies various augmentation techniques while
preserving the mask-image correspondence.

Features:
- Multiple augmentation techniques
- Preserves image-mask correspondence
- Creates balanced augmented dataset
- Generates new train/val/test splits
- Progress tracking and logging
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import json
import random
from typing import Tuple, List, Dict
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_augmentation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataAugmenter:
    """Advanced data augmentation for skin lesion segmentation."""
    
    def __init__(self, 
                 source_dir: str = "data/ISIC2018_proc",
                 target_dir: str = "data/ISIC2018_proc_augmented",
                 augmentation_factor: int = 3,
                 image_size: int = 384):
        """
        Initialize the data augmenter.
        
        Args:
            source_dir: Source dataset directory
            target_dir: Target directory for augmented data
            augmentation_factor: Number of augmented versions per original image
            image_size: Target image size for resizing
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.augmentation_factor = augmentation_factor
        self.image_size = image_size
        
        # Create target directories
        self.setup_directories()
        
        # Define augmentation pipelines
        self.setup_augmentation_pipelines()
        
    def setup_directories(self):
        """Create target directory structure."""
        directories = [
            "train_images", "train_masks",
            "val_images", "val_masks", 
            "test_images", "test_masks"
        ]
        
        for dir_name in directories:
            (self.target_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created augmented dataset directories in: {self.target_dir}")
    
    def setup_augmentation_pipelines(self):
        """Define various augmentation pipelines."""
        
        # Pipeline 1: Geometric transformations
        self.geometric_pipeline = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=30, 
                border_mode=cv2.BORDER_REFLECT_101, 
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ElasticTransform(
                alpha=1, 
                sigma=50, 
                border_mode=cv2.BORDER_REFLECT_101, 
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5, 
                distort_limit=0.3, 
                border_mode=cv2.BORDER_REFLECT_101, 
                p=0.3
            ),
        ], additional_targets={'mask': 'mask'})
        
        # Pipeline 2: Color and lighting augmentations
        self.color_pipeline = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.7
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.ChannelShuffle(p=0.2),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.6
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        ], additional_targets={'mask': 'mask'})
        
        # Pipeline 3: Noise and blur augmentations
        self.noise_pipeline = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.OneOf([
                A.GaussNoise(var_limit=50.0, p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=7, p=0.3),
            ], p=0.3),
            A.RandomShadow(p=0.2),
        ], additional_targets={'mask': 'mask'})
        
        # Pipeline 4: Mixed augmentations (combines multiple techniques)
        self.mixed_pipeline = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.5),
            ], p=0.7),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.3),
            ], p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.2, 
                rotate_limit=15, 
                border_mode=cv2.BORDER_REFLECT_101, 
                p=0.8
            ),
        ], additional_targets={'mask': 'mask'})
        
        # Store all pipelines
        self.pipelines = {
            'geometric': self.geometric_pipeline,
            'color': self.color_pipeline,
            'noise': self.noise_pipeline,
            'mixed': self.mixed_pipeline
        }
        
        logger.info(f"Setup {len(self.pipelines)} augmentation pipelines")
    
    def load_image_and_mask(self, image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess image and mask pair."""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        return image, mask
    
    def save_augmented_pair(self, 
                           image: np.ndarray, 
                           mask: np.ndarray, 
                           save_dir: str,
                           filename: str):
        """Save augmented image and mask pair."""
        image_save_path = self.target_dir / f"{save_dir}_images" / f"{filename}.png"
        mask_save_path = self.target_dir / f"{save_dir}_masks" / f"{filename}_segmentation.png"
        
        # Save image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_save_path), image_bgr)
        
        # Save mask
        cv2.imwrite(str(mask_save_path), mask)
    
    def augment_split(self, split: str) -> Dict[str, int]:
        """Augment a specific data split."""
        image_dir = self.source_dir / f"{split}_images"
        mask_dir = self.source_dir / f"{split}_masks"
        
        if not image_dir.exists() or not mask_dir.exists():
            logger.warning(f"Split {split} not found, skipping...")
            return {}
        
        # Get all image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        stats = {
            'original': len(image_files),
            'augmented': 0,
            'total': 0
        }
        
        logger.info(f"Augmenting {split} split: {len(image_files)} original images")
        
        # First, copy original images
        for image_path in tqdm(image_files, desc=f"Copying original {split}"):
            # Handle mask filename properly for PNG images
            if image_path.suffix == '.png':
                mask_filename = image_path.stem + '_segmentation.png'
            else:
                mask_filename = image_path.stem + '_segmentation.png'
            mask_path = mask_dir / mask_filename
            
            if mask_path.exists():
                # Load original pair
                image, mask = self.load_image_and_mask(image_path, mask_path)
                
                # Resize only (no other augmentation for originals)
                resized = A.Resize(self.image_size, self.image_size)(image=image, mask=mask)
                
                # Save original (resized)
                filename = f"orig_{image_path.stem}"
                self.save_augmented_pair(
                    resized['image'], 
                    resized['mask'], 
                    split, 
                    filename
                )
        
        stats['total'] += stats['original']
        
        # Generate augmented versions
        pipeline_names = list(self.pipelines.keys())
        
        for aug_idx in range(self.augmentation_factor):
            logger.info(f"Generating augmentation round {aug_idx + 1}/{self.augmentation_factor}")
            
            for image_path in tqdm(image_files, desc=f"Augmenting {split} round {aug_idx + 1}"):
                # Handle mask filename properly for PNG images
                if image_path.suffix == '.png':
                    mask_filename = image_path.stem + '_segmentation.png'
                else:
                    mask_filename = image_path.stem + '_segmentation.png'
                mask_path = mask_dir / mask_filename
                
                if mask_path.exists():
                    # Load original pair
                    image, mask = self.load_image_and_mask(image_path, mask_path)
                    
                    # Randomly select pipeline
                    pipeline_name = random.choice(pipeline_names)
                    pipeline = self.pipelines[pipeline_name]
                    
                    # Apply augmentation
                    try:
                        augmented = pipeline(image=image, mask=mask)
                        
                        # Save augmented pair
                        filename = f"aug_{pipeline_name}_{aug_idx}_{image_path.stem}"
                        self.save_augmented_pair(
                            augmented['image'], 
                            augmented['mask'], 
                            split, 
                            filename
                        )
                        
                        stats['augmented'] += 1
                        stats['total'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error augmenting {image_path}: {e}")
        
        logger.info(f"Completed {split} augmentation: {stats}")
        return stats
    
    def create_augmented_dataset(self) -> Dict[str, Dict[str, int]]:
        """Create the complete augmented dataset."""
        logger.info("Starting data augmentation process...")
        
        all_stats = {}
        
        # Augment each split
        for split in ['train', 'val', 'test']:
            all_stats[split] = self.augment_split(split)
        
        # Save augmentation summary
        summary_path = self.target_dir / "augmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        logger.info(f"Augmentation completed! Summary saved to: {summary_path}")
        return all_stats

def create_augmented_splits(augmented_dir: str = "data/ISIC2018_augmented", 
                           splits_dir: str = "splits"):
    """Create new train/val/test splits for the augmented dataset."""
    
    augmented_path = Path(augmented_dir)
    splits_path = Path(splits_dir)
    
    # Create splits directory if it doesn't exist
    splits_path.mkdir(exist_ok=True)
    
    # Get all image files from each split
    splits = {}
    
    for split in ['train', 'val', 'test']:
        image_dir = augmented_path / f"{split}_images"
        if image_dir.exists():
            image_files = [f.stem for f in image_dir.glob("*.png") if not f.name.endswith('_segmentation.png')]
            splits[split] = sorted(image_files)
            logger.info(f"Found {len(image_files)} images in augmented {split} split")
    
    # Save new splits
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_name = f"augmented_splits_{timestamp}"
    
    for split, files in splits.items():
        split_file = splits_path / f"{split_name}_{split}.txt"
        with open(split_file, 'w') as f:
            for filename in files:
                f.write(f"{filename}\n")
        
        logger.info(f"Saved {split} split: {split_file} ({len(files)} files)")
    
    # Create a summary file
    summary = {
        "split_name": split_name,
        "created": datetime.now().isoformat(),
        "data_dir": augmented_dir,
        "statistics": {split: len(files) for split, files in splits.items()},
        "total_samples": sum(len(files) for files in splits.values())
    }
    
    summary_file = splits_path / f"{split_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Split summary saved: {summary_file}")
    logger.info(f"Total augmented samples: {summary['total_samples']}")
    
    return split_name, summary

def create_augmented_config(base_config: str = "configs/quantized_mamba_unet.json",
                           split_name: str = None,
                           augmented_dir: str = "data/ISIC2018_augmented"):
    """Create a new config file for training with augmented data."""
    
    # Load base config
    with open(base_config, 'r') as f:
        config = json.load(f)
    
    # Update config for augmented data
    config['data']['data_dir'] = augmented_dir
    if split_name:
        config['data']['splits_dir'] = f"splits/{split_name}"
    
    # Adjust training parameters for larger dataset
    original_epochs = config['training']['epochs']
    config['training']['epochs'] = max(30, original_epochs // 2)  # Fewer epochs needed with more data
    config['training']['early_stopping']['patience'] = 10  # Reduced patience
    
    # Update output directory
    model_name = config['model']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['output_dir'] = f"runs/{model_name}_augmented_{timestamp}"
    
    # Save new config
    new_config_path = f"configs/{model_name}_augmented.json"
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created augmented training config: {new_config_path}")
    return new_config_path

def main():
    """Main function to run data augmentation and setup."""
    parser = argparse.ArgumentParser(description="Augment ISIC2018 dataset")
    parser.add_argument("--source_dir", default="data/ISIC2018_advanced", 
                       help="Source dataset directory")
    parser.add_argument("--target_dir", default="data/ISIC2018_augmented", 
                       help="Target directory for augmented data")
    parser.add_argument("--augmentation_factor", type=int, default=3, 
                       help="Number of augmented versions per original image")
    parser.add_argument("--image_size", type=int, default=384, 
                       help="Target image size")
    parser.add_argument("--base_config", default="configs/quantized_mamba_unet.json",
                       help="Base config file for creating augmented config")
    
    args = parser.parse_args()
    
    # Step 1: Create augmented dataset
    logger.info("Starting data augmentation pipeline...")
    
    augmenter = DataAugmenter(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        augmentation_factor=args.augmentation_factor,
        image_size=args.image_size
    )
    
    # Generate augmented data
    stats = augmenter.create_augmented_dataset()
    
    # Step 2: Create new splits
    logger.info("Creating new data splits...")
    split_name, split_summary = create_augmented_splits(args.target_dir)
    
    # Step 3: Create training config
    logger.info("Creating training configuration...")
    config_path = create_augmented_config(args.base_config, split_name, args.target_dir)
    
    # Final summary
    logger.info("Data augmentation pipeline completed!")
    logger.info("=" * 60)
    logger.info("AUGMENTATION SUMMARY:")
    for split, split_stats in stats.items():
        if split_stats:
            logger.info(f"   {split.upper()}: {split_stats['original']} -> {split_stats['total']} " +
                       f"({split_stats['augmented']} augmented)")
    logger.info(f"Augmented data: {args.target_dir}")
    logger.info(f"New splits: splits/{split_name}")
    logger.info(f"Training config: {config_path}")
    logger.info("=" * 60)
    logger.info("Ready to train with augmented data!")
    logger.info(f"   Run: python train.py --config {config_path}")

if __name__ == "__main__":
    main()
