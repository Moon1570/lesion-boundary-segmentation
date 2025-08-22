#!/usr/bin/env python3
"""
Preprocessing script for ISIC2018 lesion boundary segmentation dataset.

This script implements deterministic preprocessing including:
- Canonical resizing: shorter side → 384px, center-pad to 384×384
- Normalization: per-image min-max to [0,1], then dataset standardization
- Optional DullRazor hair removal using morphological operations

Usage:
    python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc
    python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc --hair-removal dullrazor
"""

import argparse
import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class CanonicalPreprocessor:
    """Handles canonical resizing and padding operations."""
    
    def __init__(self, target_size: int = 384):
        self.target_size = target_size
    
    def resize_and_pad(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Resize image/mask: shorter side → target_size, then center-pad to square.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            is_mask: If True, uses nearest-neighbor interpolation
            
        Returns:
            Processed image/mask as numpy array (target_size, target_size, C) or (target_size, target_size)
        """
        h, w = image.shape[:2]
        
        # Calculate resize factor based on shorter side
        scale = self.target_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize using appropriate interpolation
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Handle case where resized dimensions might be larger than target due to scaling
        if new_h > self.target_size or new_w > self.target_size:
            # Center crop if any dimension exceeds target
            start_h = max(0, (new_h - self.target_size) // 2)
            start_w = max(0, (new_w - self.target_size) // 2)
            end_h = min(new_h, start_h + self.target_size)
            end_w = min(new_w, start_w + self.target_size)
            
            if len(image.shape) == 3:
                cropped = resized[start_h:end_h, start_w:end_w, :]
                # If cropped is smaller than target, pad it
                if cropped.shape[0] < self.target_size or cropped.shape[1] < self.target_size:
                    padded = np.zeros((self.target_size, self.target_size, image.shape[2]), dtype=image.dtype)
                    pad_h = (self.target_size - cropped.shape[0]) // 2
                    pad_w = (self.target_size - cropped.shape[1]) // 2
                    padded[pad_h:pad_h + cropped.shape[0], pad_w:pad_w + cropped.shape[1], :] = cropped
                    return padded
                else:
                    return cropped
            else:
                cropped = resized[start_h:end_h, start_w:end_w]
                # If cropped is smaller than target, pad it
                if cropped.shape[0] < self.target_size or cropped.shape[1] < self.target_size:
                    padded = np.zeros((self.target_size, self.target_size), dtype=image.dtype)
                    pad_h = (self.target_size - cropped.shape[0]) // 2
                    pad_w = (self.target_size - cropped.shape[1]) // 2
                    padded[pad_h:pad_h + cropped.shape[0], pad_w:pad_w + cropped.shape[1]] = cropped
                    return padded
                else:
                    return cropped
        
        # Center padding for smaller images
        pad_h = (self.target_size - new_h) // 2
        pad_w = (self.target_size - new_w) // 2
        
        if len(image.shape) == 3:
            padded = np.zeros((self.target_size, self.target_size, image.shape[2]), dtype=image.dtype)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
        else:
            padded = np.zeros((self.target_size, self.target_size), dtype=image.dtype)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
            
        return padded


class DullRazorHairRemoval:
    """Implements DullRazor hair removal using morphological operations and inpainting."""
    
    def __init__(self):
        # Create morphological kernels for hair detection
        self.kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        self.kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    def detect_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect hair in dermoscopic images using morphological operations.
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            Binary mask indicating hair locations
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological operations for hair detection
        # Black top-hat to detect dark structures (hair)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel_1)
        
        # Threshold to create binary mask
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Morphological closing to connect hair segments
        hair_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel_2)
        
        # Remove small artifacts
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, self.kernel_3)
        
        return hair_mask
    
    def remove_hair(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair from dermoscopic image using DullRazor method.
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            Image with hair removed
        """
        # Detect hair mask
        hair_mask = self.detect_hair_mask(image)
        
        # Apply bilinear inpainting to remove hair
        result = cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)
        
        return result


class DatasetNormalizer:
    """Handles dataset-level normalization statistics computation and application."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def compute_dataset_stats(self, image_paths: list, preprocessor: CanonicalPreprocessor) -> Dict[str, float]:
        """
        Compute dataset-level mean and std statistics from training images.
        
        Args:
            image_paths: List of paths to training images
            preprocessor: Preprocessor for canonical resizing
            
        Returns:
            Dictionary containing mean and std values
        """
        print("Computing dataset statistics...")
        
        pixel_values = []
        
        for img_path in tqdm(image_paths, desc="Computing stats"):
            # Load and preprocess image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocessor.resize_and_pad(image)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Collect pixel values
            pixel_values.append(image.flatten())
        
        # Concatenate all pixel values
        all_pixels = np.concatenate(pixel_values)
        
        # Compute statistics
        self.mean = float(np.mean(all_pixels))
        self.std = float(np.std(all_pixels))
        
        return {"mean": self.mean, "std": self.std}
    
    def load_stats(self, stats_path: str):
        """Load pre-computed statistics."""
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std = stats["std"]
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply normalization: min-max to [0,1], then standardize with dataset mean/std.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image
        """
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply dataset standardization if stats are available
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        
        return image


def process_split(input_dir: Path, output_dir: Path, split: str, 
                 preprocessor: CanonicalPreprocessor, normalizer: DatasetNormalizer,
                 hair_remover: Optional[DullRazorHairRemoval] = None):
    """
    Process a single data split (train/val/test).
    
    Args:
        input_dir: Input directory containing the split
        output_dir: Output directory for processed data
        split: Split name ('train', 'val', or 'test')
        preprocessor: Canonical preprocessor
        normalizer: Dataset normalizer
        hair_remover: Optional hair removal processor
    """
    split_input_dir = input_dir / f"{split}_images"
    split_output_dir = output_dir / f"{split}_images"
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process masks for train split
    if split == "train":
        mask_input_dir = input_dir / "train_masks"
        mask_output_dir = output_dir / "train_masks"
        mask_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(split_input_dir.glob("*.jpg")) + list(split_input_dir.glob("*.png"))
    
    print(f"Processing {split} split: {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc=f"Processing {split}"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply hair removal if specified
        if hair_remover is not None:
            image = hair_remover.remove_hair(image)
        
        # Apply canonical preprocessing
        image = preprocessor.resize_and_pad(image)
        
        # Apply normalization
        image = normalizer.normalize_image(image)
        
        # Convert back to uint8 for saving (denormalize)
        if normalizer.mean is not None and normalizer.std is not None:
            # Denormalize for saving
            image_save = (image * normalizer.std + normalizer.mean) * 255.0
        else:
            image_save = image * 255.0
        
        image_save = np.clip(image_save, 0, 255).astype(np.uint8)
        image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
        
        # Save image as PNG (lossless)
        output_path = split_output_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(output_path), image_save)
        
        # Process corresponding mask for training data
        if split == "train":
            mask_path = mask_input_dir / f"{img_path.stem}_segmentation.png"
            if mask_path.exists():
                # Load mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # Apply canonical preprocessing with nearest-neighbor
                mask = preprocessor.resize_and_pad(mask, is_mask=True)
                
                # Save mask
                mask_output_path = mask_output_dir / f"{img_path.stem}_segmentation.png"
                cv2.imwrite(str(mask_output_path), mask)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ISIC2018 dataset")
    parser.add_argument("--input_dir", type=str, default="data/ISIC2018",
                       help="Input directory containing raw ISIC2018 data")
    parser.add_argument("--output_dir", type=str, default="data/ISIC2018_proc",
                       help="Output directory for processed data")
    parser.add_argument("--target_size", type=int, default=384,
                       help="Target image size (default: 384)")
    parser.add_argument("--hair-removal", type=str, choices=["dullrazor"], default=None,
                       help="Apply hair removal method")
    parser.add_argument("--recompute_stats", action="store_true",
                       help="Recompute dataset statistics even if they exist")
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    preprocessor = CanonicalPreprocessor(args.target_size)
    normalizer = DatasetNormalizer()
    
    # Initialize hair remover if specified
    hair_remover = None
    if args.hair_removal == "dullrazor":
        print("Initializing DullRazor hair removal...")
        hair_remover = DullRazorHairRemoval()
    
    # Compute or load dataset statistics
    stats_path = output_dir / "dataset_stats.json"
    
    if not stats_path.exists() or args.recompute_stats:
        # Get training image paths
        train_images = list((input_dir / "train_images").glob("*.jpg"))
        
        if not train_images:
            raise FileNotFoundError(f"No training images found in {input_dir / 'train_images'}")
        
        # Compute dataset statistics
        stats = normalizer.compute_dataset_stats(train_images, preprocessor)
        
        # Save statistics
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics computed: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        # Load existing statistics
        normalizer.load_stats(stats_path)
        print(f"Loaded dataset statistics: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    
    # Process each split
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = input_dir / f"{split}_images"
        if split_dir.exists():
            process_split(input_dir, output_dir, split, preprocessor, normalizer, hair_remover)
        else:
            print(f"Warning: {split} split not found, skipping...")
    
    print(f"\nPreprocessing completed! Processed data saved to: {output_dir}")
    print(f"Dataset statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
