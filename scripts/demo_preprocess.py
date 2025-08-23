#!/usr/bin/env python3
"""
Demo script to test preprocessing functionality on a few sample images.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from preprocess import CanonicalPreprocessor, DullRazorHairRemoval, DatasetNormalizer


def demo_preprocessing():
    """Demonstrate preprocessing pipeline on sample images."""
    
    # Setup
    data_dir = Path("data/ISIC2018")
    train_images_dir = data_dir / "train_images"
    train_masks_dir = data_dir / "train_masks"
    
    # Get first few images for demo
    image_files = list(train_images_dir.glob("*.jpg"))[:3]
    
    if not image_files:
        print("No training images found!")
        return
    
    # Initialize processors
    preprocessor = CanonicalPreprocessor(384)
    hair_remover = DullRazorHairRemoval()
    
    print(f"Demo preprocessing on {len(image_files)} images...")
    
    # Create demo output directory in runs/figs
    demo_dir = Path("runs/figs/preprocessing_demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {img_path.name}...")
        
        # Load original image
        original = cv2.imread(str(img_path))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Apply hair removal
        hair_removed = hair_remover.remove_hair(original_rgb.copy())
        
        # Apply canonical preprocessing
        resized_original = preprocessor.resize_and_pad(original_rgb)
        resized_hair_removed = preprocessor.resize_and_pad(hair_removed)
        
        # Load and process corresponding mask
        mask_path = train_masks_dir / f"{img_path.stem}_segmentation.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            resized_mask = preprocessor.resize_and_pad(mask, is_mask=True)
        else:
            resized_mask = None
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3 if resized_mask is not None else 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title(f"Original\n{original_rgb.shape}")
        axes[0, 0].axis('off')
        
        # Hair removed
        axes[0, 1].imshow(hair_removed)
        axes[0, 1].set_title("Hair Removed (DullRazor)")
        axes[0, 1].axis('off')
        
        # Mask if available
        if resized_mask is not None:
            axes[0, 2].imshow(mask, cmap='gray')
            axes[0, 2].set_title(f"Original Mask\n{mask.shape}")
            axes[0, 2].axis('off')
        
        # Resized original
        axes[1, 0].imshow(resized_original)
        axes[1, 0].set_title(f"Resized Original\n{resized_original.shape}")
        axes[1, 0].axis('off')
        
        # Resized hair removed
        axes[1, 1].imshow(resized_hair_removed)
        axes[1, 1].set_title(f"Resized Hair Removed\n{resized_hair_removed.shape}")
        axes[1, 1].axis('off')
        
        # Resized mask if available
        if resized_mask is not None:
            axes[1, 2].imshow(resized_mask, cmap='gray')
            axes[1, 2].set_title(f"Resized Mask\n{resized_mask.shape}")
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(demo_dir / f"demo_{img_path.stem}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Original shape: {original_rgb.shape}")
        print(f"  Processed shape: {resized_original.shape}")
        if resized_mask is not None:
            print(f"  Mask shape: {resized_mask.shape}")
    
    print(f"\nDemo completed! Check {demo_dir} for results.")


if __name__ == "__main__":
    demo_preprocessing()
