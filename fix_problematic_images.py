#!/usr/bin/env python3
"""
Script to fix the problematic images that were processed with incorrect dimensions.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys
sys.path.append('scripts')

from preprocess import CanonicalPreprocessor, DullRazorHairRemoval, DatasetNormalizer

def find_problematic_images():
    """Find all images with incorrect dimensions."""
    proc_dir = Path('data/ISIC2018_proc/train_images')
    problematic = []
    
    print("Scanning for problematic images...")
    for img_path in tqdm(proc_dir.glob('*.png')):
        img = cv2.imread(str(img_path))
        if img is not None and img.shape != (384, 384, 3):
            problematic.append(img_path.stem)
    
    return problematic

def fix_problematic_images(problematic_list):
    """Reprocess the problematic images with the fixed pipeline."""
    
    # Initialize processors
    preprocessor = CanonicalPreprocessor(384)
    hair_remover = DullRazorHairRemoval()
    normalizer = DatasetNormalizer()
    
    # Load dataset statistics
    stats_path = Path("data/ISIC2018_proc/dataset_stats.json")
    if stats_path.exists():
        normalizer.load_stats(str(stats_path))
    else:
        print("Warning: No dataset stats found, using computed values")
        normalizer.mean = 0.6042
        normalizer.std = 0.1817
    
    input_dir = Path("data/ISIC2018/train_images")
    output_dir = Path("data/ISIC2018_proc/train_images")
    mask_input_dir = Path("data/ISIC2018/train_masks")
    mask_output_dir = Path("data/ISIC2018_proc/train_masks")
    
    print(f"Reprocessing {len(problematic_list)} problematic images...")
    
    for img_name in tqdm(problematic_list, desc="Fixing images"):
        # Process image
        img_path = input_dir / f"{img_name}.jpg"
        
        if img_path.exists():
            # Load and process image
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply hair removal
            image_processed = hair_remover.remove_hair(image_rgb)
            
            # Apply canonical preprocessing
            image_resized = preprocessor.resize_and_pad(image_processed)
            
            # Apply normalization
            image_normalized = normalizer.normalize_image(image_resized)
            
            # Convert back for saving
            if normalizer.mean is not None and normalizer.std is not None:
                image_save = (image_normalized * normalizer.std + normalizer.mean) * 255.0
            else:
                image_save = image_normalized * 255.0
            
            image_save = np.clip(image_save, 0, 255).astype(np.uint8)
            image_save_bgr = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
            
            # Save fixed image
            output_path = output_dir / f"{img_name}.png"
            cv2.imwrite(str(output_path), image_save_bgr)
            
            # Also fix corresponding mask
            mask_path = mask_input_dir / f"{img_name}_segmentation.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_resized = preprocessor.resize_and_pad(mask, is_mask=True)
                mask_output_path = mask_output_dir / f"{img_name}_segmentation.png"
                cv2.imwrite(str(mask_output_path), mask_resized)

def verify_fix():
    """Verify that all images now have correct dimensions."""
    proc_dir = Path('data/ISIC2018_proc/train_images')
    incorrect = 0
    total = 0
    
    print("Verifying fix...")
    for img_path in tqdm(proc_dir.glob('*.png')):
        img = cv2.imread(str(img_path))
        total += 1
        if img is not None and img.shape != (384, 384, 3):
            incorrect += 1
            print(f"Still incorrect: {img_path.name} - {img.shape}")
    
    print(f"Verification complete: {incorrect}/{total} images still have incorrect dimensions")
    return incorrect == 0

def main():
    print("🔧 Fixing problematic processed images")
    print("=" * 50)
    
    # Find problematic images
    problematic = find_problematic_images()
    print(f"Found {len(problematic)} problematic images")
    
    if len(problematic) == 0:
        print("No problematic images found!")
        return
    
    # Show some examples
    print("\nExamples of problematic images:")
    for i, name in enumerate(problematic[:10]):
        print(f"  {name}")
    if len(problematic) > 10:
        print(f"  ... and {len(problematic) - 10} more")
    
    # Ask for confirmation
    response = input(f"\nReprocess {len(problematic)} images? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Fix the images
    fix_problematic_images(problematic)
    
    # Verify the fix
    if verify_fix():
        print("\n✅ All images fixed successfully!")
    else:
        print("\n❌ Some images still have incorrect dimensions")

if __name__ == "__main__":
    main()
