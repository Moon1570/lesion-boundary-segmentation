#!/usr/bin/env python3
"""
Debug script to identify where the dimension bug occurs in preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('scripts')

from preprocess import CanonicalPreprocessor, DullRazorHairRemoval, DatasetNormalizer

def debug_image_processing(image_name="ISIC_0000507.jpg"):
    """Debug the preprocessing steps for a problematic image."""
    
    # Load original image
    img_path = Path(f"data/ISIC2018/train_images/{image_name}")
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    
    print(f"Debugging: {image_name}")
    print("=" * 40)
    
    # Step 1: Load original
    original = cv2.imread(str(img_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    print(f"1. Original loaded: {original_rgb.shape}")
    
    # Step 2: Hair removal
    hair_remover = DullRazorHairRemoval()
    hair_removed = hair_remover.remove_hair(original_rgb.copy())
    print(f"2. After hair removal: {hair_removed.shape}")
    
    # Step 3: Canonical preprocessing
    preprocessor = CanonicalPreprocessor(384)
    resized = preprocessor.resize_and_pad(hair_removed)
    print(f"3. After resize_and_pad: {resized.shape}")
    
    # Step 4: Normalization
    normalizer = DatasetNormalizer()
    # Load existing stats
    stats_path = Path("data/ISIC2018_proc/dataset_stats.json")
    if stats_path.exists():
        normalizer.load_stats(str(stats_path))
    else:
        normalizer.mean = 0.6042
        normalizer.std = 0.1817
    
    normalized = normalizer.normalize_image(resized)
    print(f"4. After normalization: {normalized.shape}")
    
    # Step 5: Convert back for saving
    if normalizer.mean is not None and normalizer.std is not None:
        image_save = (normalized * normalizer.std + normalizer.mean) * 255.0
    else:
        image_save = normalized * 255.0
    
    image_save = np.clip(image_save, 0, 255).astype(np.uint8)
    print(f"5. Before BGR conversion: {image_save.shape}")
    
    image_save_bgr = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
    print(f"6. After BGR conversion: {image_save_bgr.shape}")
    
    # Save debug image
    debug_path = f"debug_{image_name.replace('.jpg', '.png')}"
    cv2.imwrite(debug_path, image_save_bgr)
    print(f"7. Saved debug image: {debug_path}")
    
    # Verify saved image
    saved_img = cv2.imread(debug_path)
    print(f"8. Loaded saved image: {saved_img.shape if saved_img is not None else None}")

if __name__ == "__main__":
    debug_image_processing("ISIC_0000507.jpg")
    print("\n" + "="*40)
    debug_image_processing("ISIC_0000000.jpg")  # Compare with working image
