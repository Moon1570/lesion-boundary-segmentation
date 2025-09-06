#!/usr/bin/env python3
"""
Robust Advanced Preprocessing Pipeline for ISIC2018 Dataset

This module implements edge-enhanced preprocessing with improved error handling
and compatibility across different image formats and OpenCV versions.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from skimage import filters, restoration, exposure
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


class RobustAdvancedPreprocessor:
    """
    Robust advanced preprocessing pipeline with enhanced edge clarity.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (384, 384),
                 edge_enhancement_strength: float = 1.5,
                 contrast_enhancement: float = 1.8):
        """
        Initialize the robust preprocessor.
        
        Args:
            target_size: Target image dimensions
            edge_enhancement_strength: Strength of edge enhancement (0.5-3.0)
            contrast_enhancement: Contrast enhancement factor (1.0-3.0)
        """
        self.target_size = target_size
        self.edge_strength = np.clip(edge_enhancement_strength, 0.5, 3.0)
        self.contrast_factor = np.clip(contrast_enhancement, 1.0, 3.0)
        
    def remove_hair_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts using morphological operations.
        
        Args:
            image: Input RGB image
            
        Returns:
            Image with hair artifacts removed
        """
        try:
            # Convert to grayscale for hair detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Create kernels for detecting hair-like structures
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 1))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 17))
            
            # Apply morphological operations to detect hair
            hair_mask1 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel1)
            hair_mask2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel2)
            hair_mask = cv2.add(hair_mask1, hair_mask2)
            
            # Threshold to create binary mask
            _, hair_mask = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)
            
            # Refine mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply inpainting to remove detected hair
            result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
            return result
        except:
            return image
    
    def enhance_edges_robust(self, image: np.ndarray) -> np.ndarray:
        """
        Apply robust edge enhancement using safe operations.
        
        Args:
            image: Input RGB image
            
        Returns:
            Edge-enhanced image
        """
        try:
            # Convert to LAB color space for better edge preservation
            if image.max() > 1:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.astype(np.float32)
                
            lab = rgb2lab(image_float)
            L_channel = lab[:, :, 0]
            
            # Apply Gaussian filtering for unsharp masking
            blurred = ndimage.gaussian_filter(L_channel, sigma=1.0)
            unsharp = L_channel + self.edge_strength * 0.3 * (L_channel - blurred)
            
            # Apply Sobel edge detection
            grad_x = ndimage.sobel(L_channel, axis=1)
            grad_y = ndimage.sobel(L_channel, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Combine edge enhancements
            enhanced_L = unsharp + self.edge_strength * 0.2 * np.clip(gradient_magnitude, 0, 5)
            enhanced_L = np.clip(enhanced_L, 0, 100)
            
            # Update LAB image
            lab[:, :, 0] = enhanced_L
            
            # Convert back to RGB
            result = lab2rgb(lab)
            
            if image.max() > 1:
                return (np.clip(result, 0, 1) * 255).astype(np.uint8)
            else:
                return np.clip(result, 0, 1).astype(np.float32)
                
        except Exception as e:
            print(f"Edge enhancement failed: {e}, returning original")
            return image
    
    def enhance_contrast_robust(self, image: np.ndarray) -> np.ndarray:
        """
        Apply robust contrast enhancement.
        
        Args:
            image: Input RGB image
            
        Returns:
            Contrast-enhanced image
        """
        try:
            # Ensure proper format
            if image.max() > 1:
                image_float = image.astype(np.float32) / 255.0
                is_uint8 = True
            else:
                image_float = image.astype(np.float32)
                is_uint8 = False
            
            # Convert to LAB for better control
            lab = rgb2lab(image_float)
            L_channel = lab[:, :, 0] / 100.0  # Normalize to 0-1
            
            # Apply CLAHE using skimage
            clahe_enhanced = exposure.equalize_adapthist(L_channel, clip_limit=0.02)
            
            # Apply gamma correction
            mean_intensity = np.mean(L_channel)
            gamma = 0.7 + 0.6 * (1 - mean_intensity)  # Adaptive gamma
            gamma_enhanced = np.power(L_channel, gamma)
            
            # Combine enhancements
            final_L = 0.6 * clahe_enhanced + 0.4 * gamma_enhanced
            
            # Update LAB image
            lab[:, :, 0] = final_L * 100.0
            
            # Convert back to RGB
            result = lab2rgb(lab)
            
            if is_uint8:
                return (np.clip(result, 0, 1) * 255).astype(np.uint8)
            else:
                return np.clip(result, 0, 1).astype(np.float32)
                
        except Exception as e:
            print(f"Contrast enhancement failed: {e}, returning original")
            return image
    
    def denoise_robust(self, image: np.ndarray) -> np.ndarray:
        """
        Apply robust denoising.
        
        Args:
            image: Input RGB image
            
        Returns:
            Denoised image
        """
        try:
            # Convert to uint8 if needed
            if image.max() <= 1:
                image_uint8 = (image * 255).astype(np.uint8)
                was_float = True
            else:
                image_uint8 = image.astype(np.uint8)
                was_float = False
            
            # Apply bilateral filtering
            denoised = cv2.bilateralFilter(image_uint8, 9, 75, 75)
            
            if was_float:
                return denoised.astype(np.float32) / 255.0
            else:
                return denoised
                
        except Exception as e:
            print(f"Denoising failed: {e}, returning original")
            return image
    
    def correct_illumination_robust(self, image: np.ndarray) -> np.ndarray:
        """
        Apply robust illumination correction.
        
        Args:
            image: Input RGB image
            
        Returns:
            Illumination-corrected image
        """
        try:
            # Ensure proper format
            if image.max() > 1:
                image_float = image.astype(np.float32) / 255.0
                is_uint8 = True
            else:
                image_float = image.astype(np.float32)
                is_uint8 = False
            
            # Convert to LAB
            lab = rgb2lab(image_float)
            L_channel = lab[:, :, 0]
            
            # Estimate background using morphological opening
            kernel_size = max(10, min(L_channel.shape) // 20)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(L_channel.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            background = ndimage.gaussian_filter(background.astype(float), sigma=10)
            
            # Correct illumination
            corrected_L = L_channel - background + np.mean(background)
            corrected_L = np.clip(corrected_L, 0, 100)
            
            # Update LAB image
            lab[:, :, 0] = corrected_L
            
            # Convert back to RGB
            result = lab2rgb(lab)
            
            if is_uint8:
                return (np.clip(result, 0, 1) * 255).astype(np.uint8)
            else:
                return np.clip(result, 0, 1).astype(np.float32)
                
        except Exception as e:
            print(f"Illumination correction failed: {e}, returning original")
            return image
    
    def process_image_robust(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the complete robust preprocessing pipeline.
        
        Args:
            image: Input RGB image (0-255 range)
            
        Returns:
            Fully processed image (0-255 range)
        """
        # Ensure input is in correct format
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # Step 1: Hair artifact removal
        image = self.remove_hair_artifacts(image)
        
        # Step 2: Illumination correction
        image = self.correct_illumination_robust(image)
        
        # Step 3: Denoising
        image = self.denoise_robust(image)
        
        # Step 4: Edge enhancement
        image = self.enhance_edges_robust(image)
        
        # Step 5: Contrast enhancement
        image = self.enhance_contrast_robust(image)
        
        # Final normalization
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        return image.astype(np.uint8)
    
    def resize_image_robust(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resize image and mask robustly.
        
        Args:
            image: Input image
            mask: Optional input mask
            
        Returns:
            Resized image and mask
        """
        try:
            # Direct resize to target size
            resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            result_mask = None
            if mask is not None:
                result_mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            return resized_image, result_mask
            
        except Exception as e:
            print(f"Resize failed: {e}")
            return image, mask


def process_dataset_robust(input_dir: str, output_dir: str, subset_size: Optional[int] = None):
    """
    Process dataset with robust error handling.
    
    Args:
        input_dir: Input directory containing processed ISIC2018 data
        output_dir: Output directory for advanced processed data
        subset_size: Optional size for creating a subset
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = RobustAdvancedPreprocessor(
        target_size=(384, 384),
        edge_enhancement_strength=1.8,
        contrast_enhancement=1.6
    )
    
    processed_count = 0
    
    # Process each subdirectory
    for subdir in ['train_images', 'val_images', 'test_images', 'train_masks']:
        input_subdir = input_path / subdir
        output_subdir = output_path / subdir
        
        if not input_subdir.exists():
            print(f"Skipping {subdir} - directory not found")
            continue
            
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Get list of files
        if subdir.endswith('_masks'):
            file_pattern = "*.png"
            is_mask = True
        else:
            file_pattern = "*.png"
            is_mask = False
        
        files = list(input_subdir.glob(file_pattern))
        
        # Create subset if specified
        if subset_size and not is_mask:
            files = files[:subset_size]
            print(f"Creating subset of {len(files)} files for {subdir}")
        elif subset_size and is_mask:
            # For masks, match the subset size but only process existing masks
            mask_names = set()
            train_dir = input_path / "train_images"
            if train_dir.exists():
                train_files = list(train_dir.glob("*.png"))[:subset_size]
                for train_file in train_files:
                    base_name = train_file.stem
                    mask_name = f"{base_name}_segmentation.png"
                    mask_names.add(mask_name)
            
            # Filter masks to match subset
            files = [f for f in files if f.name in mask_names]
            print(f"Processing {len(files)} masks matching subset for {subdir}")
        
        print(f"Processing {len(files)} files in {subdir}...")
        
        # Process files with progress bar
        success_count = 0
        for file_path in tqdm(files, desc=f"Processing {subdir}"):
            try:
                if is_mask:
                    # Simple processing for masks (resize only)
                    mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    
                    # Resize mask
                    resized_mask, _ = preprocessor.resize_image_robust(mask)
                    
                    # Save processed mask
                    output_file = output_subdir / file_path.name
                    cv2.imwrite(str(output_file), resized_mask)
                    success_count += 1
                    
                else:
                    # Advanced processing for images
                    image = cv2.imread(str(file_path))
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply robust preprocessing
                    processed_image = preprocessor.process_image_robust(image)
                    
                    # Resize image
                    resized_image, _ = preprocessor.resize_image_robust(processed_image)
                    
                    # Convert RGB back to BGR for saving
                    output_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                    
                    # Save processed image
                    output_file = output_subdir / file_path.name
                    cv2.imwrite(str(output_file), output_image)
                    success_count += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Successfully processed {success_count}/{len(files)} files in {subdir}")
        processed_count += success_count
    
    # Calculate and save new statistics
    print("Calculating new dataset statistics...")
    
    train_dir = output_path / "train_images"
    if train_dir.exists():
        train_files = list(train_dir.glob("*.png"))[:50]  # Sample for statistics
        
        if train_files:
            pixel_values = []
            for file_path in tqdm(train_files, desc="Calculating statistics"):
                try:
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        # Convert to RGB and normalize
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pixel_values.extend(image.flatten())
                except:
                    continue
            
            if pixel_values:
                pixel_values = np.array(pixel_values, dtype=np.float32) / 255.0
                new_stats = {
                    "mean": float(np.mean(pixel_values)),
                    "std": float(np.std(pixel_values))
                }
                
                # Save new statistics
                with open(output_path / "dataset_stats.json", 'w') as f:
                    json.dump(new_stats, f, indent=2)
                
                print(f"New dataset statistics: mean={new_stats['mean']:.4f}, std={new_stats['std']:.4f}")
    
    print(f"✅ Robust preprocessing completed! Processed {processed_count} files total.")


def main():
    """Main function for robust preprocessing."""
    parser = argparse.ArgumentParser(description="Robust advanced preprocessing for ISIC2018 dataset")
    parser.add_argument("--input-dir", type=str, default="data/ISIC2018_proc",
                      help="Input directory with processed ISIC2018 data")
    parser.add_argument("--output-dir", type=str, default="data/ISIC2018_advanced",
                      help="Output directory for advanced processed data")
    parser.add_argument("--subset-size", type=int, default=None,
                      help="Create subset with specified number of images")
    
    args = parser.parse_args()
    
    print("🔬 Robust Advanced Preprocessing Pipeline for ISIC2018 Dataset")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    if args.subset_size:
        print(f"Creating subset with {args.subset_size} images per split")
    
    print("=" * 60)
    
    # Process dataset
    process_dataset_robust(args.input_dir, args.output_dir, args.subset_size)
    
    print("\n✅ Robust advanced preprocessing completed successfully!")
    print(f"📁 Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
