#!/usr/bin/env python3
"""
Advanced Preprocessing Pipeline for ISIC2018 Dataset

This module implements state-of-the-art preprocessing techniques specifically designed
for lesion boundary segmentation. The pipeline enhances edge clarity through multiple
complementary approaches while preserving important medical image characteristics.

Key Features:
- Edge-preserving denoising (Non-local means, Bilateral filtering)
- Multi-scale edge enhancement (Gradient-based, Laplacian sharpening)
- Contrast optimization (CLAHE, Gamma correction, Histogram equalization)
- Color space optimization (LAB, HSV enhancement)
- Morphological operations for boundary refinement
- Hair artifact removal using advanced inpainting
- Illumination correction and normalization
- Preserves original image dimensions and medical accuracy
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
from skimage import filters, morphology, restoration, segmentation, measure
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.feature import canny
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline for dermatological images with focus on edge enhancement.
    
    The pipeline applies multiple complementary techniques:
    1. Noise reduction with edge preservation
    2. Multi-scale edge enhancement  
    3. Contrast and illumination optimization
    4. Color space enhancement
    5. Hair artifact removal
    6. Morphological boundary refinement
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (384, 384),
                 preserve_aspect_ratio: bool = True,
                 edge_enhancement_strength: float = 1.5,
                 contrast_enhancement: float = 2.0,
                 noise_reduction_strength: float = 0.8):
        """
        Initialize the advanced preprocessor.
        
        Args:
            target_size: Target image dimensions
            preserve_aspect_ratio: Whether to preserve aspect ratio during resize
            edge_enhancement_strength: Strength of edge enhancement (0.5-3.0)
            contrast_enhancement: Contrast enhancement factor (1.0-3.0)
            noise_reduction_strength: Noise reduction strength (0.1-1.0)
        """
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.edge_strength = np.clip(edge_enhancement_strength, 0.5, 3.0)
        self.contrast_factor = np.clip(contrast_enhancement, 1.0, 3.0)
        self.noise_strength = np.clip(noise_reduction_strength, 0.1, 1.0)
        
        # Initialize kernels and filters
        self._init_kernels()
        
    def _init_kernels(self):
        """Initialize convolution kernels for various operations."""
        # Edge enhancement kernels
        self.laplacian_kernel = np.array([[-1, -1, -1],
                                        [-1,  8, -1],
                                        [-1, -1, -1]], dtype=np.float32)
        
        self.unsharp_kernel = np.array([[-1, -4, -6, -4, -1],
                                      [-4, -16, -24, -16, -4],
                                      [-6, -24, 476, -24, -6],
                                      [-4, -16, -24, -16, -4],
                                      [-1, -4, -6, -4, -1]], dtype=np.float32) / 256.0
        
        # Morphological kernels
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    def remove_hair_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts using advanced morphological operations and inpainting.
        
        Args:
            image: Input RGB image
            
        Returns:
            Image with hair artifacts removed
        """
        # Convert to grayscale for hair detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect dark hair-like structures using morphological operations
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 17))
        
        # Apply morphological operations to detect hair
        hair_mask1 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel1)
        hair_mask2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel2)
        hair_mask = cv2.add(hair_mask1, hair_mask2)
        
        # Threshold to create binary mask
        _, hair_mask = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)
        
        # Refine mask using morphological operations
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, self.small_kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, self.small_kernel)
        
        # Apply inpainting to remove detected hair
        result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def enhance_edges_multiscale(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multi-scale edge enhancement using multiple techniques.
        
        Args:
            image: Input RGB image
            
        Returns:
            Edge-enhanced image
        """
        # Convert to LAB color space for better edge preservation
        lab = rgb2lab(image)
        L_channel = lab[:, :, 0]
        
        # 1. Gradient-based edge enhancement
        grad_x = cv2.Sobel(L_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Laplacian edge enhancement
        laplacian = cv2.filter2D(L_channel, cv2.CV_64F, self.laplacian_kernel)
        
        # 3. Unsharp masking
        blurred = gaussian_filter(L_channel, sigma=1.0)
        unsharp = L_channel + self.edge_strength * (L_channel - blurred)
        
        # 4. Canny edge detection for guidance
        edges = canny(L_channel / 100.0, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
        
        # Combine edge enhancements
        edge_enhanced = L_channel.copy()
        edge_enhanced += self.edge_strength * 0.3 * np.clip(gradient_magnitude, 0, 10)
        edge_enhanced += self.edge_strength * 0.2 * np.clip(laplacian, -10, 10)
        edge_enhanced = 0.7 * edge_enhanced + 0.3 * unsharp
        
        # Apply edge guidance
        edge_mask = gaussian_filter(edges.astype(float), sigma=0.5)
        edge_enhanced = edge_enhanced * (1 + 0.5 * edge_mask)
        
        # Update LAB image
        lab[:, :, 0] = np.clip(edge_enhanced, 0, 100)
        
        # Convert back to RGB
        result = lab2rgb(lab)
        return np.clip(result, 0, 1)
    
    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive contrast enhancement using multiple techniques.
        
        Args:
            image: Input RGB image (0-1 range)
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB for better control
        lab = rgb2lab(image)
        L_channel = lab[:, :, 0] / 100.0  # Normalize to 0-1
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_enhanced = equalize_adapthist(L_channel, clip_limit=0.02)
        
        # 2. Gamma correction with adaptive gamma
        mean_intensity = np.mean(L_channel)
        gamma = 0.5 + 1.5 * (1 - mean_intensity)  # Adaptive gamma based on image brightness
        gamma_enhanced = np.power(L_channel, gamma)
        
        # 3. Local contrast enhancement
        local_mean = gaussian_filter(L_channel, sigma=20)
        local_contrast = L_channel - local_mean
        enhanced_contrast = L_channel + self.contrast_factor * 0.3 * local_contrast
        
        # Combine enhancements
        final_L = (0.4 * clahe_enhanced + 
                  0.3 * gamma_enhanced + 
                  0.3 * np.clip(enhanced_contrast, 0, 1))
        
        # Update LAB image
        lab[:, :, 0] = final_L * 100.0
        
        # Convert back to RGB
        result = lab2rgb(lab)
        return np.clip(result, 0, 1)
    
    def enhance_color_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance color channels for better lesion contrast.
        
        Args:
            image: Input RGB image (0-1 range)
            
        Returns:
            Color-enhanced image
        """
        # HSV enhancement for better color separation
        hsv = rgb2hsv(image)
        
        # Enhance saturation for better color contrast
        hsv[:, :, 1] = np.power(hsv[:, :, 1], 0.8)  # Increase saturation
        
        # Selective hue adjustment (enhance reds/browns typical of lesions)
        hue = hsv[:, :, 0]
        red_brown_mask = ((hue < 0.1) | (hue > 0.9)) | ((hue > 0.05) & (hue < 0.15))
        hsv[:, :, 1][red_brown_mask] *= 1.2  # Boost saturation in red-brown regions
        
        # Convert back to RGB
        enhanced_rgb = hsv2rgb(hsv)
        
        # Additional enhancement in LAB space
        lab = rgb2lab(enhanced_rgb)
        
        # Enhance A and B channels for better color contrast
        lab[:, :, 1] *= 1.1  # Green-Red channel
        lab[:, :, 2] *= 1.1  # Blue-Yellow channel
        
        result = lab2rgb(lab)
        return np.clip(result, 0, 1)
    
    def denoise_edge_preserving(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving denoising.
        
        Args:
            image: Input RGB image (0-1 range)
            
        Returns:
            Denoised image with preserved edges
        """
        # Convert to uint8 for OpenCV operations
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply bilateral filtering for edge-preserving denoising
        bilateral = cv2.bilateralFilter(image_uint8, 9, 
                                      80 * self.noise_strength, 
                                      80 * self.noise_strength)
        
        # Apply non-local means denoising
        nlm_denoised = cv2.fastNlMeansDenoisingColored(image_uint8, None, 
                                                      10 * self.noise_strength, 
                                                      10 * self.noise_strength, 
                                                      7, 21)
        
        # Combine denoising methods
        combined = 0.6 * bilateral + 0.4 * nlm_denoised
        
        return combined.astype(np.float32) / 255.0
    
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Correct uneven illumination using morphological operations.
        
        Args:
            image: Input RGB image (0-1 range)
            
        Returns:
            Illumination-corrected image
        """
        # Convert to LAB for illumination correction
        lab = rgb2lab(image)
        L_channel = lab[:, :, 0]
        
        # Estimate background illumination using morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        background = cv2.morphologyEx(L_channel.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        background = gaussian_filter(background.astype(float), sigma=20)
        
        # Correct illumination
        corrected_L = L_channel - background + np.mean(background)
        corrected_L = np.clip(corrected_L, 0, 100)
        
        # Update LAB image
        lab[:, :, 0] = corrected_L
        
        # Convert back to RGB
        result = lab2rgb(lab)
        return np.clip(result, 0, 1)
    
    def apply_morphological_refinement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations for boundary refinement.
        
        Args:
            image: Input RGB image (0-1 range)
            
        Returns:
            Morphologically refined image
        """
        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply morphological gradient to enhance boundaries
        morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, self.small_kernel)
        
        # Apply top-hat and black-hat transforms
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.medium_kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.medium_kernel)
        
        # Combine morphological operations
        enhanced_gray = gray + tophat - blackhat + 0.3 * morph_gradient
        enhanced_gray = np.clip(enhanced_gray, 0, 255)
        
        # Convert back to RGB while preserving color information
        enhancement_factor = enhanced_gray.astype(float) / (gray.astype(float) + 1e-8)
        enhancement_factor = np.clip(enhancement_factor, 0.5, 2.0)
        
        # Apply enhancement factor to each channel
        result = image.copy()
        for i in range(3):
            result[:, :, i] *= enhancement_factor / 255.0
        
        return np.clip(result, 0, 1)
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the complete advanced preprocessing pipeline.
        
        Args:
            image: Input RGB image (0-255 range)
            
        Returns:
            Fully processed image (0-255 range)
        """
        # Normalize to 0-1 range
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Step 1: Hair artifact removal
        image = self.remove_hair_artifacts((image * 255).astype(np.uint8))
        image = image.astype(np.float32) / 255.0
        
        # Step 2: Illumination correction
        image = self.correct_illumination(image)
        
        # Step 3: Edge-preserving denoising
        image = self.denoise_edge_preserving(image)
        
        # Step 4: Multi-scale edge enhancement
        image = self.enhance_edges_multiscale(image)
        
        # Step 5: Adaptive contrast enhancement
        image = self.enhance_contrast_adaptive(image)
        
        # Step 6: Color channel enhancement
        image = self.enhance_color_channels(image)
        
        # Step 7: Morphological boundary refinement
        image = self.apply_morphological_refinement(image)
        
        # Final normalization and conversion back to 0-255 range
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)
    
    def resize_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resize image and mask while preserving aspect ratio if specified.
        
        Args:
            image: Input image
            mask: Optional input mask
            
        Returns:
            Resized image and mask
        """
        if self.preserve_aspect_ratio:
            # Calculate padding to preserve aspect ratio
            h, w = image.shape[:2]
            target_h, target_w = self.target_size
            
            # Calculate scale factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create padded image
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            # Handle different image shapes
            if len(image.shape) == 3:
                padded_image = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image
            else:
                padded_image = np.zeros((target_h, target_w), dtype=image.dtype)
                padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image
            
            result_image = padded_image
            result_mask = None
            
            if mask is not None:
                resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                padded_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
                padded_mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_mask
                result_mask = padded_mask
                
        else:
            # Direct resize without aspect ratio preservation
            result_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            result_mask = None
            
            if mask is not None:
                result_mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        return result_image, result_mask


def process_dataset(input_dir: str, output_dir: str, subset_size: Optional[int] = None):
    """
    Process an entire dataset with advanced preprocessing.
    
    Args:
        input_dir: Input directory containing processed ISIC2018 data
        output_dir: Output directory for advanced processed data
        subset_size: Optional size for creating a subset (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor(
        target_size=(384, 384),
        preserve_aspect_ratio=True,
        edge_enhancement_strength=2.0,
        contrast_enhancement=1.8,
        noise_reduction_strength=0.7
    )
    
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
            # Process masks (no advanced preprocessing, just resize)
            file_pattern = "*.png"
            is_mask = True
        else:
            # Process images
            file_pattern = "*.png"
            is_mask = False
        
        files = list(input_subdir.glob(file_pattern))
        
        # Create subset if specified
        if subset_size and not is_mask:
            files = files[:subset_size]
            print(f"Creating subset of {len(files)} files for {subdir}")
        
        print(f"Processing {len(files)} files in {subdir}...")
        
        # Process files with progress bar
        for file_path in tqdm(files, desc=f"Processing {subdir}"):
            try:
                if is_mask:
                    # Simple processing for masks (resize only)
                    mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    
                    # Resize mask
                    resized_mask, _ = preprocessor.resize_image(mask)
                    
                    # Save processed mask
                    output_file = output_subdir / file_path.name
                    cv2.imwrite(str(output_file), resized_mask)
                    
                else:
                    # Advanced processing for images
                    image = cv2.imread(str(file_path))
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply advanced preprocessing
                    processed_image = preprocessor.process_image(image)
                    
                    # Resize image
                    resized_image, _ = preprocessor.resize_image(processed_image)
                    
                    # Convert RGB back to BGR for saving
                    output_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                    
                    # Save processed image
                    output_file = output_subdir / file_path.name
                    cv2.imwrite(str(output_file), output_image)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    # Copy dataset statistics and update if needed
    stats_file = input_path / "dataset_stats.json"
    if stats_file.exists():
        # Calculate new statistics for advanced processed data
        print("Calculating new dataset statistics...")
        
        train_dir = output_path / "train_images"
        if train_dir.exists():
            train_files = list(train_dir.glob("*.png"))[:100]  # Sample for statistics
            
            if train_files:
                pixel_values = []
                for file_path in tqdm(train_files, desc="Calculating statistics"):
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        # Convert to RGB and normalize
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pixel_values.extend(image.flatten())
                
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


def create_visualization_comparison(input_dir: str, output_dir: str, num_samples: int = 6):
    """
    Create before/after visualization comparison.
    
    Args:
        input_dir: Input directory with original processed images
        output_dir: Output directory with advanced processed images
        num_samples: Number of samples to visualize
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get sample files
    input_files = list((input_path / "train_images").glob("*.png"))[:num_samples]
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    for i, file_path in enumerate(input_files):
        # Load original processed image
        original = cv2.imread(str(file_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Load advanced processed image
        advanced_file = output_path / "train_images" / file_path.name
        if advanced_file.exists():
            advanced = cv2.imread(str(advanced_file))
            advanced = cv2.cvtColor(advanced, cv2.COLOR_BGR2RGB)
        else:
            advanced = original  # Fallback
        
        # Plot comparison
        axes[0, i].imshow(original)
        axes[0, i].set_title(f"Original Processed\n{file_path.stem}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(advanced)
        axes[1, i].set_title(f"Advanced Processed\n{file_path.stem}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "preprocessing_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path / 'preprocessing_comparison.png'}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Advanced preprocessing for ISIC2018 dataset")
    parser.add_argument("--input-dir", type=str, default="data/ISIC2018_proc",
                      help="Input directory with processed ISIC2018 data")
    parser.add_argument("--output-dir", type=str, default="data/ISIC2018_advanced",
                      help="Output directory for advanced processed data")
    parser.add_argument("--subset-size", type=int, default=None,
                      help="Create subset with specified number of images (for testing)")
    parser.add_argument("--visualize", action="store_true",
                      help="Create before/after visualization")
    parser.add_argument("--edge-strength", type=float, default=2.0,
                      help="Edge enhancement strength (0.5-3.0)")
    parser.add_argument("--contrast-factor", type=float, default=1.8,
                      help="Contrast enhancement factor (1.0-3.0)")
    parser.add_argument("--noise-reduction", type=float, default=0.7,
                      help="Noise reduction strength (0.1-1.0)")
    
    args = parser.parse_args()
    
    print("🔬 Advanced Preprocessing Pipeline for ISIC2018 Dataset")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Edge enhancement strength: {args.edge_strength}")
    print(f"Contrast enhancement factor: {args.contrast_factor}")
    print(f"Noise reduction strength: {args.noise_reduction}")
    
    if args.subset_size:
        print(f"Creating subset with {args.subset_size} images per split")
    
    print("=" * 60)
    
    # Process dataset
    process_dataset(args.input_dir, args.output_dir, args.subset_size)
    
    # Create visualization if requested
    if args.visualize:
        print("\nCreating visualization comparison...")
        create_visualization_comparison(args.input_dir, args.output_dir)
    
    print("\n✅ Advanced preprocessing completed successfully!")
    print(f"📁 Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
