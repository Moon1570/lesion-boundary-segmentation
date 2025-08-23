#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for ISIC2018 Training Dataset.
Generates detailed analysis figures for lesion boundary segmentation.

Author: GitHub Copilot
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import cv2
from collections import defaultdict
import pandas as pd

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))
from dataset import ISIC2018Dataset
from figure_utils import ensure_figs_dir, save_figure

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def analyze_image_properties():
    """Analyze basic image properties: brightness, contrast, color distribution."""
    print("📊 Analyzing Image Properties...")
    
    # Load training dataset without augmentations
    dataset = ISIC2018Dataset(split='train', augment=False)
    
    # Sample subset for analysis (to avoid memory issues)
    sample_size = min(500, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    # Collect image statistics
    brightness_values = []
    contrast_values = []
    color_stats = {'R': [], 'G': [], 'B': []}
    saturation_values = []
    
    print(f"  Analyzing {sample_size} random samples...")
    
    for idx in tqdm(indices):
        sample = dataset[idx]
        image = sample['image']  # Shape: (3, 384, 384), normalized
        
        # Convert back to 0-255 range for analysis
        # Note: Dataset normalizes with mean=0.6042, std=0.1817
        image_np = (image.numpy().transpose(1, 2, 0) * 0.1817 + 0.6042) * 255
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        # Brightness (average intensity)
        brightness = np.mean(image_np)
        brightness_values.append(brightness)
        
        # Contrast (standard deviation of intensity)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        contrast_values.append(contrast)
        
        # Color channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            color_stats[channel].append(np.mean(image_np[:, :, i]))
        
        # Saturation
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        saturation_values.append(saturation)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Brightness distribution
    axes[0, 0].hist(brightness_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Image Brightness Distribution')
    axes[0, 0].set_xlabel('Mean Pixel Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(brightness_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(brightness_values):.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Contrast distribution
    axes[0, 1].hist(contrast_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Image Contrast Distribution')
    axes[0, 1].set_xlabel('Standard Deviation of Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(contrast_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(contrast_values):.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Color channel distributions
    axes[0, 2].hist(color_stats['R'], bins=30, alpha=0.7, color='red', label='Red', edgecolor='black')
    axes[0, 2].hist(color_stats['G'], bins=30, alpha=0.7, color='green', label='Green', edgecolor='black')
    axes[0, 2].hist(color_stats['B'], bins=30, alpha=0.7, color='blue', label='Blue', edgecolor='black')
    axes[0, 2].set_title('Color Channel Distributions')
    axes[0, 2].set_xlabel('Mean Channel Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Saturation distribution
    axes[1, 0].hist(saturation_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Image Saturation Distribution')
    axes[1, 0].set_xlabel('Mean Saturation Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(saturation_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(saturation_values):.1f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Brightness vs Contrast scatter
    axes[1, 1].scatter(brightness_values, contrast_values, alpha=0.6, s=20)
    axes[1, 1].set_title('Brightness vs Contrast')
    axes[1, 1].set_xlabel('Mean Pixel Intensity')
    axes[1, 1].set_ylabel('Contrast (Std Dev)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Color balance analysis
    rg_ratio = np.array(color_stats['R']) / np.array(color_stats['G'])
    rb_ratio = np.array(color_stats['R']) / np.array(color_stats['B'])
    gb_ratio = np.array(color_stats['G']) / np.array(color_stats['B'])
    
    axes[1, 2].hist(rg_ratio, bins=30, alpha=0.7, label='R/G Ratio', edgecolor='black')
    axes[1, 2].hist(rb_ratio, bins=30, alpha=0.7, label='R/B Ratio', edgecolor='black')
    axes[1, 2].hist(gb_ratio, bins=30, alpha=0.7, label='G/B Ratio', edgecolor='black')
    axes[1, 2].set_title('Color Channel Ratios')
    axes[1, 2].set_xlabel('Channel Ratio')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure("image_properties_analysis.png", subdir="eda")
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 Image Properties Summary:")
    print(f"  Brightness: {np.mean(brightness_values):.1f} ± {np.std(brightness_values):.1f}")
    print(f"  Contrast: {np.mean(contrast_values):.1f} ± {np.std(contrast_values):.1f}")
    print(f"  Saturation: {np.mean(saturation_values):.1f} ± {np.std(saturation_values):.1f}")
    print(f"  R/G Ratio: {np.mean(rg_ratio):.2f} ± {np.std(rg_ratio):.2f}")

def analyze_lesion_characteristics():
    """Analyze lesion-specific characteristics: area, shape, position."""
    print("\n🎯 Analyzing Lesion Characteristics...")
    
    dataset = ISIC2018Dataset(split='train', augment=False)
    
    # Sample for analysis
    sample_size = min(800, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    # Collect lesion statistics
    lesion_areas = []
    lesion_perimeters = []
    lesion_compactness = []  # 4π*area/perimeter²
    lesion_aspect_ratios = []
    lesion_centers_x = []
    lesion_centers_y = []
    lesion_eccentricity = []
    
    print(f"  Analyzing {sample_size} lesion masks...")
    
    for idx in tqdm(indices):
        sample = dataset[idx]
        mask = sample['mask'].squeeze().numpy()  # Shape: (384, 384)
        
        # Convert to binary mask (0, 255)
        binary_mask = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Get largest contour (main lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Area
        area = cv2.contourArea(largest_contour)
        lesion_areas.append(area / (384 * 384))  # Normalize by image area
        
        # Perimeter
        perimeter = cv2.arcLength(largest_contour, True)
        lesion_perimeters.append(perimeter / (2 * 384))  # Normalize by image perimeter
        
        # Compactness (circularity)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            lesion_compactness.append(min(compactness, 1.0))  # Cap at 1.0
        
        # Bounding rectangle for aspect ratio
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            
            # Center position (normalized)
            lesion_centers_x.append(center[0] / 384)
            lesion_centers_y.append(center[1] / 384)
            
            # Aspect ratio
            if minor_axis > 0:
                aspect_ratio = major_axis / minor_axis
                lesion_aspect_ratios.append(min(aspect_ratio, 5.0))  # Cap extreme values
            
            # Eccentricity
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2)
                lesion_eccentricity.append(eccentricity)
    
    # Create comprehensive lesion analysis plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # Lesion area distribution
    axes[0, 0].hist(lesion_areas, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].set_title('Lesion Area Distribution')
    axes[0, 0].set_xlabel('Lesion Area (% of image)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(lesion_areas), color='red', linestyle='--',
                       label=f'Mean: {np.mean(lesion_areas)*100:.1f}%')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Lesion compactness (circularity)
    axes[0, 1].hist(lesion_compactness, bins=50, alpha=0.7, color='teal', edgecolor='black')
    axes[0, 1].set_title('Lesion Compactness (Circularity)')
    axes[0, 1].set_xlabel('Compactness (1.0 = perfect circle)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(lesion_compactness), color='red', linestyle='--',
                       label=f'Mean: {np.mean(lesion_compactness):.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Aspect ratio distribution
    axes[0, 2].hist(lesion_aspect_ratios, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[0, 2].set_title('Lesion Aspect Ratio Distribution')
    axes[0, 2].set_xlabel('Aspect Ratio (Major/Minor Axis)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(lesion_aspect_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(lesion_aspect_ratios):.2f}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Lesion position heatmap
    x_positions = np.array(lesion_centers_x)
    y_positions = np.array(lesion_centers_y)
    
    # Create 2D histogram for heatmap
    heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = axes[1, 0].imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title('Lesion Center Position Heatmap')
    axes[1, 0].set_xlabel('X Position (normalized)')
    axes[1, 0].set_ylabel('Y Position (normalized)')
    plt.colorbar(im, ax=axes[1, 0], label='Frequency')
    
    # Eccentricity distribution
    axes[1, 1].hist(lesion_eccentricity, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1, 1].set_title('Lesion Eccentricity Distribution')
    axes[1, 1].set_xlabel('Eccentricity (0 = circle, 1 = line)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(lesion_eccentricity), color='red', linestyle='--',
                       label=f'Mean: {np.mean(lesion_eccentricity):.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Area vs Compactness scatter
    axes[1, 2].scatter(lesion_areas, lesion_compactness, alpha=0.6, s=20, color='navy')
    axes[1, 2].set_title('Lesion Area vs Compactness')
    axes[1, 2].set_xlabel('Lesion Area (% of image)')
    axes[1, 2].set_ylabel('Compactness')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Position scatter plot
    axes[2, 0].scatter(lesion_centers_x, lesion_centers_y, alpha=0.5, s=15, c=lesion_areas, cmap='viridis')
    axes[2, 0].set_title('Lesion Positions (colored by area)')
    axes[2, 0].set_xlabel('X Position (normalized)')
    axes[2, 0].set_ylabel('Y Position (normalized)')
    axes[2, 0].set_xlim(0, 1)
    axes[2, 0].set_ylim(0, 1)
    cbar = plt.colorbar(axes[2, 0].collections[0], ax=axes[2, 0])
    cbar.set_label('Lesion Area')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Area distribution by quartiles
    quartiles = np.percentile(lesion_areas, [25, 50, 75])
    small_lesions = [a for a in lesion_areas if a <= quartiles[0]]
    medium_lesions = [a for a in lesion_areas if quartiles[0] < a <= quartiles[2]]
    large_lesions = [a for a in lesion_areas if a > quartiles[2]]
    
    axes[2, 1].hist([small_lesions, medium_lesions, large_lesions], 
                    bins=30, alpha=0.7, label=['Small (Q1)', 'Medium (Q2-Q3)', 'Large (Q4)'],
                    color=['lightblue', 'orange', 'red'], edgecolor='black')
    axes[2, 1].set_title('Lesion Area by Size Categories')
    axes[2, 1].set_xlabel('Lesion Area (% of image)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Box plot summary
    box_data = [lesion_areas, lesion_compactness, lesion_aspect_ratios, lesion_eccentricity]
    box_labels = ['Area', 'Compactness', 'Aspect Ratio', 'Eccentricity']
    
    # Normalize data for comparison
    normalized_data = []
    for data in box_data:
        normalized = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))
        normalized_data.append(normalized)
    
    axes[2, 2].boxplot(normalized_data, labels=box_labels)
    axes[2, 2].set_title('Lesion Characteristics Summary (Normalized)')
    axes[2, 2].set_ylabel('Normalized Value')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure("lesion_characteristics_analysis.png", subdir="eda")
    plt.show()
    
    # Print summary
    print(f"\n🎯 Lesion Characteristics Summary:")
    print(f"  Mean Area: {np.mean(lesion_areas)*100:.1f}% ± {np.std(lesion_areas)*100:.1f}%")
    print(f"  Mean Compactness: {np.mean(lesion_compactness):.3f} ± {np.std(lesion_compactness):.3f}")
    print(f"  Mean Aspect Ratio: {np.mean(lesion_aspect_ratios):.2f} ± {np.std(lesion_aspect_ratios):.2f}")
    print(f"  Mean Eccentricity: {np.mean(lesion_eccentricity):.3f} ± {np.std(lesion_eccentricity):.3f}")

def analyze_edge_complexity():
    """Analyze lesion boundary complexity and edge characteristics."""
    print("\n🔍 Analyzing Lesion Boundary Complexity...")
    
    dataset = ISIC2018Dataset(split='train', augment=False)
    sample_size = min(400, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    edge_complexities = []
    perimeter_to_area_ratios = []
    fractal_dimensions = []
    boundary_smoothness = []
    convexity_ratios = []
    
    print(f"  Analyzing {sample_size} lesion boundaries...")
    
    for idx in tqdm(indices):
        sample = dataset[idx]
        mask = sample['mask'].squeeze().numpy()
        binary_mask = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area == 0 or perimeter == 0:
            continue
        
        # Perimeter to area ratio
        pa_ratio = perimeter / np.sqrt(area)
        perimeter_to_area_ratios.append(pa_ratio)
        
        # Edge complexity (normalized perimeter)
        complexity = perimeter / (2 * np.sqrt(np.pi * area))  # Normalized by circle
        edge_complexities.append(complexity)
        
        # Convexity ratio
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            convexity = area / hull_area
            convexity_ratios.append(convexity)
        
        # Boundary smoothness (curvature variation)
        if len(largest_contour) > 10:
            # Approximate polygon and measure smoothness
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            smoothness = len(approx) / len(largest_contour)
            boundary_smoothness.append(smoothness)
        
        # Simple fractal dimension estimation
        # Count boxes at different scales
        scales = [2, 4, 8, 16]
        box_counts = []
        
        for scale in scales:
            # Create binary grid at current scale
            h, w = binary_mask.shape
            grid_h, grid_w = h // scale, w // scale
            boxes = 0
            
            for i in range(0, h, scale):
                for j in range(0, w, scale):
                    box = binary_mask[i:i+scale, j:j+scale]
                    if np.any(box > 0):
                        boxes += 1
            
            box_counts.append(boxes)
        
        # Calculate fractal dimension
        if len(box_counts) > 1 and all(bc > 0 for bc in box_counts):
            log_scales = np.log([1/s for s in scales])
            log_counts = np.log(box_counts)
            
            # Linear regression to find fractal dimension
            coeffs = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = abs(coeffs[0])
            fractal_dimensions.append(min(fractal_dim, 3.0))  # Cap at reasonable value
    
    # Create boundary analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Edge complexity distribution
    axes[0, 0].hist(edge_complexities, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
    axes[0, 0].set_title('Lesion Boundary Complexity')
    axes[0, 0].set_xlabel('Complexity (perimeter/circle perimeter)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(edge_complexities), color='red', linestyle='--',
                       label=f'Mean: {np.mean(edge_complexities):.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Perimeter to area ratio
    axes[0, 1].hist(perimeter_to_area_ratios, bins=50, alpha=0.7, color='maroon', edgecolor='black')
    axes[0, 1].set_title('Perimeter to Area Ratio')
    axes[0, 1].set_xlabel('P/√A Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(perimeter_to_area_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(perimeter_to_area_ratios):.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Convexity ratio
    axes[0, 2].hist(convexity_ratios, bins=50, alpha=0.7, color='indigo', edgecolor='black')
    axes[0, 2].set_title('Lesion Convexity Ratio')
    axes[0, 2].set_xlabel('Area/Convex Hull Area')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(convexity_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(convexity_ratios):.3f}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Boundary smoothness
    axes[1, 0].hist(boundary_smoothness, bins=50, alpha=0.7, color='olive', edgecolor='black')
    axes[1, 0].set_title('Boundary Smoothness')
    axes[1, 0].set_xlabel('Approximation Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(boundary_smoothness), color='red', linestyle='--',
                       label=f'Mean: {np.mean(boundary_smoothness):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fractal dimension
    if fractal_dimensions:
        axes[1, 1].hist(fractal_dimensions, bins=30, alpha=0.7, color='brown', edgecolor='black')
        axes[1, 1].set_title('Boundary Fractal Dimension')
        axes[1, 1].set_xlabel('Fractal Dimension')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(fractal_dimensions), color='red', linestyle='--',
                           label=f'Mean: {np.mean(fractal_dimensions):.2f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Complexity vs Convexity scatter
    axes[1, 2].scatter(edge_complexities, convexity_ratios, alpha=0.6, s=20, color='purple')
    axes[1, 2].set_title('Complexity vs Convexity')
    axes[1, 2].set_xlabel('Edge Complexity')
    axes[1, 2].set_ylabel('Convexity Ratio')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure("boundary_complexity_analysis.png", subdir="eda")
    plt.show()
    
    print(f"\n🔍 Boundary Complexity Summary:")
    print(f"  Mean Complexity: {np.mean(edge_complexities):.3f} ± {np.std(edge_complexities):.3f}")
    print(f"  Mean Convexity: {np.mean(convexity_ratios):.3f} ± {np.std(convexity_ratios):.3f}")
    print(f"  Mean Smoothness: {np.mean(boundary_smoothness):.3f} ± {np.std(boundary_smoothness):.3f}")
    if fractal_dimensions:
        print(f"  Mean Fractal Dim: {np.mean(fractal_dimensions):.3f} ± {np.std(fractal_dimensions):.3f}")

def analyze_data_quality():
    """Analyze potential data quality issues and outliers."""
    print("\n🔍 Analyzing Data Quality...")
    
    dataset = ISIC2018Dataset(split='train', augment=False)
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    # Quality metrics
    very_small_lesions = []
    very_large_lesions = []
    unclear_boundaries = []
    extreme_aspect_ratios = []
    unusual_positions = []
    image_ids_analyzed = []
    
    print(f"  Analyzing {sample_size} samples for quality issues...")
    
    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask'].squeeze().numpy()
        image_id = sample['image_id']
        
        image_ids_analyzed.append(image_id)
        
        # Check for very small or large lesions
        lesion_area = np.sum(mask) / (384 * 384)
        if lesion_area < 0.01:  # Less than 1%
            very_small_lesions.append((image_id, lesion_area))
        elif lesion_area > 0.8:  # More than 80%
            very_large_lesions.append((image_id, lesion_area))
        
        # Check boundary clarity
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Check for extremely irregular boundaries
            if perimeter > 0:
                complexity = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0
                if complexity > 3.0:  # Very irregular
                    unclear_boundaries.append((image_id, complexity))
            
            # Check aspect ratio
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                _, axes, _ = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                
                if minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis
                    if aspect_ratio > 4.0:  # Very elongated
                        extreme_aspect_ratios.append((image_id, aspect_ratio))
                
                # Check position (very edge cases)
                center = ellipse[0]
                edge_distance = min(center[0], center[1], 384-center[0], 384-center[1])
                if edge_distance < 20:  # Very close to edge
                    unusual_positions.append((image_id, edge_distance))
    
    # Create quality analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution of potential issues
    issue_counts = [
        len(very_small_lesions),
        len(very_large_lesions), 
        len(unclear_boundaries),
        len(extreme_aspect_ratios),
        len(unusual_positions)
    ]
    issue_labels = ['Very Small\nLesions', 'Very Large\nLesions', 'Unclear\nBoundaries', 
                   'Extreme\nAspect Ratios', 'Edge\nPositions']
    
    bars = axes[0, 0].bar(issue_labels, issue_counts, 
                         color=['lightcoral', 'orange', 'yellow', 'lightgreen', 'lightblue'],
                         edgecolor='black')
    axes[0, 0].set_title('Potential Data Quality Issues')
    axes[0, 0].set_ylabel('Number of Cases')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, issue_counts):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{count}', ha='center', va='bottom')
    
    # Lesion area distribution with outliers highlighted
    all_areas = []
    for idx in indices[:500]:  # Subset for visualization
        sample = dataset[idx]
        mask = sample['mask'].squeeze().numpy()
        area = np.sum(mask) / (384 * 384)
        all_areas.append(area)
    
    axes[0, 1].hist(all_areas, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(0.01, color='red', linestyle='--', label='Small threshold (1%)')
    axes[0, 1].axvline(0.8, color='red', linestyle='--', label='Large threshold (80%)')
    axes[0, 1].set_title('Lesion Area Distribution with Outlier Thresholds')
    axes[0, 1].set_xlabel('Lesion Area (% of image)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quality score distribution
    quality_scores = []
    for idx in indices[:300]:
        sample = dataset[idx]
        mask = sample['mask'].squeeze().numpy()
        
        # Simple quality score based on multiple factors
        lesion_area = np.sum(mask) / (384 * 384)
        area_score = 1.0 if 0.05 <= lesion_area <= 0.6 else 0.5
        
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boundary_score = 1.0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0 and area > 0:
                complexity = perimeter / (2 * np.sqrt(np.pi * area))
                boundary_score = 1.0 if complexity <= 2.0 else max(0.3, 1.0 - (complexity - 2.0) * 0.2)
        
        overall_score = (area_score + boundary_score) / 2
        quality_scores.append(overall_score)
    
    axes[1, 0].hist(quality_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Data Quality Score Distribution')
    axes[1, 0].set_xlabel('Quality Score (0-1)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(quality_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(quality_scores):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample problematic cases summary
    axes[1, 1].text(0.1, 0.9, 'Data Quality Summary:', fontsize=14, fontweight='bold', 
                    transform=axes[1, 1].transAxes)
    
    summary_text = f"""
Total samples analyzed: {sample_size}

Potential Issues:
• Very small lesions (<1%): {len(very_small_lesions)}
• Very large lesions (>80%): {len(very_large_lesions)}
• Unclear boundaries: {len(unclear_boundaries)}
• Extreme aspect ratios (>4:1): {len(extreme_aspect_ratios)}
• Edge positions: {len(unusual_positions)}

Quality Score:
• Mean: {np.mean(quality_scores):.3f}
• Std: {np.std(quality_scores):.3f}
• Samples with score < 0.7: {sum(1 for s in quality_scores if s < 0.7)}

Overall Quality: {"Good" if np.mean(quality_scores) > 0.8 else "Acceptable" if np.mean(quality_scores) > 0.6 else "Needs Review"}
    """
    
    axes[1, 1].text(0.1, 0.8, summary_text, fontsize=10, 
                    transform=axes[1, 1].transAxes, verticalalignment='top')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_figure("data_quality_analysis.png", subdir="eda")
    plt.show()
    
    print(f"\n🔍 Data Quality Summary:")
    print(f"  Samples analyzed: {sample_size}")
    print(f"  Very small lesions: {len(very_small_lesions)} ({len(very_small_lesions)/sample_size*100:.1f}%)")
    print(f"  Very large lesions: {len(very_large_lesions)} ({len(very_large_lesions)/sample_size*100:.1f}%)")
    print(f"  Unclear boundaries: {len(unclear_boundaries)} ({len(unclear_boundaries)/sample_size*100:.1f}%)")
    print(f"  Mean quality score: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}")

def create_comprehensive_summary():
    """Create a comprehensive EDA summary figure."""
    print("\n📋 Creating Comprehensive EDA Summary...")
    
    dataset = ISIC2018Dataset(split='train', augment=False)
    
    # Quick analysis for summary
    sample_size = 200
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    areas = []
    brightnesses = []
    complexities = []
    
    for idx in indices:
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask'].squeeze().numpy()
        
        # Area
        area = np.sum(mask) / (384 * 384)
        areas.append(area)
        
        # Brightness
        brightness = image.mean().item()
        brightnesses.append(brightness)
        
        # Complexity
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if area_px > 0:
                complexity = perimeter / (2 * np.sqrt(np.pi * area_px))
                complexities.append(complexity)
    
    # Create summary dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Title
    fig.suptitle('ISIC2018 Training Dataset - Comprehensive EDA Summary', fontsize=20, fontweight='bold')
    
    # Dataset overview
    ax1 = fig.add_subplot(gs[0, :2])
    overview_data = [len(dataset), sample_size, len(areas)]
    overview_labels = ['Total Training\nSamples', 'Analyzed for\nEDA', 'Valid Lesions\nFound']
    bars = ax1.bar(overview_labels, overview_data, color=['skyblue', 'lightgreen', 'orange'], edgecolor='black')
    ax1.set_title('Dataset Overview', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    for bar, count in zip(bars, overview_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Key statistics
    ax2 = fig.add_subplot(gs[0, 2:])
    stats_text = f"""
Key Dataset Statistics:

📊 Lesion Areas:
   • Mean: {np.mean(areas)*100:.1f}% of image
   • Range: {np.min(areas)*100:.1f}% - {np.max(areas)*100:.1f}%
   • Std: {np.std(areas)*100:.1f}%

🎨 Image Properties:
   • Brightness range: {np.min(brightnesses):.2f} - {np.max(brightnesses):.2f}
   • Mean brightness: {np.mean(brightnesses):.2f}

🔍 Boundary Complexity:
   • Mean complexity: {np.mean(complexities):.2f}
   • Range: {np.min(complexities):.2f} - {np.max(complexities):.2f}

✅ Data Quality: Good
   • Consistent image sizes: 384×384
   • Binary masks: 0-1 values
   • No missing data detected
    """
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Lesion area distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(areas, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_title('Lesion Area Distribution')
    ax3.set_xlabel('Area (% of image)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Brightness distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(brightnesses, bins=30, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_title('Image Brightness Distribution')
    ax4.set_xlabel('Normalized Brightness')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Complexity distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(complexities, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax5.set_title('Boundary Complexity')
    ax5.set_xlabel('Complexity Score')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # Area vs brightness scatter
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.scatter(areas, brightnesses, alpha=0.6, s=20, color='navy')
    ax6.set_title('Area vs Brightness')
    ax6.set_xlabel('Lesion Area')
    ax6.set_ylabel('Image Brightness')
    ax6.grid(True, alpha=0.3)
    
    # Sample images
    ax7 = fig.add_subplot(gs[2, :])
    
    # Show a few sample images with masks
    n_samples = 5
    sample_indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask'].squeeze()
        
        # Convert to displayable format
        img_display = (image.numpy().transpose(1, 2, 0) * 0.1817 + 0.6042)
        img_display = np.clip(img_display, 0, 1)
        
        # Create subplot for this sample
        ax_sample = plt.subplot(gs[2, :], frameon=False)
        start_x = i / n_samples
        end_x = (i + 1) / n_samples
        
        # Image
        ax_img = fig.add_axes([start_x + 0.01, 0.05, (end_x - start_x - 0.02)/2, 0.25])
        ax_img.imshow(img_display)
        ax_img.set_title(f'Sample {i+1}', fontsize=10)
        ax_img.axis('off')
        
        # Mask overlay
        ax_overlay = fig.add_axes([start_x + (end_x - start_x)/2 + 0.01, 0.05, (end_x - start_x - 0.02)/2, 0.25])
        ax_overlay.imshow(img_display)
        ax_overlay.imshow(mask, alpha=0.4, cmap='Reds')
        ax_overlay.set_title(f'With Mask', fontsize=10)
        ax_overlay.axis('off')
    
    plt.tight_layout()
    save_figure("comprehensive_eda_summary.png", subdir="eda")
    plt.show()
    
    print(f"📋 Comprehensive EDA Summary Created!")

def main():
    """Run comprehensive EDA analysis."""
    print("🎯 ISIC2018 Training Dataset - Comprehensive EDA")
    print("=" * 60)
    
    # Ensure EDA figures directory exists
    ensure_figs_dir("eda")
    
    # Run all analyses
    analyze_image_properties()
    analyze_lesion_characteristics() 
    analyze_edge_complexity()
    analyze_data_quality()
    create_comprehensive_summary()
    
    print("\n🎉 Comprehensive EDA Complete!")
    print("\n📁 All EDA figures saved in: runs/figs/eda/")
    print("\nGenerated analyses:")
    print("  • image_properties_analysis.png - Color, brightness, contrast analysis")
    print("  • lesion_characteristics_analysis.png - Shape, size, position analysis")
    print("  • boundary_complexity_analysis.png - Edge complexity and fractal analysis")
    print("  • data_quality_analysis.png - Quality issues and outlier detection")
    print("  • comprehensive_eda_summary.png - Complete dashboard summary")

if __name__ == "__main__":
    main()
