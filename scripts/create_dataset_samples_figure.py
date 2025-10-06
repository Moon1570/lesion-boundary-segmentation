#!/usr/bin/env python3
"""
Generate a figure showing sample images from the ISIC2018 dataset with their corresponding 
ground truth masks for the paper. This creates a publication-ready figure displaying
diversity of skin lesions and their segmentation masks.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Create the output directory if it doesn't exist
os.makedirs("paper_figures/keep", exist_ok=True)

# Set high-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Define paths
data_dir = Path("data/ISIC2018")
images_dir = data_dir / "train_images"
masks_dir = data_dir / "train_masks"

# Find all image and mask files
image_files = sorted(list(images_dir.glob("*.jpg")))
mask_files = {
    path.stem: path for path in masks_dir.glob("*.png")
}

# Select random samples that have both image and mask files
all_image_ids = [path.stem for path in image_files]
all_mask_ids = [id.replace("_segmentation", "") for id in mask_files.keys()]
valid_ids = [id for id in all_image_ids if id in all_mask_ids]

print(f"Found {len(valid_ids)} valid image-mask pairs")

# Select 8 random samples
if len(valid_ids) < 8:
    print(f"Warning: Only {len(valid_ids)} valid samples available")
    sample_count = len(valid_ids)
else:
    sample_count = 8
    
valid_sample_ids = random.sample(valid_ids, sample_count)
print(f"Selected {len(valid_sample_ids)} samples for visualization")

# Create 2x4 grid: 4 samples with originals and segmentations side by side
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Create custom colormap for the masks (transparent to red)
colors = [(1, 0, 0, 0), (1, 0, 0, 0.7)]  # transparent to semi-transparent red
red_cmap = LinearSegmentedColormap.from_list("TransparentRed", colors, N=256)

# Use exactly 4 samples
sample_count = min(4, len(valid_sample_ids))
for i, sample_id in enumerate(valid_sample_ids[:sample_count]):
    try:
        # Load image and mask
        img_path = images_dir / f"{sample_id}.jpg"
        mask_path = masks_dir / f"{sample_id}_segmentation.png"
        
        # Read files
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            continue
            
        mask_binary = mask > 128  # Convert to binary
        
        # Calculate row and column indices
        row = i // 2  # 0 for first two samples, 1 for next two
        col = (i % 2) * 2  # 0, 2, 0, 2 for the four samples
        
        # Plot original image
        axes[row, col].imshow(img)
        axes[row, col].axis('off')  # No axes or labels
        
        # Plot image with segmentation overlay
        axes[row, col+1].imshow(img)
        axes[row, col+1].imshow(mask_binary, cmap=red_cmap)
        axes[row, col+1].axis('off')  # No axes or labels
        
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        continue

# Simple tight layout with no titles or captions
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Minimal spacing between images

# Save in multiple formats for publication
try:
    # Save with a name reflecting the 2x4 layout
    plt.savefig("paper_figures/keep/dataset_samples_2x4.png", dpi=300, bbox_inches='tight')
    print("Saved PNG format")
    plt.savefig("paper_figures/keep/dataset_samples_2x4.pdf", bbox_inches='tight')
    print("Saved PDF format")
    plt.savefig("paper_figures/keep/dataset_samples_2x4.tiff", dpi=600, format='tiff', bbox_inches='tight')
    print("Saved TIFF format")
    print("Dataset figure with 4 samples in 2x4 layout (original + segmentation) has been generated and saved")
except Exception as e:
    print(f"Error saving figure: {e}")