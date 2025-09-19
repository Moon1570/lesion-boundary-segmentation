#!/usr/bin/env python3
"""
Create comparison visualizations between original and advanced processed images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random


def create_comparison_visualization(input_dir: str, output_dir: str, 
                                  num_samples: int = 8, save_path: str = None):
    """
    Create side-by-side comparison of original vs advanced processed images.
    
    Args:
        input_dir: Directory with original processed images
        output_dir: Directory with advanced processed images  
        num_samples: Number of samples to show
        save_path: Where to save the visualization
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get sample files from train_images
    input_files = list((input_path / "train_images").glob("*.png"))
    
    # Randomly sample if we have more files than needed
    if len(input_files) > num_samples:
        input_files = random.sample(input_files, num_samples)
    else:
        input_files = input_files[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, len(input_files), figsize=(4*len(input_files), 8))
    
    if len(input_files) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, file_path in enumerate(input_files):
        # Load original processed image
        original = cv2.imread(str(file_path))
        if original is not None:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            continue
        
        # Load advanced processed image
        advanced_file = output_path / "train_images" / file_path.name
        if advanced_file.exists():
            advanced = cv2.imread(str(advanced_file))
            if advanced is not None:
                advanced = cv2.cvtColor(advanced, cv2.COLOR_BGR2RGB)
            else:
                advanced = original
        else:
            advanced = original
        
        # Plot comparison
        axes[0, i].imshow(original)
        axes[0, i].set_title(f"Original\n{file_path.stem}", fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(advanced)
        axes[1, i].set_title(f"Advanced\n{file_path.stem}", fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle("Original vs Advanced Processed Images", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    if save_path is None:
        save_path = output_path / "preprocessing_comparison.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Visualization saved: {save_path}")


def create_edge_enhancement_demo(input_dir: str, output_dir: str, save_path: str = None):
    """
    Create a detailed demo showing edge enhancement effects.
    
    Args:
        input_dir: Directory with original processed images
        output_dir: Directory with advanced processed images
        save_path: Where to save the demo
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get a few sample files
    input_files = list((input_path / "train_images").glob("*.png"))[:4]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    for i, file_path in enumerate(input_files):
        if i >= 4:
            break
            
        # Load images
        original = cv2.imread(str(file_path))
        if original is not None:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            continue
            
        advanced_file = output_path / "train_images" / file_path.name
        if advanced_file.exists():
            advanced = cv2.imread(str(advanced_file))
            if advanced is not None:
                advanced = cv2.cvtColor(advanced, cv2.COLOR_BGR2RGB)
            else:
                advanced = original
        else:
            advanced = original
        
        # Plot side by side
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original: {file_path.stem}", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(advanced)
        axes[i, 1].set_title(f"Enhanced: {file_path.stem}", fontsize=10)
        axes[i, 1].axis('off')
    
    plt.suptitle("Edge Enhancement & Contrast Improvement Results", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        save_path = output_path / "edge_enhancement_demo.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🎨 Edge enhancement demo saved: {save_path}")


def analyze_preprocessing_effects(input_dir: str, output_dir: str):
    """
    Analyze the effects of preprocessing on image statistics.
    
    Args:
        input_dir: Directory with original processed images
        output_dir: Directory with advanced processed images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Sample images for analysis
    input_files = list((input_path / "train_images").glob("*.png"))[:20]
    
    original_stats = []
    advanced_stats = []
    
    print("📈 Analyzing preprocessing effects...")
    
    for file_path in input_files:
        # Load original
        original = cv2.imread(str(file_path))
        if original is not None:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            original_stats.append({
                'mean': np.mean(original),
                'std': np.std(original),
                'contrast': np.std(original) / np.mean(original) if np.mean(original) > 0 else 0
            })
        
        # Load advanced
        advanced_file = output_path / "train_images" / file_path.name
        if advanced_file.exists():
            advanced = cv2.imread(str(advanced_file))
            if advanced is not None:
                advanced = cv2.cvtColor(advanced, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                advanced_stats.append({
                    'mean': np.mean(advanced),
                    'std': np.std(advanced),
                    'contrast': np.std(advanced) / np.mean(advanced) if np.mean(advanced) > 0 else 0
                })
    
    if original_stats and advanced_stats:
        orig_mean = np.mean([s['mean'] for s in original_stats])
        orig_std = np.mean([s['std'] for s in original_stats])
        orig_contrast = np.mean([s['contrast'] for s in original_stats])
        
        adv_mean = np.mean([s['mean'] for s in advanced_stats])
        adv_std = np.mean([s['std'] for s in advanced_stats])
        adv_contrast = np.mean([s['contrast'] for s in advanced_stats])
        
        print("\n📊 Preprocessing Effects Analysis:")
        print("=" * 50)
        print(f"Original  - Mean: {orig_mean:.4f}, Std: {orig_std:.4f}, Contrast: {orig_contrast:.4f}")
        print(f"Advanced  - Mean: {adv_mean:.4f}, Std: {adv_std:.4f}, Contrast: {adv_contrast:.4f}")
        print(f"Changes   - Mean: {adv_mean-orig_mean:+.4f}, Std: {adv_std-orig_std:+.4f}, Contrast: {adv_contrast-orig_contrast:+.4f}")
        print("=" * 50)
        
        improvement_metrics = {
            'contrast_improvement': ((adv_contrast - orig_contrast) / orig_contrast) * 100,
            'std_change': ((adv_std - orig_std) / orig_std) * 100,
            'mean_stability': abs(adv_mean - orig_mean) / orig_mean * 100
        }
        
        print(f"📈 Contrast improved by: {improvement_metrics['contrast_improvement']:.1f}%")
        print(f"📊 Std deviation change: {improvement_metrics['std_change']:.1f}%")
        print(f"🎯 Mean stability: {improvement_metrics['mean_stability']:.1f}% deviation")


def main():
    """Main function for creating visualizations."""
    parser = argparse.ArgumentParser(description="Create preprocessing comparison visualizations")
    parser.add_argument("--input-dir", type=str, default="data/ISIC2018_proc",
                      help="Input directory with original processed images")
    parser.add_argument("--output-dir", type=str, default="data/ISIC2018_advanced",
                      help="Output directory with advanced processed images")
    parser.add_argument("--num-samples", type=int, default=6,
                      help="Number of sample images to show")
    
    args = parser.parse_args()
    
    print("🎨 Creating Advanced Preprocessing Visualizations")
    print("=" * 50)
    
    # Create comparison visualization
    create_comparison_visualization(
        args.input_dir, 
        args.output_dir, 
        num_samples=args.num_samples
    )
    
    # Create edge enhancement demo
    create_edge_enhancement_demo(
        args.input_dir,
        args.output_dir
    )
    
    # Analyze preprocessing effects
    analyze_preprocessing_effects(
        args.input_dir,
        args.output_dir
    )
    
    print("\n✅ All visualizations created successfully!")


if __name__ == "__main__":
    main()
