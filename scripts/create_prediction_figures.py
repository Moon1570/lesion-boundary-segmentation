#!/usr/bin/env python3
"""
Generate publication-ready prediction visualizations using actual validation data.

This script creates side-by-side comparisons of input images, ground truth masks,
and model predictions for inclusion in academic papers.

Usage:
    python scripts/create_prediction_figures.py --model_path runs/ckpts/checkpoints/best_checkpoint.pth
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.dataset import create_data_loaders, ISIC2018Dataset
from models.unet import UNet
from utils.metrics import SegmentationMetrics

# Suppress warnings
warnings.filterwarnings('ignore')


class PredictionVisualizer:
    """Create publication-ready prediction visualizations."""
    
    def __init__(self, model_path: str, config_path: str = "configs/train_with_masks.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Load data
        self.data_loaders = self.load_data(config_path)
        
        # Metrics calculator
        self.metrics = SegmentationMetrics()
        
        # Setup matplotlib
        self.setup_style()
        
    def setup_style(self):
        """Setup matplotlib style for publication."""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Convert path to lowercase for case-insensitive comparison
        model_path_lower = model_path.lower()
        
        # Check model type from checkpoint path
        if "lightweight_duaskinseg" in model_path_lower:
            # Import the LightweightDuaSkinSeg model
            from models.lightweight_duaskinseg import LightweightDuaSkinSeg
            
            # Create appropriate model
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
                model = LightweightDuaSkinSeg(
                    img_size=model_config.get('img_size', 384),
                    patch_size=model_config.get('patch_size', 16),
                    in_channels=model_config.get('in_channels', 3),
                    num_classes=model_config.get('out_channels', 1)
                )
            else:
                model = LightweightDuaSkinSeg(img_size=384, patch_size=16, in_channels=3, num_classes=1)
                
            print(f"📋 Loading LightweightDuaSkinSeg model")
        
        elif "duaskinseg" in model_path_lower:
            # Import the DuaSkinSeg model
            from models.duaskinseg import DuaSkinSeg
            
            # Create appropriate model
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
                model = DuaSkinSeg(
                    img_size=model_config.get('img_size', 256),
                    in_channels=model_config.get('in_channels', 3),
                    num_classes=model_config.get('out_channels', 1)
                )
            else:
                model = DuaSkinSeg(img_size=256, in_channels=3, num_classes=1)
            
            print(f"📋 Loading DuaSkinSeg model")
        
        elif "unetmamba" in model_path_lower or "mamba_unet" in model_path_lower:
            # Import the MambaUNet model
            from models.mamba_unet import MambaUNet
            
            # Create appropriate model
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
                model = MambaUNet(
                    img_size=model_config.get('img_size', 256),
                    n_channels=model_config.get('in_channels', 3),
                    n_classes=model_config.get('out_channels', 1)
                )
            else:
                model = MambaUNet(img_size=256, n_channels=3, n_classes=1)
                
            print(f"📋 Loading MambaUNet model")
            
        else:
            # Default: UNet model
            print(f"📋 Loading UNet model")
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
                model = UNet(
                    n_channels=model_config.get('in_channels', 3),
                    n_classes=model_config.get('out_channels', 1)
                )
            else:
                model = UNet(n_channels=3, n_classes=1)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ Model loaded from {model_path}")
        return model
    
    def load_data(self, config_path: str):
        """Load validation data."""
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        data_config = config['data']
        
        # Create data loaders with explicit arguments
        data_loaders = create_data_loaders(
            data_dir=data_config['data_dir'],
            splits_dir=data_config.get('splits_dir', 'splits'),
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            image_size=data_config['image_size'],
            pin_memory=data_config.get('pin_memory', True)
        )
        print(f"✅ Data loaded: {len(data_loaders['val'])} validation batches")
        
        return data_loaders
    
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Generate predictions for a batch of images."""
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.sigmoid(outputs)
            return predictions
    
    def create_prediction_grid(self, num_samples: int = 8, save_path: str = "prediction_grid.png"):
        """Create a grid of prediction comparisons."""
        # Get validation samples
        val_loader = self.data_loaders['val']
        samples_collected = 0
        
        # Storage for samples
        images_list = []
        masks_list = []
        predictions_list = []
        metrics_list = []
        
        print(f"🔍 Collecting {num_samples} validation samples...")
        
        for batch_idx, batch_data in enumerate(val_loader):
            if samples_collected >= num_samples:
                break
            
            # The batch_data should be a dictionary with 'image', 'mask', and 'image_id'
            if isinstance(batch_data, dict):
                images = batch_data['image']
                masks = batch_data['mask']
                image_ids = batch_data.get('image_id', None)
            else:
                print(f"Unexpected batch format: {type(batch_data)}")
                print(f"Batch contents: {batch_data}")
                continue
                
            # Generate predictions
            predictions = self.predict_batch(images)
            
            # Process each sample in batch
            batch_size = min(images.shape[0], num_samples - samples_collected)
            
            for i in range(batch_size):
                # Convert to numpy
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                mask = masks[i].cpu().numpy().squeeze()
                pred = predictions[i].cpu().numpy().squeeze()
                
                # Normalize image for display
                img = (img - img.min()) / (img.max() - img.min())
                
                # Threshold prediction
                pred_binary = (pred > 0.5).astype(np.float32)
                
                # Calculate metrics
                sample_metrics = self.calculate_sample_metrics(mask, pred_binary)
                
                # Store
                images_list.append(img)
                masks_list.append(mask)
                predictions_list.append(pred_binary)
                metrics_list.append(sample_metrics)
                
                samples_collected += 1
                
                if samples_collected >= num_samples:
                    break
        
        print(f"✅ Collected {samples_collected} samples")
        
        # Create visualization
        self.plot_prediction_grid(
            images_list, masks_list, predictions_list, metrics_list, save_path
        )
    
    def calculate_sample_metrics(self, mask: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single sample."""
        # Convert to torch tensors
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        
        # Calculate metrics
        dice = self.metrics.dice_coefficient(pred_tensor, mask_tensor).item()
        iou = self.metrics.iou_score(pred_tensor, mask_tensor).item()
        pixel_acc = self.metrics.pixel_accuracy(pred_tensor, mask_tensor).item()
        
        return {
            'dice': dice,
            'iou': iou,
            'pixel_accuracy': pixel_acc
        }
    
    def plot_prediction_grid(self, images: List[np.ndarray], masks: List[np.ndarray], 
                           predictions: List[np.ndarray], metrics: List[Dict], 
                           save_path: str):
        """Plot grid of predictions."""
        n_samples = len(images)
        n_cols = 4  # Image, Mask, Prediction, Overlay
        n_rows = n_samples
        
        fig = plt.figure(figsize=(16, 4.5 * n_samples))
        plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing
        
        for i in range(n_samples):
            # Original image
            ax1 = plt.subplot(n_rows, n_cols, i * n_cols + 1)
            ax1.imshow(images[i])
            ax1.set_title('Input Image' if i == 0 else '', fontweight='bold', fontsize=16)
            ax1.axis('off')
            
            # Ground truth mask
            ax2 = plt.subplot(n_rows, n_cols, i * n_cols + 2)
            ax2.imshow(masks[i], cmap='gray')
            ax2.set_title('Ground Truth' if i == 0 else '', fontweight='bold', fontsize=16)
            ax2.axis('off')
            
            # Prediction
            ax3 = plt.subplot(n_rows, n_cols, i * n_cols + 3)
            ax3.imshow(predictions[i], cmap='gray')
            ax3.set_title('Prediction' if i == 0 else '', fontweight='bold', fontsize=16)
            ax3.axis('off')
            
            # Overlay
            ax4 = plt.subplot(n_rows, n_cols, i * n_cols + 4)
            ax4.imshow(images[i])
            
            # Create colored overlays
            mask_overlay = np.zeros((*masks[i].shape, 4))
            pred_overlay = np.zeros((*predictions[i].shape, 4))
            
            # Ground truth in green
            mask_overlay[masks[i] > 0.5] = [0, 1, 0, 0.5]
            # Prediction in red
            pred_overlay[predictions[i] > 0.5] = [1, 0, 0, 0.5]
            
            ax4.imshow(mask_overlay)
            ax4.imshow(pred_overlay)
            ax4.set_title('Overlay (GT: Green, Pred: Red)' if i == 0 else '', fontweight='bold', fontsize=16)
            ax4.axis('off')
            
            # Add metrics text below the prediction and overlay
            ax3.text(0.5, -0.15, f"Dice: {metrics[i]['dice']:.3f}", transform=ax3.transAxes,
                    horizontalalignment='center', fontsize=11, fontweight='bold')
            
            ax4.text(0.5, -0.15, f"IoU: {metrics[i]['iou']:.3f}", transform=ax4.transAxes,
                    horizontalalignment='center', fontsize=11, fontweight='bold')
        
        plt.suptitle('Lesion Boundary Segmentation Results', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Prediction grid saved to {save_path}")
        
        # Print average metrics
        avg_dice = np.mean([m['dice'] for m in metrics])
        avg_iou = np.mean([m['iou'] for m in metrics])
        avg_pixel_acc = np.mean([m['pixel_accuracy'] for m in metrics])
        
        print(f"📈 Average Metrics:")
        print(f"   Dice: {avg_dice:.4f}")
        print(f"   IoU: {avg_iou:.4f}")
        print(f"   Pixel Accuracy: {avg_pixel_acc:.4f}")
    
    def create_single_comparison(self, sample_idx: int = None, save_path: str = "single_prediction.png"):
        """Create detailed comparison for a single sample."""
        # Get a sample
        val_loader = self.data_loaders['val']
        
        if sample_idx is None:
            sample_idx = random.randint(0, len(val_loader) - 1)
        
        # Get specific batch
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx == sample_idx // batch_data['image'].shape[0]:
                idx_in_batch = sample_idx % batch_data['image'].shape[0]
                # Extract images and masks from dictionary
                images = batch_data['image']
                masks = batch_data['mask']
                break
        
        # Extract sample
        image = images[idx_in_batch:idx_in_batch+1]
        mask = masks[idx_in_batch:idx_in_batch+1]
        
        # Generate prediction
        prediction = self.predict_batch(image)
        
        # Convert to numpy
        img = image[0].cpu().numpy().transpose(1, 2, 0)
        mask_np = mask[0].cpu().numpy().squeeze()
        pred_np = prediction[0].cpu().numpy().squeeze()
        
        # Normalize image
        img = (img - img.min()) / (img.max() - img.min())
        
        # Create detailed visualization
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, hspace=0.5, wspace=0.2)
        
        # Main comparison (top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask_np, cmap='gray')
        ax2.set_title('Ground Truth Mask', fontsize=20, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_np, cmap='hot')
        ax3.set_title('Prediction (Probability)', fontsize=20, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(pred_np > 0.5, cmap='gray')
        ax4.set_title('Binary Prediction', fontsize=20, fontweight='bold')
        ax4.axis('off')
        
        # Overlays (middle row)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(img)
        mask_overlay = np.zeros((*mask_np.shape, 4))
        mask_overlay[mask_np > 0.5] = [0, 1, 0, 0.6]
        ax5.imshow(mask_overlay)
        ax5.set_title('Image + Ground Truth', fontsize=20, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(img)
        pred_overlay = np.zeros((*pred_np.shape, 4))
        pred_overlay[pred_np > 0.5] = [1, 0, 0, 0.6]
        ax6.imshow(pred_overlay)
        ax6.set_title('Image + Prediction', fontsize=20, fontweight='bold')
        ax6.axis('off')
        
        # Error analysis
        ax7 = fig.add_subplot(gs[1, 2])
        pred_binary = (pred_np > 0.5).astype(np.float32)
        error_map = np.abs(mask_np - pred_binary)
        ax7.imshow(error_map, cmap='Reds')
        ax7.set_title('Error Map', fontsize=20, fontweight='bold')
        ax7.axis('off')
        
        # Combined overlay
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(img)
        ax8.imshow(mask_overlay)
        ax8.imshow(pred_overlay)
        ax8.set_title('Combined Overlay', fontsize=20, fontweight='bold')
        ax8.axis('off')
        
        # Add metrics below each image in the first row
        metrics = self.calculate_sample_metrics(mask_np, pred_binary)
        
        # Original Image - no metrics needed
        
        # Ground Truth Mask - add area information
        ax2.text(0.5, -0.1, f"Area: {np.sum(mask_np > 0.5)} pixels", 
                 transform=ax2.transAxes, fontsize=14, fontweight='bold',
                 horizontalalignment='center')
        
        # Prediction Probability - add confidence
        ax3.text(0.5, -0.1, f"Mean Confidence: {np.mean(pred_np):.4f}", 
                 transform=ax3.transAxes, fontsize=14, fontweight='bold',
                 horizontalalignment='center')
        
        # Binary Prediction - add dice and IoU
        ax4.text(0.5, -0.15, f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}", 
                 transform=ax4.transAxes, fontsize=14, fontweight='bold',
                 horizontalalignment='center')
        ax4.text(0.5, -0.05, f"Area: {np.sum(pred_binary > 0.5)} pixels", 
                 transform=ax4.transAxes, fontsize=14, fontweight='bold',
                 horizontalalignment='center')
        
        # Add metrics to combined overlay
        ax8.text(0.5, -0.1, f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}", 
                 transform=ax8.transAxes, fontsize=14, fontweight='bold',
                 horizontalalignment='center')
        
        plt.suptitle(f'Lesion Segmentation Analysis - Sample {sample_idx}', fontsize=20, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔍 Detailed comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate prediction visualizations for paper')
    parser.add_argument('--model_path', type=str,
                       default='runs/ckpts/checkpoints/best_checkpoint.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str,
                       default='configs/train_with_masks.json',
                       help='Path to training configuration')
    parser.add_argument('--output_dir', type=str,
                       default='paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--num_samples', type=int, default=6,
                       help='Number of samples for grid visualization')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"❌ Model checkpoint not found: {args.model_path}")
        return
    
    print("🎨 Creating prediction visualizations for paper...")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Output: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize visualizer
    visualizer = PredictionVisualizer(args.model_path, args.config_path)
    
    # Create visualizations
    print(f"\n📊 Creating prediction grid with {args.num_samples} samples...")
    grid_path = output_dir / "prediction_grid_detailed.png"
    visualizer.create_prediction_grid(args.num_samples, str(grid_path))
    
    print("🔍 Creating detailed single prediction analysis...")
    single_path = output_dir / "single_prediction_detailed.png"
    visualizer.create_single_comparison(save_path=str(single_path))
    
    print(f"\n✅ Prediction visualizations complete!")
    print(f"📂 Files saved to: {args.output_dir}")
    print("🎨 Ready for paper inclusion!")


if __name__ == "__main__":
    main()
