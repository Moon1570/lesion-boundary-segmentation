#!/usr/bin/env python3
"""
Training visualization utilities.

Provides visualization capabilities for training monitoring:
- Prediction overlays
- Training progress plots
- Loss curves
- Metric tracking
- Sample predictions during training

Author: GitHub Copilot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image


class TrainingVisualizer:
    """
    Handles visualization during training.
    
    Creates visualizations for:
    - Prediction overlays during validation
    - Training progress monitoring
    - Loss and metric curves
    - Sample predictions at different epochs
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.predictions_dir = self.output_dir / 'predictions'
        self.curves_dir = self.output_dir / 'curves'
        self.samples_dir = self.output_dir / 'samples'
        
        for dir_path in [self.predictions_dir, self.curves_dir, self.samples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def denormalize_image(self, image: torch.Tensor, 
                         mean: List[float] = [0.485, 0.456, 0.406],
                         std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Denormalize image tensor for visualization.
        
        Args:
            image: Normalized image tensor (C, H, W)
            mean: Normalization mean values
            std: Normalization std values
        
        Returns:
            Denormalized image as numpy array (H, W, C)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Denormalize
        for i in range(3):
            image[i] = image[i] * std[i] + mean[i]
        
        # Clip to valid range and convert to uint8
        image = np.clip(image, 0, 1)
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                      prediction: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Create overlay visualization of image, ground truth, and prediction.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Ground truth mask (H, W)
            prediction: Predicted mask (H, W)
            alpha: Transparency for overlays
        
        Returns:
            Overlay image (H, W, 3)
        """
        overlay = image.copy()
        
        # Convert masks to binary
        mask_binary = (mask > 0.5).astype(np.uint8)
        pred_binary = (prediction > 0.5).astype(np.uint8)
        
        # Create colored overlays
        # Ground truth in green
        gt_color = np.array([0, 255, 0])  # Green
        gt_overlay = np.zeros_like(image)
        gt_overlay[mask_binary == 1] = gt_color
        
        # Prediction in red
        pred_color = np.array([255, 0, 0])  # Red
        pred_overlay = np.zeros_like(image)
        pred_overlay[pred_binary == 1] = pred_color
        
        # Overlap in yellow
        overlap = np.logical_and(mask_binary, pred_binary)
        overlap_color = np.array([255, 255, 0])  # Yellow
        overlap_overlay = np.zeros_like(image)
        overlap_overlay[overlap] = overlap_color
        
        # Combine overlays
        overlay = image.astype(np.float32)
        overlay = overlay * (1 - alpha) + gt_overlay * alpha
        overlay = overlay * (1 - alpha) + pred_overlay * alpha
        overlay = overlay * (1 - alpha) + overlap_overlay * alpha
        
        return np.clip(overlay, 0, 255).astype(np.uint8)
    
    def save_predictions(self, images: torch.Tensor, masks: torch.Tensor,
                        predictions: torch.Tensor, epoch: int, split: str = 'val',
                        max_samples: int = 8):
        """
        Save prediction visualizations.
        
        Args:
            images: Batch of images (B, C, H, W)
            masks: Batch of ground truth masks (B, 1, H, W)
            predictions: Batch of predictions (B, 1, H, W)
            epoch: Current epoch number
            split: Data split name
            max_samples: Maximum number of samples to visualize
        """
        batch_size = min(images.size(0), max_samples)
        
        # Apply sigmoid to predictions if needed
        if predictions.min() < 0 or predictions.max() > 1:
            pred_probs = torch.sigmoid(predictions)
        else:
            pred_probs = predictions
        
        # Create figure
        fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # Get data for this sample
            image = images[i]  # (C, H, W)
            mask = masks[i, 0]  # (H, W)
            pred_prob = pred_probs[i, 0]  # (H, W)
            pred_binary = (pred_prob > 0.5).float()  # (H, W)
            
            # Convert to numpy
            image_np = self.denormalize_image(image)
            mask_np = mask.cpu().numpy()
            pred_prob_np = pred_prob.cpu().numpy()
            pred_binary_np = pred_binary.cpu().numpy()
            
            # Plot original image
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Plot ground truth
            axes[i, 1].imshow(image_np)
            axes[i, 1].imshow(mask_np, alpha=0.5, cmap='Greens')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Plot prediction probability
            axes[i, 2].imshow(image_np)
            axes[i, 2].imshow(pred_prob_np, alpha=0.5, cmap='Reds', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Prediction (prob)')
            axes[i, 2].axis('off')
            
            # Plot overlay
            overlay = self.create_overlay(image_np, mask_np, pred_binary_np)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (GT: Green, Pred: Red)')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.predictions_dir / f'epoch_{epoch:03d}_{split}_predictions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved predictions: {save_path}")
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                           train_metrics: Dict[str, List[float]], 
                           val_metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None):
        """
        Plot training and validation curves.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch  
            train_metrics: Training metrics per epoch
            val_metrics: Validation metrics per epoch
            save_path: Path to save the plot
        """
        epochs = range(1, len(train_losses) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice coefficient
        if 'dice' in train_metrics and 'dice' in val_metrics:
            axes[0, 1].plot(epochs, train_metrics['dice'], 'b-', label='Train Dice', linewidth=2)
            axes[0, 1].plot(epochs, val_metrics['dice'], 'r-', label='Val Dice', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Coefficient')
            axes[0, 1].set_title('Dice Coefficient')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # IoU score
        if 'iou' in train_metrics and 'iou' in val_metrics:
            axes[1, 0].plot(epochs, train_metrics['iou'], 'b-', label='Train IoU', linewidth=2)
            axes[1, 0].plot(epochs, val_metrics['iou'], 'r-', label='Val IoU', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU Score')
            axes[1, 0].set_title('IoU Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Additional metrics (F1, Precision, Recall)
        metric_names = ['f1', 'precision', 'sensitivity']
        colors = ['g-', 'm-', 'c-']
        
        for metric, color in zip(metric_names, colors):
            if metric in val_metrics:
                axes[1, 1].plot(epochs, val_metrics[metric], color, 
                               label=f'Val {metric.capitalize()}', linewidth=2)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Additional Validation Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.curves_dir / 'training_curves.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved training curves: {save_path}")
    
    def plot_metric_distribution(self, metrics: Dict[str, np.ndarray], 
                               title: str = "Metric Distribution",
                               save_path: Optional[str] = None):
        """
        Plot distribution of metrics across samples.
        
        Args:
            metrics: Dictionary of metric arrays
            title: Plot title
            save_path: Path to save the plot
        """
        num_metrics = len(metrics)
        if num_metrics == 0:
            return
        
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Plot histogram
            ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric_name.capitalize())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric_name.capitalize()} Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}±{std_val:.3f}')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(num_metrics, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.curves_dir / 'metric_distribution.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved metric distribution: {save_path}")
    
    def create_training_summary(self, config: Dict, final_metrics: Dict[str, float],
                              training_time: float, save_path: Optional[str] = None):
        """
        Create a comprehensive training summary visualization.
        
        Args:
            config: Training configuration
            final_metrics: Final validation metrics
            training_time: Total training time in seconds
            save_path: Path to save the summary
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Training configuration
        config_text = f"""
Training Configuration:
Model: {config.get('model', {}).get('name', 'Unknown')}
Loss: {config.get('loss', {}).get('name', 'Unknown')}
Optimizer: {config.get('optimizer', {}).get('name', 'Unknown')}
Learning Rate: {config.get('optimizer', {}).get('lr', 'Unknown')}
Batch Size: {config.get('data', {}).get('batch_size', 'Unknown')}
Epochs: {config.get('training', {}).get('epochs', 'Unknown')}
Mixed Precision: {config.get('training', {}).get('use_amp', 'Unknown')}
Training Time: {training_time/3600:.2f} hours
        """
        
        axes[0, 0].text(0.05, 0.95, config_text, transform=axes[0, 0].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[0, 0].set_title('Training Configuration')
        axes[0, 0].axis('off')
        
        # Final metrics bar plot
        metric_names = list(final_metrics.keys())
        metric_values = list(final_metrics.values())
        
        axes[0, 1].barh(metric_names, metric_values, color='skyblue', edgecolor='navy')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_title('Final Validation Metrics')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Model architecture summary (if available)
        model_info = config.get('model', {})
        model_text = f"""
Model Architecture Details:
In Channels: {model_info.get('in_channels', 3)}
Out Channels: {model_info.get('out_channels', 1)}
Encoder Channels: {model_info.get('encoder_channels', 'Default')}
Bottleneck: {model_info.get('bottleneck_channels', 'Default')}
        """
        
        axes[1, 0].text(0.05, 0.95, model_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 0].set_title('Model Architecture')
        axes[1, 0].axis('off')
        
        # Performance summary
        perf_text = f"""
Performance Summary:
Best Validation Dice: {final_metrics.get('dice', 0):.4f}
Best Validation IoU: {final_metrics.get('iou', 0):.4f}
Pixel Accuracy: {final_metrics.get('pixel_accuracy', 0):.4f}
Sensitivity: {final_metrics.get('sensitivity', 0):.4f}
Specificity: {final_metrics.get('specificity', 0):.4f}
Precision: {final_metrics.get('precision', 0):.4f}
        """
        
        axes[1, 1].text(0.05, 0.95, perf_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'training_summary.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📋 Saved training summary: {save_path}")


def test_visualizer():
    """Test the visualization functionality."""
    print("🧪 Testing Training Visualizer")
    
    # Create test data
    batch_size, height, width = 4, 128, 128
    
    # Random images and masks
    images = torch.randn(batch_size, 3, height, width)
    masks = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32)
    predictions = torch.randn(batch_size, 1, height, width)  # Logits
    
    # Initialize visualizer
    visualizer = TrainingVisualizer('test_output')
    
    # Test prediction saving
    visualizer.save_predictions(images, masks, predictions, epoch=1, split='test')
    
    # Test training curves
    train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    val_losses = [1.1, 0.9, 0.7, 0.6, 0.5]
    train_metrics = {'dice': [0.5, 0.6, 0.7, 0.75, 0.8]}
    val_metrics = {'dice': [0.45, 0.55, 0.65, 0.7, 0.75]}
    
    visualizer.plot_training_curves(train_losses, val_losses, train_metrics, val_metrics)
    
    print("✅ Visualizer test completed!")


if __name__ == "__main__":
    test_visualizer()
