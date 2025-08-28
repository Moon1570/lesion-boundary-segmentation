#!/usr/bin/env python3
"""
Metrics for evaluating segmentation performance.

Implements comprehensive metrics for medical image segmentation:
- Intersection over Union (IoU)
- Dice Coefficient
- Pixel Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score
- Boundary-based metrics

Author: GitHub Copilot
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
import torch.nn.functional as F


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    
    Calculates various metrics for binary segmentation evaluation.
    All metrics are calculated per batch and can be aggregated.
    """
    
    def __init__(self, device: str = 'cuda', smooth: float = 1e-6):
        self.device = device
        self.smooth = smooth
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice coefficient (F1-score for segmentation).
        
        Args:
            pred: Predicted masks (B, 1, H, W) or (B, H, W)
            target: Ground truth masks (B, 1, H, W) or (B, H, W)
        
        Returns:
            Dice coefficient per sample in batch
        """
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice
    
    def iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Intersection over Union (IoU) score.
        
        Args:
            pred: Predicted masks (B, 1, H, W) or (B, H, W)
            target: Ground truth masks (B, 1, H, W) or (B, H, W)
        
        Returns:
            IoU score per sample in batch
        """
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou
    
    def pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate pixel-wise accuracy."""
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        correct = (pred == target).float()
        accuracy = correct.mean(dim=1)
        return accuracy
    
    def sensitivity_recall(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate sensitivity (recall, true positive rate).
        Measures ability to correctly identify positive pixels.
        """
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        true_positives = (pred * target).sum(dim=1)
        actual_positives = target.sum(dim=1)
        
        sensitivity = (true_positives + self.smooth) / (actual_positives + self.smooth)
        return sensitivity
    
    def specificity(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate specificity (true negative rate).
        Measures ability to correctly identify negative pixels.
        """
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        true_negatives = ((1 - pred) * (1 - target)).sum(dim=1)
        actual_negatives = (1 - target).sum(dim=1)
        
        specificity = (true_negatives + self.smooth) / (actual_negatives + self.smooth)
        return specificity
    
    def precision(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate precision (positive predictive value).
        Measures accuracy of positive predictions.
        """
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        true_positives = (pred * target).sum(dim=1)
        predicted_positives = pred.sum(dim=1)
        
        precision = (true_positives + self.smooth) / (predicted_positives + self.smooth)
        return precision
    
    def f1_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        precision = self.precision(pred, target)
        recall = self.sensitivity_recall(pred, target)
        
        f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
        return f1
    
    def boundary_iou(self, pred: torch.Tensor, target: torch.Tensor, 
                    boundary_width: int = 2) -> torch.Tensor:
        """
        Calculate IoU score specifically for boundary regions.
        
        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            boundary_width: Width of boundary region to evaluate
        
        Returns:
            Boundary IoU score per sample
        """
        batch_size = pred.size(0)
        boundary_ious = []
        
        for i in range(batch_size):
            pred_i = pred[i, 0].cpu().numpy()
            target_i = target[i, 0].cpu().numpy()
            
            # Extract boundary regions
            pred_boundary = self._extract_boundary(pred_i, boundary_width)
            target_boundary = self._extract_boundary(target_i, boundary_width)
            
            # Calculate IoU for boundary
            intersection = np.logical_and(pred_boundary, target_boundary).sum()
            union = np.logical_or(pred_boundary, target_boundary).sum()
            
            if union == 0:
                boundary_iou = 1.0  # Perfect match when no boundary exists
            else:
                boundary_iou = intersection / union
            
            boundary_ious.append(boundary_iou)
        
        return torch.tensor(boundary_ious, device=self.device)
    
    def _extract_boundary(self, mask: np.ndarray, width: int) -> np.ndarray:
        """Extract boundary region from binary mask."""
        from scipy import ndimage
        
        # Dilate and erode to get boundary
        dilated = ndimage.binary_dilation(mask, iterations=width)
        eroded = ndimage.binary_erosion(mask, iterations=width)
        boundary = dilated ^ eroded
        
        return boundary
    
    def hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Hausdorff distance between boundaries.
        
        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
        
        Returns:
            Hausdorff distance per sample
        """
        batch_size = pred.size(0)
        hausdorff_distances = []
        
        for i in range(batch_size):
            pred_i = pred[i, 0].cpu().numpy()
            target_i = target[i, 0].cpu().numpy()
            
            try:
                from scipy.spatial.distance import directed_hausdorff
                
                # Get boundary points
                pred_points = np.argwhere(self._extract_boundary(pred_i, 1))
                target_points = np.argwhere(self._extract_boundary(target_i, 1))
                
                if len(pred_points) == 0 or len(target_points) == 0:
                    hd = 0.0
                else:
                    hd1 = directed_hausdorff(pred_points, target_points)[0]
                    hd2 = directed_hausdorff(target_points, pred_points)[0]
                    hd = max(hd1, hd2)
                
                hausdorff_distances.append(hd)
                
            except ImportError:
                # Fallback if scipy not available
                hausdorff_distances.append(0.0)
        
        return torch.tensor(hausdorff_distances, device=self.device)
    
    def calculate_batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics for a batch.
        
        Args:
            pred: Predicted masks (B, 1, H, W) - binary values {0, 1}
            target: Ground truth masks (B, 1, H, W) - binary values {0, 1}
        
        Returns:
            Dictionary of metric names and their batch-averaged values
        """
        # Ensure binary values
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        # Calculate all metrics
        metrics = {}
        
        # Core segmentation metrics
        metrics['dice'] = self.dice_coefficient(pred, target).mean().item()
        metrics['iou'] = self.iou_score(pred, target).mean().item()
        metrics['pixel_accuracy'] = self.pixel_accuracy(pred, target).mean().item()
        
        # Classification metrics
        metrics['sensitivity'] = self.sensitivity_recall(pred, target).mean().item()
        metrics['specificity'] = self.specificity(pred, target).mean().item()
        metrics['precision'] = self.precision(pred, target).mean().item()
        metrics['f1'] = self.f1_score(pred, target).mean().item()
        
        # Boundary metrics (computationally expensive, so optional)
        try:
            metrics['boundary_iou'] = self.boundary_iou(pred, target).mean().item()
            metrics['hausdorff'] = self.hausdorff_distance(pred, target).mean().item()
        except Exception:
            # Skip boundary metrics if they fail
            pass
        
        return metrics
    
    def calculate_dataset_metrics(self, all_predictions: torch.Tensor, 
                                all_targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate metrics for entire dataset.
        
        Args:
            all_predictions: All predictions concatenated (N, 1, H, W)
            all_targets: All targets concatenated (N, 1, H, W)
        
        Returns:
            Dictionary of metric names and their dataset-wide values
        """
        # Process in smaller batches to avoid memory issues
        batch_size = 32
        num_samples = all_predictions.size(0)
        
        all_metrics = {
            'dice': [], 'iou': [], 'pixel_accuracy': [],
            'sensitivity': [], 'specificity': [], 'precision': [], 'f1': []
        }
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_pred = all_predictions[i:end_idx]
            batch_target = all_targets[i:end_idx]
            
            # Calculate metrics for this batch
            batch_dice = self.dice_coefficient(batch_pred, batch_target)
            batch_iou = self.iou_score(batch_pred, batch_target)
            batch_acc = self.pixel_accuracy(batch_pred, batch_target)
            batch_sens = self.sensitivity_recall(batch_pred, batch_target)
            batch_spec = self.specificity(batch_pred, batch_target)
            batch_prec = self.precision(batch_pred, batch_target)
            batch_f1 = self.f1_score(batch_pred, batch_target)
            
            # Accumulate
            all_metrics['dice'].extend(batch_dice.cpu().numpy())
            all_metrics['iou'].extend(batch_iou.cpu().numpy())
            all_metrics['pixel_accuracy'].extend(batch_acc.cpu().numpy())
            all_metrics['sensitivity'].extend(batch_sens.cpu().numpy())
            all_metrics['specificity'].extend(batch_spec.cpu().numpy())
            all_metrics['precision'].extend(batch_prec.cpu().numpy())
            all_metrics['f1'].extend(batch_f1.cpu().numpy())
        
        # Calculate final statistics
        final_metrics = {}
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            final_metrics[f'{metric_name}_mean'] = np.mean(values)
            final_metrics[f'{metric_name}_std'] = np.std(values)
            final_metrics[f'{metric_name}_median'] = np.median(values)
        
        return final_metrics


def test_metrics():
    """Test the metrics implementation."""
    print("🧪 Testing Segmentation Metrics")
    
    # Create sample data
    batch_size, height, width = 4, 128, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Perfect prediction (should give perfect scores)
    target = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32, device=device)
    pred_perfect = target.clone()
    
    # Random prediction
    pred_random = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32, device=device)
    
    # Initialize metrics
    metrics = SegmentationMetrics(device=device)
    
    # Test perfect prediction
    perfect_metrics = metrics.calculate_batch_metrics(pred_perfect, target)
    print("\n📊 Perfect Prediction Metrics:")
    for metric, value in perfect_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Test random prediction
    random_metrics = metrics.calculate_batch_metrics(pred_random, target)
    print("\n🎲 Random Prediction Metrics:")
    for metric, value in random_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print("\n✅ Metrics test completed!")


if __name__ == "__main__":
    test_metrics()
