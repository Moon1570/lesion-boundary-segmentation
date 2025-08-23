#!/usr/bin/env python3
"""
Loss functions for lesion boundary segmentation.

Implements various loss functions suitable for medical image segmentation:
- Binary Cross Entropy
- Dice Loss  
- Focal Loss
- Boundary Loss
- Combined losses

Author: GitHub Copilot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice coefficient measures overlap between predicted and ground truth masks.
    Loss = 1 - Dice coefficient
    
    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        sigmoid: Whether to apply sigmoid to predictions (default: True)
    """
    
    def __init__(self, smooth: float = 1.0, sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses on hard examples.
    Useful when background pixels dominate the image.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        sigmoid: Whether to apply sigmoid to predictions (default: True)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, sigmoid: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = sigmoid
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            predictions = torch.sigmoid(predictions)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Compute p_t
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better edge detection.
    
    Emphasizes pixels near the boundary to improve segmentation edges.
    Uses distance transform to weight pixels based on distance to boundary.
    
    Args:
        theta0: Distance threshold (default: 3.0)
        theta: Distance scaling factor (default: 5.0)
    """
    
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def _compute_sdf(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance function (approximation).
        Positive inside mask, negative outside.
        """
        # Convert to numpy for distance transform
        mask_np = mask.detach().cpu().numpy()
        
        try:
            from scipy.ndimage import distance_transform_edt
            
            sdf_list = []
            for i in range(mask_np.shape[0]):  # Batch dimension
                for j in range(mask_np.shape[1]):  # Channel dimension
                    binary_mask = mask_np[i, j] > 0.5
                    
                    # Distance transform for foreground and background
                    if binary_mask.any():
                        pos_dist = distance_transform_edt(binary_mask)
                        neg_dist = distance_transform_edt(1 - binary_mask)
                        sdf = pos_dist - neg_dist
                    else:
                        sdf = -np.ones_like(binary_mask, dtype=np.float32)
                    
                    sdf_list.append(sdf)
            
            # Reshape back to original shape
            sdf_array = np.stack(sdf_list).reshape(mask_np.shape)
            return torch.from_numpy(sdf_array).to(mask.device)
        
        except ImportError:
            # Fallback: simple edge detection
            print("⚠️ scipy not available, using simple edge detection for boundary loss")
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
            edges = F.conv2d(mask, kernel, padding=1)
            return torch.abs(edges)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Compute signed distance function for targets
        sdf_targets = self._compute_sdf(targets)
        
        # Boundary weight: high near boundaries, low far from boundaries
        boundary_weights = torch.exp(-torch.abs(sdf_targets) / self.theta)
        boundary_weights = torch.where(torch.abs(sdf_targets) <= self.theta0, 
                                     torch.ones_like(boundary_weights), 
                                     boundary_weights)
        
        # Weighted binary cross entropy with logits (safe for autocast)
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        weighted_loss = boundary_weights * bce_loss
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for robust segmentation.
    
    Combines multiple loss functions for better performance:
    - BCE for pixel-wise accuracy
    - Dice for region overlap
    - Focal for class imbalance (optional)
    - Boundary for edge accuracy (optional)
    
    Args:
        bce_weight: Weight for BCE loss (default: 1.0)
        dice_weight: Weight for Dice loss (default: 1.0)
        focal_weight: Weight for Focal loss (default: 0.0)
        boundary_weight: Weight for Boundary loss (default: 0.0)
    """
    
    def __init__(
        self, 
        bce_weight: float = 1.0,
        dice_weight: float = 1.0, 
        focal_weight: float = 0.0,
        boundary_weight: float = 0.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        # Initialize loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, sigmoid=True)
        
        if boundary_weight > 0:
            self.boundary_loss = BoundaryLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        # BCE Loss
        if self.bce_weight > 0:
            bce = self.bce_loss(predictions, targets)
            total_loss += self.bce_weight * bce
        
        # Dice Loss
        if self.dice_weight > 0:
            dice = self.dice_loss(predictions, targets)
            total_loss += self.dice_weight * dice
        
        # Focal Loss
        if self.focal_weight > 0:
            focal = self.focal_loss(predictions, targets)
            total_loss += self.focal_weight * focal
        
        # Boundary Loss
        if self.boundary_weight > 0:
            boundary = self.boundary_loss(predictions, targets)
            total_loss += self.boundary_weight * boundary
        
        return total_loss


def create_loss_function(loss_type: str = "bce_dice", **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function
            - "bce": Binary Cross Entropy
            - "dice": Dice Loss
            - "focal": Focal Loss
            - "bce_dice": Combined BCE + Dice (default)
            - "all": All losses combined
        **kwargs: Additional arguments for loss functions
    
    Returns:
        Configured loss function
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    
    elif loss_type == "bce_dice":
        return CombinedLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.0,
            boundary_weight=0.0,
            **kwargs
        )
    
    elif loss_type == "all":
        return CombinedLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5,
            boundary_weight=0.5,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_loss_functions():
    """Test all loss functions with sample data."""
    print("🧪 Testing Loss Functions")
    print("=" * 50)
    
    # Create sample data
    batch_size, channels, height, width = 2, 1, 32, 32
    predictions = torch.randn(batch_size, channels, height, width)
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # Test all loss functions
    loss_functions = {
        "BCE": nn.BCEWithLogitsLoss(),
        "Dice": DiceLoss(),
        "Focal": FocalLoss(),
        "BCE + Dice": create_loss_function("bce_dice"),
        "All Combined": create_loss_function("all")
    }
    
    for name, loss_fn in loss_functions.items():
        try:
            loss_value = loss_fn(predictions, targets)
            print(f"✅ {name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"❌ {name}: {e}")


if __name__ == "__main__":
    test_loss_functions()
