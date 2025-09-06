"""
Models package for ISIC2018 lesion boundary segmentation.

Contains:
- UNet: Custom U-Net implementation
- UNetMonai: MONAI U-Net wrapper
- AttentionUNet: U-Net with attention gates
- UNetPlusPlus: Nested U-Net (U-Net++)
- DuaSkinSeg: Dual encoder architecture (MobileNetV2 + ViT)
- MambaUNet: Mamba-based U-Net for long-range dependencies
- QuantizedMambaUNet: Quantized Mamba U-Net for 8GB GPU deployment
- Advanced loss functions for segmentation
- Model utilities and helpers
"""

from .unet import UNet, UNetMonai, create_unet_small
from .enhanced_unet import AttentionUNet, UNetPlusPlus
from .duaskinseg import DuaSkinSeg, create_duaskinseg
from .lightweight_duaskinseg import LightweightDuaSkinSeg, create_lightweight_duaskinseg
from .mamba_unet import MambaUNet, LightweightMambaUNet
from .quantized_mamba_unet import QuantizedMambaUNet, create_quantized_mamba_unet
from .advanced_losses import (
    FocalLoss, TverskyLoss, FocalTverskyLoss, 
    IoULoss, ComboLoss, AdvancedCombinedLoss
)

__all__ = [
    'UNet',
    'UNetMonai', 
    'AttentionUNet',
    'UNetPlusPlus',
    'DuaSkinSeg',
    'LightweightDuaSkinSeg',
    'MambaUNet',
    'LightweightMambaUNet',
    'QuantizedMambaUNet',
    'create_unet_small',
    'create_duaskinseg',
    'create_lightweight_duaskinseg',
    'create_quantized_mamba_unet',
    'FocalLoss',
    'TverskyLoss', 
    'FocalTverskyLoss',
    'IoULoss',
    'ComboLoss',
    'AdvancedCombinedLoss'
]
