#!/usr/bin/env python3
"""
UNetMamba implementation inspired by the paper "UNetMamba: An Efficient UNet-Like Mamba 
for Semantic Segmentation of High-Resolution Remote Sensing Images"
https://arxiv.org/abs/2408.11545

This implementation focuses on practical memory efficiency for 8GB GPUs.
"""

import os
# Silence TensorFlow oneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("mamba_ssm not available, using efficient linear approximation")
    MAMBA_AVAILABLE = False


class EfficientMamba(nn.Module):
    """Efficient Mamba approximation for when mamba_ssm is unavailable."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Simplified state space model components
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        
        # Project and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # Conv1d requires (B, C, L)
        x = x.transpose(-1, -2)  # (B, d_inner, L)
        x = self.conv1d(x)
        x = x.transpose(-1, -2)  # (B, L, d_inner)
        
        # Apply activation and gating
        x = self.act(x) * self.act(z)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class MambaSegmentationDecoder(nn.Module):
    """
    Mamba Segmentation Decoder (MSD) - core component of UNetMamba.
    Applies Mamba efficiently only to decoder features.
    """
    
    def __init__(self, dim: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=dim, d_state=d_state, expand=expand)
        else:
            self.mamba = EfficientMamba(d_model=dim, d_state=d_state, expand=expand)
        
        # Skip connection with learnable scaling
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Apply normalization and Mamba
        x_norm = self.norm(x_seq)
        x_mamba = self.mamba(x_norm)
        
        # Residual connection with learnable scaling
        x_out = x_seq + self.alpha * x_mamba
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        
        return x_out


class ConvBlock(nn.Module):
    """Standard U-Net convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetMamba(nn.Module):
    """
    UNetMamba: Efficient U-Net with Mamba in decoder for medical image segmentation.
    
    Key features:
    - Regular CNN encoder for efficiency
    - Mamba-enhanced decoder for long-range dependencies
    - Memory-efficient design for 8GB GPUs
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        encoder_depths: List[int] = [2, 2, 2, 2],
        d_state: int = 16,
        expand: int = 2,
        use_mamba_levels: List[int] = [1, 2, 3],  # Which decoder levels use Mamba
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_mamba_levels = use_mamba_levels
        
        # Calculate channel sizes
        channels = [base_channels * (2 ** i) for i in range(len(encoder_depths))]
        
        # Encoder (standard CNN for efficiency)
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for i, depth in enumerate(encoder_depths):
            layers = []
            for j in range(depth):
                layers.append(ConvBlock(in_ch if j == 0 else channels[i], channels[i]))
                in_ch = channels[i] if j == 0 else channels[i]
            self.encoder_blocks.append(nn.Sequential(*layers))
            
            if i < len(encoder_depths) - 1:  # No pooling after last encoder block
                self.pools.append(nn.MaxPool2d(2))
        
        # Decoder with Mamba enhancement
        self.decoder_blocks = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(len(encoder_depths) - 1):
            # Decoder block index (0 = deepest, 3 = shallowest)
            decoder_idx = len(encoder_depths) - 2 - i
            
            in_ch = channels[decoder_idx + 1]
            skip_ch = channels[decoder_idx]
            out_ch = channels[decoder_idx]
            
            # Upsample
            self.upsamples.append(
                nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            )
            
            # Decoder conv block
            self.decoder_blocks.append(
                ConvBlock(in_ch // 2 + skip_ch, out_ch)
            )
            
            # Mamba block (only for specified levels)
            if decoder_idx in self.use_mamba_levels:
                self.mamba_blocks.append(
                    MambaSegmentationDecoder(out_ch, d_state=d_state, expand=expand)
                )
            else:
                self.mamba_blocks.append(nn.Identity())
        
        # Final output layer
        self.final_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        encoder_features = []
        
        for i, (encoder_block, pool) in enumerate(zip(self.encoder_blocks, self.pools + [nn.Identity()])):
            x = encoder_block(x)
            encoder_features.append(x)
            
            if i < len(self.pools):  # Don't pool after last encoder
                x = pool(x)
        
        # Decoder path
        for i, (upsample, decoder_block, mamba_block) in enumerate(
            zip(self.upsamples, self.decoder_blocks, self.mamba_blocks)
        ):
            # Upsample
            x = upsample(x)
            
            # Skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip_features = encoder_features[skip_idx]
            
            # Handle size mismatch
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate and decode
            x = torch.cat([x, skip_features], dim=1)
            x = decoder_block(x)
            
            # Apply Mamba if specified for this level
            x = mamba_block(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x


class LightweightUNetMamba(nn.Module):
    """Ultra-lightweight UNetMamba for 8GB GPU training."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        
        self.unet_mamba = UNetMamba(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=32,  # Smaller base channels
            encoder_depths=[1, 1, 1, 1],  # Single layers for efficiency
            d_state=8,  # Smaller state dimension
            expand=1.5,  # Modest expansion
            use_mamba_levels=[1, 2],  # Only mid-level features
        )
    
    def forward(self, x):
        return self.unet_mamba(x)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the implementation
if __name__ == "__main__":
    # Test lightweight version
    model = LightweightUNetMamba(in_channels=3, num_classes=1)
    print(f"LightweightUNetMamba parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    
    print("Testing forward pass...")
    with torch.no_grad():
        out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    
    print("✅ UNetMamba test passed!")
