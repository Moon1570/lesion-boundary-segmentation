#!/usr/bin/env python3
"""
Mamba-based U-Net for medical image segmentation.

This implementation combines the power of State Space Models (Mamba) with U-Net architecture
for improved long-range dependency modeling in medical image segmentation.

Key features:
- Mamba blocks for efficient sequence modeling
- Bidirectional scanning for 2D images
- U-Net architecture with skip connections
- Optimized for medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat
import numpy as np

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("mamba_ssm not available, using simplified implementation")
    MAMBA_AVAILABLE = False


class SimplifiedMamba(nn.Module):
    """Simplified Mamba implementation when mamba_ssm is not available."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        
        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :L]  # Truncate to original length
        x = rearrange(x, 'b d l -> b l d')
        x = self.act(x)
        
        # Simplified SSM (just attention-like mechanism)
        ssm_state = self.x_proj(x)  # (B, L, 2*d_state)
        dt = self.dt_proj(x)  # (B, L, d_inner)
        
        # Simple state update (simplified)
        x = x * torch.sigmoid(dt)
        
        # Apply gate
        x = x * self.act(z)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class MambaBlock(nn.Module):
    """Mamba block with normalization and residual connection."""
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = SimplifiedMamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
    
    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Reshape to sequence
        x_seq = rearrange(x, 'b h w c -> b (h w) c')
        
        # Apply normalization and mamba
        x_norm = self.norm(x_seq)
        x_mamba = self.mamba(x_norm)
        
        # Residual connection
        x_out = x_seq + x_mamba
        
        # Reshape back
        x_out = rearrange(x_out, 'b (h w) c -> b h w c', h=H, w=W)
        
        return x_out


class BiDirectionalMamba(nn.Module):
    """Bidirectional Mamba for 2D images with multiple scanning directions."""
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        
        # Four scanning directions
        self.mamba_h1 = MambaBlock(dim, d_state, d_conv, expand)  # Left to right
        self.mamba_h2 = MambaBlock(dim, d_state, d_conv, expand)  # Right to left
        self.mamba_v1 = MambaBlock(dim, d_state, d_conv, expand)  # Top to bottom
        self.mamba_v2 = MambaBlock(dim, d_state, d_conv, expand)  # Bottom to top
        
        # Fusion layer
        self.fusion = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Convert to (B, H, W, C) for Mamba processing
        x_hwc = rearrange(x, 'b c h w -> b h w c')
        
        # Horizontal scanning (left-to-right)
        x_h1 = self.mamba_h1(x_hwc)
        
        # Horizontal scanning (right-to-left)
        x_h2_flipped = torch.flip(x_hwc, dims=[2])  # Flip width
        x_h2_flipped = self.mamba_h2(x_h2_flipped)
        x_h2 = torch.flip(x_h2_flipped, dims=[2])  # Flip back
        
        # Vertical scanning (top-to-bottom)
        x_v1_transposed = rearrange(x_hwc, 'b h w c -> b w h c')
        x_v1_transposed = self.mamba_v1(x_v1_transposed)
        x_v1 = rearrange(x_v1_transposed, 'b w h c -> b h w c')
        
        # Vertical scanning (bottom-to-top)
        x_v2_flipped = torch.flip(x_hwc, dims=[1])  # Flip height
        x_v2_transposed = rearrange(x_v2_flipped, 'b h w c -> b w h c')
        x_v2_transposed = self.mamba_v2(x_v2_transposed)
        x_v2_transposed = rearrange(x_v2_transposed, 'b w h c -> b h w c')
        x_v2 = torch.flip(x_v2_transposed, dims=[1])  # Flip back
        
        # Concatenate all directions
        x_concat = torch.cat([x_h1, x_h2, x_v1, x_v2], dim=-1)  # (B, H, W, 4*C)
        
        # Convert back to (B, C, H, W) and fuse
        x_concat = rearrange(x_concat, 'b h w c -> b c h w')
        x_fused = self.fusion(x_concat)
        
        # Residual connection
        x_out = x + x_fused
        
        return x_out


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class MambaUNet(nn.Module):
    """U-Net with Mamba blocks for improved long-range dependency modeling."""
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        channels: list = [64, 128, 256, 512, 1024],
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba_encoder: bool = True,
        use_mamba_decoder: bool = True,
        use_mamba_bottleneck: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = channels
        self.use_mamba_encoder = use_mamba_encoder
        self.use_mamba_decoder = use_mamba_decoder
        self.use_mamba_bottleneck = use_mamba_bottleneck
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.mamba_encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        in_ch = n_channels
        for ch in channels[:-1]:
            self.encoder_blocks.append(ConvBlock(in_ch, ch))
            if use_mamba_encoder:
                self.mamba_encoder_blocks.append(BiDirectionalMamba(ch, d_state, d_conv, expand))
            in_ch = ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels[-2], channels[-1])
        if use_mamba_bottleneck:
            self.mamba_bottleneck = BiDirectionalMamba(channels[-1], d_state, d_conv, expand)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.mamba_decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch = channels[-(i+1)]
            skip_ch = channels[-(i+2)]
            out_ch = channels[-(i+2)]
            
            self.upsamples.append(
                nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                ConvBlock(in_ch // 2 + skip_ch, out_ch)
            )
            if use_mamba_decoder:
                self.mamba_decoder_blocks.append(
                    BiDirectionalMamba(out_ch, d_state, d_conv, expand)
                )
        
        # Final output layer
        self.final_conv = nn.Conv2d(channels[0], n_classes, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
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
        
        for i, (conv_block, mamba_block) in enumerate(
            zip(self.encoder_blocks, 
                self.mamba_encoder_blocks if self.use_mamba_encoder else [None] * len(self.encoder_blocks))
        ):
            x = conv_block(x)
            
            if self.use_mamba_encoder and mamba_block is not None:
                x = mamba_block(x)
            
            encoder_features.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_mamba_bottleneck:
            x = self.mamba_bottleneck(x)
        
        # Decoder path
        for i, (upsample, conv_block, mamba_block) in enumerate(
            zip(self.upsamples, 
                self.decoder_blocks,
                self.mamba_decoder_blocks if self.use_mamba_decoder else [None] * len(self.decoder_blocks))
        ):
            x = upsample(x)
            
            # Skip connection
            skip_features = encoder_features[-(i+1)]
            
            # Handle size mismatch
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip_features], dim=1)
            x = conv_block(x)
            
            if self.use_mamba_decoder and mamba_block is not None:
                x = mamba_block(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x


class LightweightMambaUNet(nn.Module):
    """Lightweight version of MambaUNet for memory efficiency."""
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        base_channels: int = 8,  # Very small channels for 8GB GPU
        d_state: int = 4,        # Reduced state dimension
        d_conv: int = 3,
        expand: int = 1.0,       # No expansion for memory efficiency
    ):
        super().__init__()
        
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 6,  # Smaller progression
        ]
        
        self.mamba_unet = MambaUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            channels=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba_encoder=True,
            use_mamba_decoder=True,
            use_mamba_bottleneck=True,
        )
    
    def forward(self, x):
        return self.mamba_unet(x)


def count_parameters(model):
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test regular MambaUNet
    model = MambaUNet(n_channels=3, n_classes=1)
    print(f"MambaUNet parameters: {count_parameters(model):,}")
    
    # Test lightweight version
    lightweight_model = LightweightMambaUNet(n_channels=3, n_classes=1, base_channels=32)
    print(f"LightweightMambaUNet parameters: {count_parameters(lightweight_model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    
    if torch.cuda.is_available():
        model = model.to(device)
        lightweight_model = lightweight_model.to(device)
        x = x.to(device)
    
    print("Testing MambaUNet forward pass...")
    with torch.no_grad():
        out1 = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out1.shape}")
    
    print("Testing LightweightMambaUNet forward pass...")
    with torch.no_grad():
        out2 = lightweight_model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out2.shape}")
    
    print("✅ All tests passed!")
