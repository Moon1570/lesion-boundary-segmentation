#!/usr/bin/env python3
"""
DuaSkinSeg: Dual Encoder Deep Learning Model for Skin Lesion Segmentation

Based on the paper:
"Precision and efficiency in skin cancer segmentation through a dual encoder deep learning model"
by Asaad Ahmed et al., Nature Scientific Reports (2025)

This implementation combines:
1. MobileNetV2 encoder for efficient local feature extraction
2. Vision Transformer (ViT) encoder for global context capture
3. CNN decoder with skip connections for feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torchvision.models as models
from transformers import ViTModel, ViTConfig
import math


class PatchEmbedding(nn.Module):
    """
    Convert image to patches and embed them.
    Based on Vision Transformer patch embedding.
    """
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Patches of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Vision Transformer.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and MLP.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection  
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder for global feature extraction.
    """
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features at multiple layers for skip connections.
        
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            List of feature tensors from layers 3, 6, 9, 12
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        features = []
        
        # Extract features at specific layers (as mentioned in paper)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i + 1 in [3, 6, 9, 12]:  # Layers Z3, Z6, Z9, Z12
                features.append(x)
        
        return features


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 Encoder for efficient local feature extraction.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load pre-trained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features
        
        # Extract features from bottleneck 13 (as mentioned in paper)
        self.bottleneck_13 = nn.Sequential(*list(self.features.children())[:14])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract local features using MobileNetV2.
        
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Feature tensor from bottleneck 13
        """
        return self.bottleneck_13(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connections and feature fusion.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        
        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DuaSkinSeg(nn.Module):
    """
    DuaSkinSeg: Dual Encoder Deep Learning Model for Skin Lesion Segmentation.
    
    Combines MobileNetV2 and Vision Transformer encoders with CNN decoder.
    """
    def __init__(self, 
                 img_size: int = 256,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Dual Encoders
        self.mobilenet_encoder = MobileNetV2Encoder(pretrained=True)
        self.vit_encoder = ViTEncoder(img_size, patch_size, in_channels, embed_dim, depth, num_heads, dropout)
        
        # Feature dimension calculations
        patches_per_side = img_size // patch_size
        
        # Projection layers to convert ViT features to spatial format
        self.vit_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ) for _ in range(4)  # For Z3, Z6, Z9, Z12
        ])
        
        # Feature fusion at bottleneck
        # MobileNetV2 bottleneck 13 outputs 96 channels at 24x24 resolution (for 384x384 input)
        mobilenet_channels = 96  # Corrected from 320
        vit_channels = 256
        
        self.fusion_conv = nn.Conv2d(mobilenet_channels + vit_channels, 512, kernel_size=1)
        
        # Decoder with skip connections
        self.decoder4 = DecoderBlock(512, 256, 256)  # From fused features + Z9
        self.decoder3 = DecoderBlock(256, 256, 128)  # + Z6
        self.decoder2 = DecoderBlock(128, 256, 64)   # + Z3
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Additional upsampling to reach original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 96 -> 192
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),   # 192 -> 384
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(8, num_classes, kernel_size=1)
        
    def reshape_vit_features(self, vit_features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Reshape ViT features from sequence format to spatial format.
        
        Args:
            vit_features: Features from ViT (B, num_patches, embed_dim)
            target_size: Target spatial size (H, W)
        Returns:
            Reshaped features (B, channels, H, W)
        """
        B, num_patches, embed_dim = vit_features.shape
        patches_per_side = int(math.sqrt(num_patches))
        
        # Reshape to spatial format
        features = vit_features.transpose(1, 2).reshape(
            B, embed_dim, patches_per_side, patches_per_side
        )
        
        # Interpolate to target size
        if features.shape[2:] != target_size:
            features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
        
        return features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DuaSkinSeg.
        
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        # Extract features from both encoders
        mobilenet_features = self.mobilenet_encoder(x)  # (B, 320, H/32, W/32)
        vit_feature_list = self.vit_encoder(x)  # List of 4 feature tensors
        
        # Process ViT features through projections
        processed_vit_features = []
        for i, (vit_feat, projection) in enumerate(zip(vit_feature_list, self.vit_projections)):
            # Project to desired channel dimension
            B, num_patches, _ = vit_feat.shape
            projected = projection(vit_feat)  # (B, num_patches, 256)
            
            # Calculate target spatial size based on decoder level
            if i == 0:  # Z3 -> decoder2 input
                target_size = (self.img_size // 8, self.img_size // 8)
            elif i == 1:  # Z6 -> decoder3 input  
                target_size = (self.img_size // 16, self.img_size // 16)
            elif i == 2:  # Z9 -> decoder4 input
                target_size = (self.img_size // 32, self.img_size // 32)
            else:  # Z12 -> bottleneck fusion
                target_size = mobilenet_features.shape[2:]
            
            spatial_feat = self.reshape_vit_features(projected, target_size)
            processed_vit_features.append(spatial_feat)
        
        # Feature fusion at bottleneck (MobileNetV2 + ViT Z12)
        fused_features = torch.cat([mobilenet_features, processed_vit_features[3]], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Decoder with skip connections
        x = self.decoder4(fused_features, processed_vit_features[2])  # + Z9
        x = self.decoder3(x, processed_vit_features[1])               # + Z6  
        x = self.decoder2(x, processed_vit_features[0])               # + Z3
        x = self.decoder1(x)
        x = self.final_upsample(x)
        
        # Final segmentation
        output = self.segmentation_head(x)
        
        return output


def create_duaskinseg(img_size: int = 384, num_classes: int = 1, **kwargs) -> DuaSkinSeg:
    """
    Create DuaSkinSeg model with specified parameters.
    
    Args:
        img_size: Input image size
        num_classes: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        DuaSkinSeg model
    """
    return DuaSkinSeg(
        img_size=img_size,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    model = create_duaskinseg(img_size=384, num_classes=1)
    
    # Test with random input
    x = torch.randn(2, 3, 384, 384)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
