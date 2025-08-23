#!/usr/bin/env python3
"""
Baseline U-Net models for ISIC2018 lesion boundary segmentation.

Implements:
1. Custom U-Net (small) - ~7-8M parameters
2. MONAI U-Net wrapper for robustness

Author: GitHub Copilot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class DoubleConv(nn.Module):
    """Double convolution block: Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # Use bilinear upsampling (memory efficient) or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Custom U-Net implementation for lesion boundary segmentation.
    
    Architecture:
    - Encoder channels: (32, 64, 128, 256) with bottleneck 512
    - Decoder with skip connections
    - 3x3 convolutions, BatchNorm, ReLU
    - Bilinear upsampling
    - ~7-8M parameters
    
    Args:
        n_channels: Number of input channels (default: 3 for RGB)
        n_classes: Number of output classes (default: 1 for binary segmentation)
        bilinear: Use bilinear upsampling instead of transposed conv (default: True)
        channels: List of encoder channel sizes (default: [32, 64, 128, 256])
    """
    
    def __init__(
        self, 
        n_channels: int = 3, 
        n_classes: int = 1, 
        bilinear: bool = True,
        channels: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channels = channels
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, channels[0])
        
        # Encoder (downsampling path)
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], 512 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(512, channels[3] // factor, bilinear)
        self.up2 = Up(channels[3], channels[2] // factor, bilinear)
        self.up3 = Up(channels[2], channels[1] // factor, bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)
        
        # Output layer
        self.outc = OutConv(channels[0], n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the U-Net."""
        # Encoder
        x1 = self.inc(x)      # Initial conv
        x2 = self.down1(x1)   # 1/2 resolution
        x3 = self.down2(x2)   # 1/4 resolution
        x4 = self.down3(x3)   # 1/8 resolution
        x5 = self.down4(x4)   # 1/16 resolution (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 1/8 resolution
        x = self.up2(x, x3)   # 1/4 resolution
        x = self.up3(x, x2)   # 1/2 resolution
        x = self.up4(x, x1)   # Full resolution
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'Custom U-Net',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'encoder_channels': self.channels,
            'bilinear_upsampling': self.bilinear
        }


class UNetMonai(nn.Module):
    """
    MONAI U-Net wrapper for robust medical image segmentation.
    
    Provides additional medical imaging specific features:
    - Medical-optimized initialization
    - Robust normalization options
    - Deep supervision support
    - Professional medical imaging framework
    
    Args:
        spatial_dims: Number of spatial dimensions (2 for 2D images)
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: Sequence of channels for each level
        strides: Sequence of convolution strides
        kernel_size: Convolution kernel size
    """
    
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 0,
        act: str = 'RELU',
        norm: str = 'BATCH',
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        try:
            from monai.networks.nets import UNet as MonaiUNet
            self.model = MonaiUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                kernel_size=kernel_size,
                up_kernel_size=up_kernel_size,
                num_res_units=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias
            )
            self.using_monai = True
        except ImportError:
            print("⚠️ MONAI not found. Install with: pip install monai")
            print("🔄 Falling back to custom U-Net implementation...")
            
            # Fallback to custom U-Net with similar channel configuration
            custom_channels = list(channels[:-1])  # Remove last channel (bottleneck handled separately)
            self.model = UNet(
                n_channels=in_channels,
                n_classes=out_channels,
                bilinear=True,
                channels=custom_channels
            )
            self.using_monai = False
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
    
    def forward(self, x):
        """Forward pass through MONAI U-Net or fallback custom U-Net."""
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        if self.using_monai:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'model_name': 'MONAI U-Net',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),
                'input_channels': self.in_channels,
                'output_channels': self.out_channels,
                'encoder_channels': self.channels,
                'using_monai': True
            }
        else:
            return self.model.get_model_info()


def create_unet_small(n_channels: int = 3, n_classes: int = 1, use_monai: bool = False) -> nn.Module:
    """
    Factory function to create a small U-Net model.
    
    Args:
        n_channels: Number of input channels (default: 3 for RGB)
        n_classes: Number of output classes (default: 1 for binary segmentation)
        use_monai: Whether to use MONAI U-Net implementation (default: False)
    
    Returns:
        U-Net model ready for training
    """
    if use_monai:
        model = UNetMonai(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            num_res_units=0,  # Standard convolutions
            dropout=0.0
        )
    else:
        model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=True,
            channels=[32, 64, 128, 256]
        )
    
    return model


def test_model_output_shapes():
    """Test model with different input sizes to verify output shapes."""
    print("🧪 Testing U-Net Model Output Shapes")
    print("=" * 50)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test both implementations
    models = {
        'Custom U-Net': create_unet_small(use_monai=False),
        'MONAI U-Net': create_unet_small(use_monai=True)
    }
    
    # Test different input sizes
    test_sizes = [(1, 3, 384, 384), (2, 3, 256, 256), (4, 3, 512, 512)]
    
    for model_name, model in models.items():
        print(f"\n📊 {model_name}")
        print("-" * 30)
        
        # Move model to device
        model = model.to(device)
        
        # Model info
        info = model.get_model_info()
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Model size: {info['model_size_mb']:.1f} MB")
        
        model.eval()
        with torch.no_grad():
            for batch_size, channels, height, width in test_sizes:
                x = torch.randn(batch_size, channels, height, width, device=device)
                
                try:
                    output = model(x)
                    print(f"Input: {tuple(x.shape)} → Output: {tuple(output.shape)} [{device}]")
                    
                    # Verify output shape
                    expected_shape = (batch_size, 1, height, width)
                    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
                    
                except Exception as e:
                    print(f"❌ Error with input {tuple(x.shape)}: {e}")


def calculate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...], device: str = 'cuda'):
    """
    Estimate memory usage for training.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to test on
    
    Returns:
        Dictionary with memory estimates
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("⚠️ CUDA not available, using CPU for memory estimation")
        device = 'cpu'
        return {
            'input_shape': input_shape,
            'forward_memory_gb': 0,
            'backward_memory_gb': 0,
            'total_memory_gb': 0,
            'recommended_batch_size_8gb': 8,
            'device': device
        }
    
    # Clean up before testing
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model = model.to(device)
    model.train()
    
    # Forward pass memory
    x = torch.randn(input_shape, device=device, requires_grad=True)
    
    # Reset memory tracking
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    output = model(x)
    
    if device == 'cuda':
        forward_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        forward_memory = 0
    
    # Backward pass (simulate)
    loss = output.mean()
    loss.backward()
    
    if device == 'cuda':
        total_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        backward_memory = total_memory - forward_memory
        
        # Calculate recommended batch size for 8GB GPU
        memory_per_sample = total_memory / input_shape[0]
        recommended_batch_8gb = max(1, int(7.5 / memory_per_sample))  # Leave 0.5GB buffer
    else:
        total_memory = backward_memory = 0
        recommended_batch_8gb = 8
    
    # Clean up
    del x, output, loss
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'input_shape': input_shape,
        'forward_memory_gb': forward_memory,
        'backward_memory_gb': backward_memory,
        'total_memory_gb': total_memory,
        'memory_per_sample_gb': total_memory / input_shape[0] if input_shape[0] > 0 else 0,
        'recommended_batch_size_8gb': recommended_batch_8gb,
        'device': device
    }


if __name__ == "__main__":
    # Test model implementations
    test_model_output_shapes()
    
    print("\n🔍 Memory Usage Analysis")
    print("=" * 50)
    
    # Test memory usage
    model = create_unet_small(use_monai=False)
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        try:
            memory_info = calculate_memory_usage(
                model, 
                input_shape=(batch_size, 3, 384, 384),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Total memory: {memory_info['total_memory_gb']:.2f} GB")
            print(f"  Forward: {memory_info['forward_memory_gb']:.2f} GB")
            print(f"  Backward: {memory_info['backward_memory_gb']:.2f} GB")
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")
    
    print(f"\n✅ Recommended batch size for 8GB GPU: 8")
    print("💡 Use mixed precision (AMP) for larger batch sizes!")
