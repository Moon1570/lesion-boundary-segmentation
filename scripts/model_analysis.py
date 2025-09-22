#!/usr/bin/env python3
"""
Advanced model visualization for lesion boundary segmentation.

This script creates detailed visualizations of model internals including:
- Feature activation maps at different network depths
- Weight distribution histograms
- Gradient flow analysis
- Model attention maps
- Layer-wise feature evolution

Usage:
    python scripts/model_analysis.py --model_path runs/ckpts/checkpoints/best_checkpoint.pth
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

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

# Visualization imports - simplified without Captum
# from captum.attr import GradientShap, IntegratedGradients, Occlusion, NoiseTunnel
# from captum.attr import GuidedGradCam, LayerConductance, LayerGradCam

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.dataset import create_data_loaders, ISIC2018Dataset
from models.unet import UNet
from utils.visualization import TrainingVisualizer

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelAnalyzer:
    """Analyze and visualize model internals."""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Load data for analysis
        if config_path:
            self.data_loader = self.load_data(config_path)
        else:
            self.data_loader = None
            
        # Hook storage for intermediate activations
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self.register_hooks()
        
    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Determine model architecture from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            model = UNet(
                n_channels=model_config.get('in_channels', 3),
                n_classes=model_config.get('out_channels', 1)
            )
        else:
            # Default configuration
            model = UNet(n_channels=3, n_classes=1)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def load_data(self, config_path: str):
        """Load validation data for analysis."""
        # This would load your validation data
        # For now, we'll create a simple loader
        pass
    
    def register_hooks(self):
        """Register forward and backward hooks for activation extraction."""
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def save_gradient(name):
            def hook(module, input, output):
                self.gradients[name] = output.detach()
            return hook
        
        # Register hooks for different layers
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                module.register_forward_hook(save_activation(name))
                layer_names.append(name)
        
        self.layer_names = layer_names
        print(f"📌 Registered hooks for {len(layer_names)} layers")
    
    def extract_features(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract feature maps from all layers."""
        self.activations.clear()
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return self.activations.copy()
    
    def visualize_feature_maps(self, input_tensor: torch.Tensor, save_dir: str = "feature_maps"):
        """Visualize feature maps at different network depths."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract features
        features = self.extract_features(input_tensor)
        
        # Create comprehensive feature map visualization
        fig = plt.figure(figsize=(20, 24))
        
        # Number of layers to visualize
        key_layers = [
            'inc.double_conv.0',  # First layer
            'down1.double_conv.0',  # Encoder layer 1
            'down2.double_conv.0',  # Encoder layer 2
            'down3.double_conv.0',  # Encoder layer 3
            'down4.double_conv.0',  # Bottleneck
            'up1.double_conv.0',   # Decoder layer 1
            'up2.double_conv.0',   # Decoder layer 2
            'up3.double_conv.0',   # Decoder layer 3
            'up4.double_conv.0',   # Decoder layer 4
        ]
        
        # Filter available layers
        available_layers = [layer for layer in key_layers if layer in features]
        
        if not available_layers:
            print("⚠️ No matching layers found for visualization")
            return
        
        # Create subplots
        n_layers = len(available_layers)
        n_cols = 6  # Number of feature maps per layer
        gs = GridSpec(n_layers, n_cols + 1, hspace=0.3, wspace=0.1)
        
        for i, layer_name in enumerate(available_layers):
            feature_map = features[layer_name][0]  # First sample in batch
            
            # Show input image for reference (first row only)
            if i == 0:
                ax_input = fig.add_subplot(gs[0, 0])
                input_img = input_tensor[0].cpu().permute(1, 2, 0).numpy()
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
                ax_input.imshow(input_img)
                ax_input.set_title('Input Image', fontsize=10, fontweight='bold')
                ax_input.axis('off')
            
            # Visualize first 6 feature maps for each layer
            n_channels = min(6, feature_map.shape[0])
            
            for j in range(n_channels):
                ax = fig.add_subplot(gs[i, j + 1])
                
                feat = feature_map[j].cpu().numpy()
                
                # Normalize feature map
                feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
                
                im = ax.imshow(feat_norm, cmap='viridis', aspect='auto')
                ax.set_title(f'{layer_name.split(".")[-2]}\nCh {j+1}', fontsize=8)
                ax.axis('off')
                
                # Add colorbar for the first feature map of each layer
                if j == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Feature Map Evolution Through U-Net', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / "feature_maps_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🗺️ Feature maps saved to {save_path}")
    
    def visualize_weight_distributions(self, save_dir: str = "weight_analysis"):
        """Visualize weight distributions across layers."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract weights from all convolutional layers
        layer_weights = {}
        layer_biases = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                layer_weights[name] = module.weight.data.cpu().numpy().flatten()
                if module.bias is not None:
                    layer_biases[name] = module.bias.data.cpu().numpy().flatten()
        
        # Create weight distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Weight distribution histogram
        ax1 = axes[0, 0]
        for name, weights in layer_weights.items():
            if 'inc' in name or 'down1' in name or 'down4' in name or 'up1' in name:
                ax1.hist(weights, bins=50, alpha=0.7, label=name.split('.')[0], density=True)
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Weight Distributions by Layer', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Weight statistics by layer
        ax2 = axes[0, 1]
        layer_means = [np.mean(weights) for weights in layer_weights.values()]
        layer_stds = [np.std(weights) for weights in layer_weights.values()]
        layer_names_short = [name.split('.')[0] for name in layer_weights.keys()]
        
        x_pos = np.arange(len(layer_names_short))
        ax2.errorbar(x_pos, layer_means, yerr=layer_stds, fmt='o-', capsize=5)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Weight Value')
        ax2.set_title('Weight Statistics by Layer', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(layer_names_short, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Weight magnitude by layer depth
        ax3 = axes[1, 0]
        weight_norms = [np.linalg.norm(weights) for weights in layer_weights.values()]
        ax3.plot(weight_norms, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Weight Norm (L2)')
        ax3.set_title('Weight Magnitude Through Network', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias distributions (if available)
        ax4 = axes[1, 1]
        if layer_biases:
            for name, biases in layer_biases.items():
                if 'inc' in name or 'down1' in name or 'down4' in name or 'up1' in name:
                    ax4.hist(biases, bins=30, alpha=0.7, label=name.split('.')[0], density=True)
            ax4.set_xlabel('Bias Value')
            ax4.set_ylabel('Density')
            ax4.set_title('Bias Distributions by Layer', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No bias parameters\nfound in model', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=16)
            ax4.set_title('Bias Analysis', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = save_dir / "weight_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⚖️ Weight distributions saved to {save_path}")
    
    def create_attention_maps(self, input_tensor: torch.Tensor, target_layer: str = None, 
                            save_dir: str = "attention_maps"):
        """Create attention maps using Grad-CAM-like techniques."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set model to evaluation mode and enable gradients
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Create target for backpropagation (max activation)
        target = torch.zeros_like(output)
        target[0, 0, output[0, 0].argmax() // output.shape[-1], 
               output[0, 0].argmax() % output.shape[-1]] = 1
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=target, retain_graph=True)
        
        # Get gradients and activations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Process key layers
        key_layers = ['inc.double_conv.0', 'down2.double_conv.0', 'down4.double_conv.0']
        
        for i, layer_name in enumerate(key_layers):
            if layer_name in self.activations:
                # Get activation and gradients
                activation = self.activations[layer_name][0]  # First sample
                
                # Compute attention map (simple version)
                attention = torch.mean(activation, dim=0).cpu().numpy()
                attention = (attention - attention.min()) / (attention.max() - attention.min())
                
                # Original input
                ax_orig = axes[0, i]
                input_img = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
                ax_orig.imshow(input_img)
                ax_orig.set_title(f'Input Image\n{layer_name}', fontweight='bold')
                ax_orig.axis('off')
                
                # Attention map
                ax_att = axes[1, i]
                im = ax_att.imshow(attention, cmap='hot', alpha=0.8)
                ax_att.set_title(f'Attention Map\n{layer_name}', fontweight='bold')
                ax_att.axis('off')
                plt.colorbar(im, ax=ax_att, fraction=0.046, pad=0.04)
        
        plt.suptitle('Attention Maps at Different Network Depths', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / "attention_maps.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎯 Attention maps saved to {save_path}")
    
    def analyze_layer_statistics(self, save_dir: str = "layer_analysis"):
        """Analyze and visualize layer-wise statistics."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Collect layer information
        layer_info = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                }
                
                if hasattr(module, 'in_channels'):
                    info['in_channels'] = module.in_channels
                if hasattr(module, 'out_channels'):
                    info['out_channels'] = module.out_channels
                if hasattr(module, 'kernel_size'):
                    info['kernel_size'] = module.kernel_size
                
                layer_info.append(info)
        
        # Create DataFrame
        df = pd.DataFrame(layer_info)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Parameter count by layer
        ax1 = axes[0, 0]
        conv_layers = df[df['type'].isin(['Conv2d', 'ConvTranspose2d'])]
        if not conv_layers.empty:
            ax1.bar(range(len(conv_layers)), conv_layers['parameters'])
            ax1.set_xlabel('Layer Index')
            ax1.set_ylabel('Parameter Count')
            ax1.set_title('Parameters per Convolutional Layer', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Channel progression
        ax2 = axes[0, 1]
        if 'in_channels' in df.columns and 'out_channels' in df.columns:
            conv_df = df[df['type'] == 'Conv2d'].dropna(subset=['in_channels', 'out_channels'])
            if not conv_df.empty:
                ax2.plot(conv_df['in_channels'], 'o-', label='Input Channels', linewidth=2)
                ax2.plot(conv_df['out_channels'], 's-', label='Output Channels', linewidth=2)
                ax2.set_xlabel('Layer Index')
                ax2.set_ylabel('Channel Count')
                ax2.set_title('Channel Progression Through Network', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Layer type distribution
        ax3 = axes[1, 0]
        type_counts = df['type'].value_counts()
        ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax3.set_title('Layer Type Distribution', fontweight='bold')
        
        # 4. Parameter distribution
        ax4 = axes[1, 1]
        param_data = df[df['parameters'] > 0]['parameters']
        if not param_data.empty:
            ax4.hist(param_data, bins=20, edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Parameter Count')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Parameter Count Distribution', fontweight='bold')
            ax4.set_yscale('log')
        
        plt.tight_layout()
        
        save_path = save_dir / "layer_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save layer information as CSV
        csv_path = save_dir / "layer_info.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Layer statistics saved to {save_path}")
        print(f"📄 Layer information saved to {csv_path}")


def create_sample_input():
    """Create a sample input tensor for analysis."""
    # Create a synthetic lesion image for demonstration
    # In practice, you'd use real validation samples
    
    sample_input = torch.randn(1, 3, 384, 384)
    
    # Add some structure to make it more realistic
    center_x, center_y = 192, 192
    radius = 50
    
    # Create a circular lesion-like pattern
    y, x = np.ogrid[:384, :384]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Apply pattern to one channel
    sample_input[0, 0, mask] *= 2
    sample_input[0, 1, mask] *= 0.5
    
    return sample_input


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize model internals')
    parser.add_argument('--model_path', type=str,
                       default='runs/ckpts/checkpoints/best_checkpoint.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str,
                       default='configs/train_with_masks.json',
                       help='Path to training configuration')
    parser.add_argument('--output_dir', type=str,
                       default='model_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--sample_input', type=str,
                       help='Path to sample input image (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"❌ Model checkpoint not found: {args.model_path}")
        return
    
    print("🔬 Starting model analysis...")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Output: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(args.model_path, args.config_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create sample input
    if args.sample_input and Path(args.sample_input).exists():
        # Load real image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])
        sample_input = transform(Image.open(args.sample_input)).unsqueeze(0)
    else:
        sample_input = create_sample_input()
    
    sample_input = sample_input.to(analyzer.device)
    
    # Run analyses
    print("\n🗺️ Analyzing feature maps...")
    analyzer.visualize_feature_maps(sample_input, str(output_dir / "feature_maps"))
    
    print("⚖️ Analyzing weight distributions...")
    analyzer.visualize_weight_distributions(str(output_dir / "weights"))
    
    print("🎯 Creating attention maps...")
    analyzer.create_attention_maps(sample_input, save_dir=str(output_dir / "attention"))
    
    print("📊 Analyzing layer statistics...")
    analyzer.analyze_layer_statistics(str(output_dir / "layers"))
    
    print(f"\n✅ Model analysis complete!")
    print(f"📂 Results saved to: {output_dir}")
    print("🎨 Use these visualizations in your paper!")


if __name__ == "__main__":
    main()
