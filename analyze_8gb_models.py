#!/usr/bin/env python3
"""
Comprehensive model analysis for 8GB GPU deployment.
Compare different model architectures and their memory requirements.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple

# Model imports
from models.quantized_mamba_unet import QuantizedMambaUNet, create_quantized_mamba_unet
from models.mamba_unet import MambaUNet, LightweightMambaUNet
from models.lightweight_duaskinseg import LightweightDuaSkinSeg
from models.unet import UNet


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB


def measure_gpu_memory(model, input_shape=(2, 3, 384, 384), iterations=3):
    """Measure GPU memory usage during forward pass."""
    if not torch.cuda.is_available():
        return {"peak_memory": 0, "allocated_memory": 0, "forward_time": 0}
    
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory
    times = []
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
    avg_time = np.mean(times)
    
    return {
        "peak_memory": peak_memory,
        "allocated_memory": allocated_memory,
        "forward_time": avg_time,
        "output_shape": output.shape
    }


def analyze_models_for_8gb_gpu():
    """Comprehensive analysis of models for 8GB GPU deployment."""
    
    print("🔍 Analyzing Models for 8GB GPU Deployment")
    print("=" * 60)
    
    # Define model configurations
    model_configs = [
        {
            "name": "UNet (Standard)",
            "model_fn": lambda: UNet(n_channels=3, n_classes=1),
            "category": "Classical"
        },
        {
            "name": "Lightweight DuaSkinSeg",
            "model_fn": lambda: LightweightDuaSkinSeg(img_size=384, num_classes=1),
            "category": "Dual Encoder"
        },
        {
            "name": "Lightweight Mamba U-Net",
            "model_fn": lambda: LightweightMambaUNet(n_channels=3, n_classes=1, base_channels=32),
            "category": "State Space Model"
        },
        {
            "name": "Quantized Mamba (Ultra-Light)",
            "model_fn": lambda: create_quantized_mamba_unet(base_channels=16),
            "category": "Quantized SSM"
        },
        {
            "name": "Quantized Mamba (Lightweight)",
            "model_fn": lambda: create_quantized_mamba_unet(base_channels=24),
            "category": "Quantized SSM"
        },
        {
            "name": "Quantized Mamba (Balanced)",
            "model_fn": lambda: create_quantized_mamba_unet(base_channels=32),
            "category": "Quantized SSM"
        },
        {
            "name": "Quantized Mamba (Performance)",
            "model_fn": lambda: create_quantized_mamba_unet(base_channels=40),
            "category": "Quantized SSM"
        },
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n🔧 Testing: {config['name']}")
        
        try:
            # Create model
            model = config["model_fn"]()
            
            # Basic metrics
            params = count_parameters(model)
            model_size_mb = calculate_model_size_mb(model)
            
            # GPU memory measurement
            gpu_metrics = measure_gpu_memory(model)
            
            # Calculate efficiency metrics
            params_per_mb = params / model_size_mb if model_size_mb > 0 else 0
            
            # Check if fits in 8GB GPU (leaving 1GB buffer)
            fits_8gb = gpu_metrics["peak_memory"] < 7.0
            
            result = {
                "Model": config["name"],
                "Category": config["category"],
                "Parameters": params,
                "Model Size (MB)": model_size_mb,
                "Peak GPU Memory (GB)": gpu_metrics["peak_memory"],
                "Forward Time (ms)": gpu_metrics["forward_time"] * 1000,
                "Params/MB": params_per_mb,
                "Fits 8GB GPU": fits_8gb,
                "Output Shape": str(gpu_metrics["output_shape"])
            }
            
            results.append(result)
            
            # Print summary
            print(f"   Parameters: {params:,}")
            print(f"   Model Size: {model_size_mb:.1f} MB")
            print(f"   Peak GPU Memory: {gpu_metrics['peak_memory']:.3f} GB")
            print(f"   Forward Time: {gpu_metrics['forward_time']*1000:.1f} ms")
            print(f"   8GB GPU Status: {'✅ FITS' if fits_8gb else '❌ TOO LARGE'}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            result = {
                "Model": config["name"],
                "Category": config["category"],
                "Parameters": 0,
                "Model Size (MB)": 0,
                "Peak GPU Memory (GB)": 0,
                "Forward Time (ms)": 0,
                "Params/MB": 0,
                "Fits 8GB GPU": False,
                "Output Shape": "Error"
            }
            results.append(result)
    
    return results


def create_comparison_plots(results):
    """Create visualization plots comparing models."""
    
    df = pd.DataFrame(results)
    
    # Filter out error results
    df_valid = df[df["Parameters"] > 0].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison for 8GB GPU Deployment', fontsize=16, fontweight='bold')
    
    # Color map for categories
    category_colors = {
        'Classical': '#FF6B6B',
        'Dual Encoder': '#4ECDC4', 
        'State Space Model': '#45B7D1',
        'Quantized SSM': '#96CEB4'
    }
    
    colors = [category_colors.get(cat, '#CCCCCC') for cat in df_valid['Category']]
    
    # 1. Parameters vs GPU Memory
    axes[0, 0].scatter(df_valid['Parameters'], df_valid['Peak GPU Memory (GB)'], 
                      c=colors, s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Parameters')
    axes[0, 0].set_ylabel('Peak GPU Memory (GB)')
    axes[0, 0].set_title('Parameters vs GPU Memory Usage')
    axes[0, 0].axhline(y=7.0, color='red', linestyle='--', alpha=0.7, label='8GB GPU Limit')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Model Size vs Forward Time
    axes[0, 1].scatter(df_valid['Model Size (MB)'], df_valid['Forward Time (ms)'], 
                      c=colors, s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Model Size (MB)')
    axes[0, 1].set_ylabel('Forward Time (ms)')
    axes[0, 1].set_title('Model Size vs Inference Speed')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Parameters by Category
    category_params = df_valid.groupby('Category')['Parameters'].mean()
    bars = axes[0, 2].bar(category_params.index, category_params.values, 
                         color=[category_colors[cat] for cat in category_params.index])
    axes[0, 2].set_ylabel('Average Parameters')
    axes[0, 2].set_title('Average Parameters by Category')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom')
    
    # 4. GPU Memory Usage by Model
    model_names = [name.split('(')[0].strip() for name in df_valid['Model']]
    bars = axes[1, 0].bar(range(len(df_valid)), df_valid['Peak GPU Memory (GB)'], 
                         color=colors)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Peak GPU Memory (GB)')
    axes[1, 0].set_title('GPU Memory Usage by Model')
    axes[1, 0].set_xticks(range(len(df_valid)))
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].axhline(y=7.0, color='red', linestyle='--', alpha=0.7, label='8GB Limit')
    axes[1, 0].legend()
    
    # 5. Efficiency Plot (Parameters per MB)
    bars = axes[1, 1].bar(range(len(df_valid)), df_valid['Params/MB'], color=colors)
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Parameters per MB')
    axes[1, 1].set_title('Parameter Efficiency')
    axes[1, 1].set_xticks(range(len(df_valid)))
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    
    # 6. 8GB GPU Compatibility
    compatible_models = df_valid[df_valid['Fits 8GB GPU'] == True]
    compatibility_colors = ['green' if fits else 'red' for fits in df_valid['Fits 8GB GPU']]
    
    bars = axes[1, 2].bar(range(len(df_valid)), [1]*len(df_valid), color=compatibility_colors, alpha=0.7)
    axes[1, 2].set_xlabel('Model')
    axes[1, 2].set_ylabel('8GB GPU Compatible')
    axes[1, 2].set_title('8GB GPU Compatibility')
    axes[1, 2].set_xticks(range(len(df_valid)))
    axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 2].set_ylim(0, 1.2)
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_yticklabels(['❌ No', '✅ Yes'])
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("runs/model_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "model_comparison_8gb_gpu.png", dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {output_dir / 'model_comparison_8gb_gpu.png'}")
    
    return fig


def generate_recommendations(results):
    """Generate deployment recommendations based on analysis."""
    
    df = pd.DataFrame(results)
    df_valid = df[df["Parameters"] > 0].copy()
    df_compatible = df_valid[df_valid["Fits 8GB GPU"] == True].copy()
    
    print("\n🎯 DEPLOYMENT RECOMMENDATIONS FOR 8GB GPU")
    print("=" * 60)
    
    if len(df_compatible) == 0:
        print("❌ No models fit in 8GB GPU with current settings!")
        return
    
    # Best overall model
    df_compatible['efficiency_score'] = (
        (1 / df_compatible['Peak GPU Memory (GB)']) * 0.3 +
        (1 / df_compatible['Forward Time (ms)']) * 0.3 + 
        (df_compatible['Parameters'] / df_compatible['Parameters'].max()) * 0.4
    )
    
    best_overall = df_compatible.loc[df_compatible['efficiency_score'].idxmax()]
    
    print(f"\n🥇 BEST OVERALL: {best_overall['Model']}")
    print(f"   • Parameters: {best_overall['Parameters']:,}")
    print(f"   • GPU Memory: {best_overall['Peak GPU Memory (GB)']:.3f} GB")
    print(f"   • Forward Time: {best_overall['Forward Time (ms)']:.1f} ms")
    print(f"   • Model Size: {best_overall['Model Size (MB)']:.1f} MB")
    
    # Most memory efficient
    most_efficient = df_compatible.loc[df_compatible['Peak GPU Memory (GB)'].idxmin()]
    print(f"\n💾 MOST MEMORY EFFICIENT: {most_efficient['Model']}")
    print(f"   • GPU Memory: {most_efficient['Peak GPU Memory (GB)']:.3f} GB")
    print(f"   • Parameters: {most_efficient['Parameters']:,}")
    
    # Fastest inference
    fastest = df_compatible.loc[df_compatible['Forward Time (ms)'].idxmin()]
    print(f"\n⚡ FASTEST INFERENCE: {fastest['Model']}")
    print(f"   • Forward Time: {fastest['Forward Time (ms)']:.1f} ms")
    print(f"   • GPU Memory: {fastest['Peak GPU Memory (GB)']:.3f} GB")
    
    # Recommended batch sizes
    print(f"\n📦 RECOMMENDED BATCH SIZES (384x384 images):")
    for _, row in df_compatible.iterrows():
        # Estimate batch size based on memory usage
        memory_per_sample = row['Peak GPU Memory (GB)'] / 2  # Current test uses batch size 2
        max_batch_size = int(6.5 / memory_per_sample)  # Leave 1.5GB buffer
        max_batch_size = max(1, min(max_batch_size, 16))  # Reasonable limits
        
        print(f"   • {row['Model']}: {max_batch_size}")
    
    print(f"\n⚙️  OPTIMIZATION TIPS:")
    print(f"   • Use mixed precision (AMP) to reduce memory by ~40%")
    print(f"   • Enable gradient checkpointing during training")
    print(f"   • Consider gradient accumulation for larger effective batch sizes")
    print(f"   • Use quantized models for deployment")
    print(f"   • Monitor GPU memory during training with nvidia-smi")


def main():
    """Main analysis function."""
    
    print("🚀 Starting comprehensive model analysis for 8GB GPU...")
    
    # Run analysis
    results = analyze_models_for_8gb_gpu()
    
    # Create comparison table
    df = pd.DataFrame(results)
    print(f"\n📋 DETAILED COMPARISON TABLE")
    print("=" * 100)
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Save results
    output_dir = Path("runs/model_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_comparison_8gb_gpu.csv", index=False)
    print(f"\n💾 Results saved to: {output_dir / 'model_comparison_8gb_gpu.csv'}")
    
    # Create visualizations
    fig = create_comparison_plots(results)
    
    # Generate recommendations
    generate_recommendations(results)
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
