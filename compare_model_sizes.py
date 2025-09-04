#!/usr/bin/env python3
"""
Model Size Comparison Script - Compare all DuaSkinSeg variants.
"""

import torch
from models.duaskinseg import create_duaskinseg
from models.lightweight_duaskinseg import create_lightweight_duaskinseg

def compare_model_sizes():
    """Compare different model variants and their sizes."""
    
    print("🔍 DuaSkinSeg Model Size Comparison")
    print("=" * 60)
    
    models = {
        "Original DuaSkinSeg": create_duaskinseg(img_size=384),
        "Lightweight DuaSkinSeg": create_lightweight_duaskinseg(img_size=384)
    }
    
    results = []
    
    for name, model in models.items():
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB (float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # Test inference
        sample_input = torch.randn(1, 3, 384, 384)
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        
        results.append({
            'name': name,
            'parameters': total_params,
            'size_mb': model_size_mb,
            'output_shape': output.shape
        })
        
        print(f"\n📊 {name}:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Model Size: {model_size_mb:.1f} MB")
        print(f"  Output Shape: {output.shape}")
    
    # Calculate reduction
    original = results[0]
    lightweight = results[1]
    
    param_reduction = (1 - lightweight['parameters'] / original['parameters']) * 100
    size_reduction = (1 - lightweight['size_mb'] / original['size_mb']) * 100
    
    print(f"\n🎯 Optimization Results:")
    print(f"  Parameter Reduction: {param_reduction:.1f}%")
    print(f"  Size Reduction: {size_reduction:.1f} MB ({size_reduction:.1f}%)")
    print(f"  Memory Savings: {original['size_mb'] - lightweight['size_mb']:.1f} MB")

if __name__ == "__main__":
    compare_model_sizes()
