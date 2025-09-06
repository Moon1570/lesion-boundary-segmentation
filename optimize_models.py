#!/usr/bin/env python3
"""
Model Optimization Tools for DuaSkinSeg and other models.

Provides:
1. Model quantization (INT8)
2. Pruning techniques
3. Knowledge distillation
4. Mobile deployment optimization
5. Memory usage analysis
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import torch.nn.utils.prune as prune
from pathlib import Path
import time
import psutil
import os


class QuantizedWrapper(nn.Module):
    """Wrapper for quantization-aware training."""
    
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def prepare_model_for_quantization(model, sample_input):
    """Prepare model for quantization."""
    wrapped_model = QuantizedWrapper(model)
    wrapped_model.eval()
    
    # Set quantization configuration
    wrapped_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    prepared_model = torch.quantization.prepare(wrapped_model)
    
    # Calibration run with sample input
    with torch.no_grad():
        prepared_model(sample_input)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)
    
    return quantized_model


def apply_structured_pruning(model, pruning_amount=0.3):
    """Apply structured pruning to convolutional layers."""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 8:
            # Apply channel-wise pruning
            prune.ln_structured(
                module, 
                name='weight', 
                amount=pruning_amount, 
                n=2, 
                dim=0  # Prune output channels
            )
    
    return model


def apply_unstructured_pruning(model, pruning_amount=0.5):
    """Apply unstructured magnitude-based pruning."""
    
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global magnitude pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount,
    )
    
    return model


def remove_pruning_reparameterization(model):
    """Remove pruning masks and make pruning permanent."""
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
        if hasattr(module, 'bias_mask'):
            prune.remove(module, 'bias')
    
    return model


def knowledge_distillation_loss(student_outputs, teacher_outputs, true_labels, temperature=4.0, alpha=0.5):
    """Knowledge distillation loss function."""
    
    # Distillation loss
    teacher_probs = torch.softmax(teacher_outputs / temperature, dim=1)
    student_log_probs = torch.log_softmax(student_outputs / temperature, dim=1)
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)
    
    # Student loss
    student_loss = nn.CrossEntropyLoss()(student_outputs, true_labels)
    
    # Combined loss
    total_loss = alpha * distillation_loss * (temperature ** 2) + (1 - alpha) * student_loss
    
    return total_loss


def analyze_model_efficiency(model, input_size=(1, 3, 384, 384), device='cpu'):
    """Analyze model efficiency metrics."""
    
    model.eval()
    model = model.to(device)
    
    # Create sample input
    sample_input = torch.randn(input_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Measure inference time
    warmup_runs = 10
    timing_runs = 100
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # Timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(timing_runs):
            _ = model(sample_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / timing_runs * 1000  # ms
    
    # Memory usage
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Model size estimation
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    results = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'inference_time_ms': avg_inference_time,
        'memory_usage_mb': memory_mb,
        'model_size_mb': param_size_mb,
        'throughput_fps': 1000 / avg_inference_time if avg_inference_time > 0 else 0
    }
    
    return results


def optimize_model_pipeline(model, sample_input, optimization_level='medium'):
    """
    Complete model optimization pipeline.
    
    Args:
        model: PyTorch model to optimize
        sample_input: Sample input tensor for calibration
        optimization_level: 'light', 'medium', 'aggressive'
    
    Returns:
        dict: Optimized models and metrics
    """
    
    results = {}
    
    # Original model analysis
    print("🔍 Analyzing original model...")
    original_metrics = analyze_model_efficiency(model, sample_input.shape)
    results['original'] = {
        'model': model,
        'metrics': original_metrics
    }
    
    if optimization_level in ['medium', 'aggressive']:
        # Apply pruning
        print("✂️  Applying pruning...")
        pruning_amount = 0.3 if optimization_level == 'medium' else 0.5
        
        pruned_model = apply_unstructured_pruning(
            model.copy() if hasattr(model, 'copy') else model,
            pruning_amount=pruning_amount
        )
        
        pruned_metrics = analyze_model_efficiency(pruned_model, sample_input.shape)
        results['pruned'] = {
            'model': pruned_model,
            'metrics': pruned_metrics
        }
    
    if optimization_level == 'aggressive':
        # Apply quantization
        print("🔢 Applying quantization...")
        try:
            quantized_model = prepare_model_for_quantization(model, sample_input)
            quantized_metrics = analyze_model_efficiency(quantized_model, sample_input.shape)
            results['quantized'] = {
                'model': quantized_model,
                'metrics': quantized_metrics
            }
        except Exception as e:
            print(f"Quantization failed: {e}")
    
    return results


def create_mobile_export(model, sample_input, output_path):
    """Export model for mobile deployment."""
    
    model.eval()
    
    # TorchScript export
    try:
        scripted_model = torch.jit.trace(model, sample_input)
        scripted_path = output_path.replace('.pth', '_scripted.pt')
        scripted_model.save(scripted_path)
        print(f"✅ TorchScript model saved: {scripted_path}")
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
    
    # ONNX export
    try:
        import torch.onnx
        onnx_path = output_path.replace('.pth', '.onnx')
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")


if __name__ == "__main__":
    print("🔧 Model Optimization Tools Demo")
    print("=" * 50)
    
    # Test with lightweight DuaSkinSeg
    from models.lightweight_duaskinseg import create_lightweight_duaskinseg
    
    model = create_lightweight_duaskinseg(img_size=384)
    sample_input = torch.randn(1, 3, 384, 384)
    
    # Run optimization pipeline
    results = optimize_model_pipeline(model, sample_input, optimization_level='aggressive')
    
    print("\n📊 Optimization Results:")
    print("-" * 50)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"\n{name.upper()} Model:")
        print(f"  Parameters: {metrics['total_parameters']:,}")
        print(f"  Model Size: {metrics['model_size_mb']:.1f} MB")
        print(f"  Inference Time: {metrics['inference_time_ms']:.2f} ms")
        print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
        
        if name != 'original':
            original_params = results['original']['metrics']['total_parameters']
            reduction = (1 - metrics['total_parameters'] / original_params) * 100
            print(f"  Parameter Reduction: {reduction:.1f}%")
    
    # Export optimized models
    output_dir = Path("optimized_models")
    output_dir.mkdir(exist_ok=True)
    
    for name, result in results.items():
        if name != 'original':
            model_path = output_dir / f"lightweight_duaskinseg_{name}.pth"
            torch.save(result['model'].state_dict(), model_path)
            print(f"💾 Saved {name} model: {model_path}")
    
    print("\n✅ Optimization complete!")
