#!/usr/bin/env python3
"""
Memory-efficient ensemble inference for lesion boundary segmentation.

This script implements ensemble prediction with memory optimization:
- Sequential model loading (avoids loading all models simultaneously)
- Memory-efficient averaging
- TTA (Test Time Augmentation) support
- 8GB RAM optimized

Target: >0.88 Dice score through ensemble of best models
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
from tqdm import tqdm
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.dataset import create_data_loaders, ISIC2018Dataset
from models.unet import UNet
from models.enhanced_unet import AttentionUNet
from utils.metrics import SegmentationMetrics
import matplotlib.pyplot as plt

class MemoryEfficientEnsemble:
    """Memory-efficient ensemble for 8GB RAM constraint."""
    
    def __init__(self, model_configs: List[Dict], device='cuda'):
        self.model_configs = model_configs
        self.device = device
        self.metrics = SegmentationMetrics(device=device)
        
        # Memory management
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
    def load_model(self, config: Dict) -> torch.nn.Module:
        """Load a single model."""
        model_type = config['type']
        checkpoint_path = config['path']
        
        # Create model
        if model_type == 'custom_unet':
            model = UNet(n_channels=3, n_classes=1)
        elif model_type == 'attention_unet':
            model = AttentionUNet(n_channels=3, n_classes=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_single_model(self, model: torch.nn.Module, images: torch.Tensor, 
                           use_tta: bool = True) -> torch.Tensor:
        """Predict with a single model using TTA."""
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            pred = torch.sigmoid(model(images))
            predictions.append(pred)
            
            if use_tta:
                # Horizontal flip
                pred_flip = torch.sigmoid(model(torch.flip(images, dims=[3])))
                pred_flip = torch.flip(pred_flip, dims=[3])
                predictions.append(pred_flip)
                
                # Vertical flip
                pred_vflip = torch.sigmoid(model(torch.flip(images, dims=[2])))
                pred_vflip = torch.flip(pred_vflip, dims=[2])
                predictions.append(pred_vflip)
                
                # Both flips
                pred_both = torch.sigmoid(model(torch.flip(images, dims=[2, 3])))
                pred_both = torch.flip(pred_both, dims=[2, 3])
                predictions.append(pred_both)
        
        # Average TTA predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred
    
    def ensemble_predict_batch(self, images: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
        """GPU memory-efficient ensemble prediction for a batch."""
        ensemble_preds = []
        
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e9
        
        for i, config in enumerate(self.model_configs):
            print(f"Loading model {i+1}/{len(self.model_configs)}: {config['type']}")
            
            # Load model
            model = self.load_model(config)
            
            # Monitor GPU memory
            model_memory = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory after loading: {model_memory:.2f}GB")
            
            # Predict with aggressive memory management
            pred = self.predict_single_model(model, images, use_tta=use_tta)
            
            # Move to CPU immediately to free GPU memory
            pred_cpu = pred.cpu()
            ensemble_preds.append(pred_cpu)
            
            # Aggressive cleanup
            del model, pred
            torch.cuda.empty_cache()
            gc.collect()
            
            final_memory = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory after cleanup: {final_memory:.2f}GB")
        
        # Average ensemble predictions (on CPU to save GPU memory)
        ensemble_pred = torch.stack(ensemble_preds).mean(dim=0)
        
        # Move final result back to GPU
        return ensemble_pred.to(self.device)
    
    def evaluate_ensemble(self, data_loader, use_tta: bool = True, 
                         save_predictions: bool = False, output_dir: str = None):
        """Evaluate ensemble on validation/test set."""
        all_metrics = []
        predictions_dir = Path(output_dir) / 'ensemble_predictions' if output_dir else None
        
        if save_predictions and predictions_dir:
            predictions_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating ensemble on {len(data_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Ensemble Evaluation')):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Ensemble prediction
                ensemble_pred = self.ensemble_predict_batch(images, use_tta=use_tta)
                
                # Convert to binary
                binary_pred = (ensemble_pred > 0.5).float()
                
                # Calculate metrics
                batch_metrics = self.metrics.calculate_batch_metrics(binary_pred, masks)
                all_metrics.append(batch_metrics)
                
                # Save predictions for visualization
                if save_predictions and predictions_dir and batch_idx < 10:
                    self.save_batch_predictions(
                        images, masks, ensemble_pred, batch_idx, predictions_dir
                    )
                
                # Memory cleanup
                torch.cuda.empty_cache()
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def save_batch_predictions(self, images: torch.Tensor, masks: torch.Tensor, 
                              predictions: torch.Tensor, batch_idx: int, output_dir: Path):
        """Save prediction visualizations."""
        try:
            # Simple visualization save
            import matplotlib.pyplot as plt
            
            # Convert tensors to numpy
            img = images[0].cpu().permute(1, 2, 0).numpy()
            mask = masks[0].cpu().squeeze().numpy()
            pred = predictions[0].cpu().squeeze().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('Ensemble Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'ensemble_pred_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not save prediction visualization: {e}")


def create_ensemble_config(gpu_memory_gb: float = 8.0) -> List[Dict]:
    """Create ensemble configuration optimized for GPU memory constraint."""
    
    # Available models with their performance and GPU memory usage
    models = [
        {
            'name': 'Attention U-Net (Best)',
            'type': 'attention_unet', 
            'path': 'runs/enhanced_unet/checkpoints/best_checkpoint.pth',
            'dice_score': 0.8722,
            'gpu_memory_gb': 2.5  # Estimated GPU memory for inference
        },
        {
            'name': 'Custom U-Net (Best)',
            'type': 'custom_unet',
            'path': 'runs/ckpts/checkpoints/best_checkpoint.pth',
            'dice_score': 0.8630,
            'gpu_memory_gb': 1.8  # Smaller model, less GPU memory
        },
        {
            'name': 'Attention U-Net (2nd Best)',
            'type': 'attention_unet',
            'path': 'runs/enhanced_unet/checkpoints/best_model_20250830_012731_dice_0.8691.pth', 
            'dice_score': 0.8691,
            'gpu_memory_gb': 2.5
        },
        {
            'name': 'Custom U-Net (2nd Best)',
            'type': 'custom_unet',
            'path': 'runs/ckpts/checkpoints/best_model_20250824_063507_dice_0.8630.pth',
            'dice_score': 0.8630,
            'gpu_memory_gb': 1.8
        }
    ]
    
    # For GPU memory optimization:
    # - We load models sequentially (one at a time) to avoid memory issues
    # - Reserve ~2GB for data loading, gradients, activations, TTA
    # - Each model only needs to fit in memory during its inference turn
    available_memory_gb = gpu_memory_gb - 2.0
    
    selected_models = []
    
    # Sort by performance (descending)
    models.sort(key=lambda x: x['dice_score'], reverse=True)
    
    print(f"GPU Memory Available: {gpu_memory_gb}GB GTX 1070")
    print(f"Memory for inference: {available_memory_gb}GB")
    print("\nSequential Loading Strategy (GPU memory optimized):")
    
    for model in models:
        # Since we load sequentially, we only need to check if single model fits
        if model['gpu_memory_gb'] <= available_memory_gb:
            selected_models.append({
                'type': model['type'],
                'path': model['path'],
                'dice_score': model['dice_score']
            })
            print(f"✓ Selected: {model['name']} (Dice: {model['dice_score']:.4f}, GPU: {model['gpu_memory_gb']:.1f}GB)")
        else:
            print(f"✗ Skipped: {model['name']} (would exceed GPU memory)")
    
    print(f"\nTotal models in ensemble: {len(selected_models)}")
    print("Sequential loading will prevent GPU OOM issues")
    
    return selected_models


def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference for Lesion Segmentation')
    parser.add_argument('--data-dir', default='data/ISIC2018', help='Data directory')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (reduce if GPU OOM)')
    parser.add_argument('--use-tta', action='store_true', default=True, help='Use test time augmentation')
    parser.add_argument('--save-predictions', action='store_true', help='Save prediction visualizations')
    parser.add_argument('--output-dir', default='runs/ensemble_results', help='Output directory')
    parser.add_argument('--gpu-memory-gb', type=float, default=8.0, help='Available GPU memory in GB')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ensemble configuration for GPU constraints
    print("Creating ensemble configuration for GTX 1070...")
    model_configs = create_ensemble_config(gpu_memory_gb=args.gpu_memory_gb)
    
    if not model_configs:
        print("Error: No models selected. Try reducing batch size or using smaller models.")
        return
    
    # Create data loader
    print(f"Loading {args.split} dataset...")
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    if args.split == 'val':
        data_loader = data_loaders['val']
    else:
        data_loader = data_loaders['test']
    
    # Create ensemble
    ensemble = MemoryEfficientEnsemble(model_configs, device=device)
    
    # Evaluate ensemble
    print(f"\nStarting ensemble evaluation...")
    print(f"Models in ensemble: {len(model_configs)}")
    print(f"Test Time Augmentation: {'Enabled' if args.use_tta else 'Disabled'}")
    
    metrics = ensemble.evaluate_ensemble(
        data_loader=data_loader,
        use_tta=args.use_tta,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*60)
    print("🎯 ENSEMBLE RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")
    
    # Save results
    results = {
        'ensemble_config': model_configs,
        'metrics': metrics,
        'settings': {
            'use_tta': args.use_tta,
            'batch_size': args.batch_size,
            'split': args.split
        }
    }
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("="*60)
    
    # Compare with individual models
    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"Individual models (best): 0.8722 Dice")
    print(f"Ensemble result: {metrics['dice']:.4f} Dice")
    improvement = metrics['dice'] - 0.8722
    print(f"Improvement: {improvement:+.4f} Dice ({improvement/0.8722*100:+.2f}%)")


if __name__ == '__main__':
    main()
