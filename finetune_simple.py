#!/usr/bin/env python3
"""
Simple Fine-tuning Script using existing training infrastructure.
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing training components
from train import LesionSegmentationTrainer


def load_model_from_checkpoint(trainer, checkpoint_path):
    """Load pretrained model weights."""
    print(f"🔄 Loading pretrained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    trainer.model.load_state_dict(state_dict, strict=False)
    
    # Print loading info
    if 'epoch' in checkpoint:
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    if 'val_dice' in checkpoint:
        print(f"✅ Pretrained Dice: {checkpoint['val_dice']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"✅ Pretrained Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint.get('val_dice', 0.0)


def freeze_encoder_layers(model):
    """Freeze encoder layers for decoder-only fine-tuning."""
    print("❄️ Freezing encoder layers...")
    
    frozen_params = 0
    total_params = 0
    
    # Freeze encoder blocks
    if hasattr(model, 'encoder_blocks'):
        for block in model.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
                total_params += param.numel()
    
    # Freeze pooling layers if they exist
    if hasattr(model, 'pools'):
        for pool in model.pools:
            for param in pool.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
                total_params += param.numel()
    
    # Count remaining trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    
    trainable_params = total_params - frozen_params
    print(f"📊 Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")


def create_finetune_config(base_config, stage, lr, epochs, batch_size=None):
    """Create fine-tuning configuration."""
    
    config = base_config.copy()
    
    # Update training parameters for fine-tuning
    config['training']['epochs'] = epochs
    config['optimizer']['lr'] = lr
    config['optimizer']['weight_decay'] = lr / 10  # Reduce weight decay
    
    # Gentle scheduler for fine-tuning
    config['scheduler'] = {
        "name": "cosine_warmup",
        "warmup_epochs": max(2, epochs // 20),
        "T_max": epochs,
        "eta_min": lr / 1000,
        "warmup_factor": 0.1
    }
    
    # Enhanced callbacks for fine-tuning
    config['callbacks']['early_stopping']['patience'] = max(15, epochs // 3)
    config['callbacks']['reduce_lr']['patience'] = max(8, epochs // 5)
    config['callbacks']['reduce_lr']['factor'] = 0.5
    
    # Batch size adjustment
    if batch_size:
        config['data']['batch_size'] = batch_size
    
    # Enhanced augmentations for fine-tuning
    if 'augmentations' in config['data']:
        config['data']['augmentations']['rotation'] = min(30, 
            config['data']['augmentations'].get('rotation', 20) * 1.2)
        config['data']['augmentations']['brightness'] = min(0.3,
            config['data']['augmentations'].get('brightness', 0.15) * 1.3)
        config['data']['augmentations']['contrast'] = min(0.3,
            config['data']['augmentations'].get('contrast', 0.15) * 1.3)
    
    # Stage-specific loss adjustments
    if stage == 'boundary_focused' and 'loss' in config:
        # Emphasize focal loss for boundary precision
        if 'weights' in config['loss']:
            config['loss']['weights']['focal'] = min(0.5, 
                config['loss']['weights'].get('focal', 0.2) * 1.5)
            config['loss']['focal_gamma'] = 3.5  # Increase focus on hard examples
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune lesion segmentation model')
    
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/finetune_standard.json',
                       help='Base configuration file')
    parser.add_argument('--stage', type=str, default='standard',
                       choices=['standard', 'decoder_only', 'boundary_focused'],
                       help='Fine-tuning strategy')
    parser.add_argument('--lr', type=float, default=5e-6,
                       help='Fine-tuning learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size override')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load base configuration
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("Using default UNetMamba config...")
        # Create a minimal config
        base_config = {
            "model": {"name": "unetmamba", "n_channels": 3, "n_classes": 1},
            "data": {
                "data_dir": "data/ISIC2018_proc",
                "splits_dir": "splits",
                "batch_size": 8,
                "num_workers": 4,
                "image_size": 384,
                "pin_memory": True,
                "augmentations": {
                    "enabled": True,
                    "rotation": 20,
                    "flip_prob": 0.5,
                    "brightness": 0.15,
                    "contrast": 0.15
                }
            },
            "loss": {
                "name": "advanced_combined",
                "weights": {"bce": 0.2, "focal": 0.3, "dice": 0.25, "tversky": 0.15, "iou": 0.1}
            },
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            "training": {"epochs": 100, "use_amp": True},
            "callbacks": {
                "early_stopping": {"patience": 20, "mode": "max", "monitor": "val_dice"},
                "reduce_lr": {"patience": 10, "factor": 0.5, "mode": "max", "monitor": "val_dice"}
            }
        }
    else:
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    
    # Create fine-tuning configuration
    finetune_config = create_finetune_config(
        base_config, args.stage, args.lr, args.epochs, args.batch_size
    )
    
    # Set output directory
    if args.output_dir:
        finetune_config['output_dir'] = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pretrained_name = Path(args.pretrained).stem
        finetune_config['output_dir'] = f"runs/finetune_{pretrained_name}_{args.stage}_{timestamp}"
    
    print("🚀 Starting Fine-Tuning...")
    print(f"📁 Pretrained: {args.pretrained}")
    print(f"🎯 Strategy: {args.stage}")
    print(f"📚 Learning Rate: {args.lr}")
    print(f"⏱️ Epochs: {args.epochs}")
    print(f"💾 Output: {finetune_config['output_dir']}")
    
    # Initialize trainer
    trainer = LesionSegmentationTrainer(finetune_config)
    
    # Load pretrained weights
    baseline_dice = load_model_from_checkpoint(trainer, args.pretrained)
    
    # Apply fine-tuning strategy
    if args.stage == 'decoder_only':
        freeze_encoder_layers(trainer.model)
    elif args.stage == 'boundary_focused':
        print("🎯 Boundary-focused fine-tuning enabled")
        # The loss configuration is already adjusted in create_finetune_config
    
    print(f"🎯 Target: Improve beyond {baseline_dice:.4f} Dice")
    print("=" * 60)
    
    # Start fine-tuning
    trainer.train()
    
    print("✅ Fine-tuning completed!")
    
    # Find best result
    checkpoint_dir = Path(finetune_config['output_dir']) / 'checkpoints'
    if checkpoint_dir.exists():
        best_files = list(checkpoint_dir.glob('best_model_*_dice_*.pth'))
        if best_files:
            latest_best = max(best_files, key=lambda x: x.stat().st_mtime)
            dice_str = latest_best.stem.split('_dice_')[-1]
            final_dice = float(dice_str)
            improvement = final_dice - baseline_dice
            print(f"🎉 Final Result: {final_dice:.4f} Dice (+{improvement:+.4f})")


if __name__ == "__main__":
    main()
