#!/usr/bin/env python3
"""
Fine-tuning script for lesion boundary segmentation models.

Multi-stage fine-tuning strategy:
1. Standard fine-tuning with reduced LR
2. Decoder-only fine-tuning  
3. Boundary-focused fine-tuning
4. Ensemble knowledge distillation
"""

import os
# Silence TensorFlow oneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List

# Import your existing modules
from train import LesionSegmentationTrainer
from scripts.dataset import create_data_loaders, ISIC2018Dataset


class FineTuningTrainer(LesionSegmentationTrainer):
    """
    Enhanced trainer specifically for fine-tuning with advanced strategies.
    """
    
    def __init__(self, config: Dict[str, Any], pretrained_path: str):
        super().__init__(config)
        self.pretrained_path = pretrained_path
        self.finetune_stage = config.get('finetune_stage', 'standard')
        self.boundary_weight = config.get('boundary_weight', 0.3)
        
        # Load pretrained model
        self.load_pretrained_model()
        
        # Setup fine-tuning specific components
        self.setup_finetune_strategy()
        
    def load_pretrained_model(self):
        """Load pretrained model weights."""
        print(f"Loading pretrained model from: {self.pretrained_path}")
        
        checkpoint = torch.load(self.pretrained_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        
        # Print loading info
        if 'epoch' in checkpoint:
            print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
        if 'val_dice' in checkpoint:
            print(f"✅ Pretrained Dice: {checkpoint['val_dice']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"✅ Pretrained Loss: {checkpoint['val_loss']:.4f}")
            
    def setup_finetune_strategy(self):
        """Setup fine-tuning strategy based on stage."""
        if self.finetune_stage == 'decoder_only':
            self.freeze_encoder()
        elif self.finetune_stage == 'boundary_focused':
            self.setup_boundary_focused_training()
        elif self.finetune_stage == 'progressive_unfreeze':
            self.setup_progressive_unfreezing()
            
        print(f"🎯 Fine-tuning strategy: {self.finetune_stage}")
        
    def freeze_encoder(self):
        """Freeze encoder layers for decoder-only fine-tuning."""
        print("❄️ Freezing encoder layers...")
        
        # Freeze encoder blocks
        if hasattr(self.model, 'encoder_blocks'):
            for block in self.model.encoder_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        
        # Freeze pooling layers if they exist
        if hasattr(self.model, 'pools'):
            for pool in self.model.pools:
                for param in pool.parameters():
                    param.requires_grad = False
                    
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        print("🔥 Unfreezing all layers...")
        for param in self.model.parameters():
            param.requires_grad = True
            
    def setup_boundary_focused_training(self):
        """Setup boundary-focused fine-tuning with enhanced loss."""
        print("🎯 Setting up boundary-focused training...")
        
        # Modify loss function to emphasize boundaries
        if hasattr(self, 'criterion'):
            original_weights = self.criterion.weights if hasattr(self.criterion, 'weights') else {}
            
            # Increase boundary weight
            if 'boundary' in original_weights:
                original_weights['boundary'] *= 2.0
            else:
                # Add boundary component if not present
                total_weight = sum(original_weights.values()) if original_weights else 1.0
                for key in original_weights:
                    original_weights[key] *= 0.8  # Reduce others
                original_weights['boundary'] = 0.2 * total_weight
                
            print(f"📊 Enhanced loss weights: {original_weights}")
            
    def setup_progressive_unfreezing(self):
        """Setup progressive unfreezing strategy."""
        print("📈 Setting up progressive unfreezing...")
        self.unfreeze_schedule = {
            5: 'decoder',      # Unfreeze decoder after 5 epochs
            15: 'skip',        # Unfreeze skip connections after 15 epochs  
            25: 'encoder',     # Unfreeze encoder after 25 epochs
        }
        
        # Start with everything frozen except final layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Only unfreeze final layers initially
        if hasattr(self.model, 'final_conv'):
            for param in self.model.final_conv.parameters():
                param.requires_grad = True
                
    def progressive_unfreeze_step(self, epoch: int):
        """Execute progressive unfreezing based on epoch."""
        if epoch in self.unfreeze_schedule:
            stage = self.unfreeze_schedule[epoch]
            print(f"🔥 Epoch {epoch}: Unfreezing {stage} layers...")
            
            if stage == 'decoder':
                if hasattr(self.model, 'decoder_blocks'):
                    for block in self.model.decoder_blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                            
            elif stage == 'skip':
                if hasattr(self.model, 'skip_connections'):
                    for skip in self.model.skip_connections:
                        for param in skip.parameters():
                            param.requires_grad = True
                            
            elif stage == 'encoder':
                # Unfreeze encoder gradually
                if hasattr(self.model, 'encoder_blocks'):
                    for block in reversed(self.model.encoder_blocks):  # Start from deepest
                        for param in block.parameters():
                            param.requires_grad = True
                        break  # Unfreeze one block at a time
                        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with fine-tuning strategies."""
        
        # Progressive unfreezing
        if self.finetune_stage == 'progressive_unfreeze':
            self.progressive_unfreeze_step(epoch)
            
        # Call parent method
        return super().train_epoch(epoch)
        
    def setup_enhanced_augmentations(self):
        """Setup enhanced augmentations for fine-tuning."""
        from torchvision import transforms
        
        # More aggressive augmentations for fine-tuning
        enhanced_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
        
        return enhanced_transforms


def create_finetune_config(base_config: Dict[str, Any], 
                          finetune_stage: str,
                          learning_rate: float = 1e-5,
                          epochs: int = 50,
                          batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Create fine-tuning configuration."""
    
    finetune_config = base_config.copy()
    
    # Update training parameters
    finetune_config['training']['epochs'] = epochs
    finetune_config['training']['use_amp'] = True
    finetune_config['training']['gradient_clipping'] = 0.5
    
    # Update optimizer
    finetune_config['optimizer']['lr'] = learning_rate
    finetune_config['optimizer']['weight_decay'] = 1e-5
    
    # Update scheduler for fine-tuning
    finetune_config['scheduler'] = {
        "name": "cosine_warmup",
        "warmup_epochs": 3,
        "T_max": epochs,
        "eta_min": learning_rate / 100,
        "warmup_factor": 0.1
    }
    
    # Batch size adjustment
    if batch_size:
        finetune_config['data']['batch_size'] = batch_size
        
    # Enhanced callbacks
    finetune_config['callbacks']['early_stopping']['patience'] = epochs // 2
    finetune_config['callbacks']['reduce_lr']['patience'] = epochs // 4
    
    # Fine-tuning specific settings
    finetune_config['finetune_stage'] = finetune_stage
    
    # Stage-specific adjustments
    if finetune_stage == 'boundary_focused':
        finetune_config['loss']['weights'] = {
            "bce": 0.2,
            "focal": 0.25,  
            "dice": 0.25,
            "tversky": 0.1,
            "iou": 0.1,
            "boundary": 0.1  # Add boundary focus
        }
        finetune_config['boundary_weight'] = 0.3
        
    elif finetune_stage == 'decoder_only':
        # Higher learning rate for decoder-only training
        finetune_config['optimizer']['lr'] = learning_rate * 2
        
    elif finetune_stage == 'progressive_unfreeze':
        # Longer training for progressive unfreezing
        finetune_config['training']['epochs'] = epochs * 2
        finetune_config['scheduler']['T_max'] = epochs * 2
        
    return finetune_config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune lesion segmentation model')
    
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--config', type=str, 
                       help='Base configuration file (optional)')
    parser.add_argument('--stage', type=str, default='standard',
                       choices=['standard', 'decoder_only', 'boundary_focused', 'progressive_unfreeze'],
                       help='Fine-tuning stage/strategy')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Fine-tuning learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (optional override)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    else:
        # Use UNetMamba config as default
        with open('configs/hypermamba_lightweight.json', 'r') as f:
            base_config = json.load(f)
    
    # Create fine-tuning configuration
    finetune_config = create_finetune_config(
        base_config,
        args.stage,
        args.lr,
        args.epochs,
        args.batch_size
    )
    
    # Set output directory
    if args.output_dir:
        finetune_config['output_dir'] = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.pretrained).parent.parent.name
        finetune_config['output_dir'] = f"runs/finetune_{model_name}_{args.stage}_{timestamp}"
    
    print("🚀 Starting Fine-Tuning...")
    print(f"📁 Pretrained model: {args.pretrained}")
    print(f"🎯 Fine-tuning stage: {args.stage}")
    print(f"📚 Learning rate: {args.lr}")
    print(f"⏱️ Epochs: {args.epochs}")
    print(f"💾 Output: {finetune_config['output_dir']}")
    
    # Initialize trainer
    trainer = FineTuningTrainer(finetune_config, args.pretrained)
    
    # Start fine-tuning
    trainer.train()
    
    print("✅ Fine-tuning completed!")


if __name__ == "__main__":
    main()
