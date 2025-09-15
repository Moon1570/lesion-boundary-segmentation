#!/usr/bin/env python3
"""
Fine-tuning script for DuaSkinSeg model on ISIC-2018 dataset.

This script implements advanced fine-tuning strategies for the DuaSkinSeg model
to improve performance beyond the baseline 0.8785 Dice score.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import argparse
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.duaskinseg import DuaSkinSeg
from train import LesionSegmentationTrainer


def dice_coefficient(preds, targets, smooth=1e-6):
    """Calculate dice coefficient."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'finetune.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, logger: logging.Logger) -> nn.Module:
    """Load pretrained weights for fine-tuning."""
    logger.info(f"Loading pretrained weights from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
        logger.info("Pretrained weights loaded successfully")
    except Exception as e:
        logger.warning(f"Loading weights with strict=False due to: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    return model


def setup_optimizer_with_layerwise_lr(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Setup optimizer with layer-wise learning rates for fine-tuning."""
    base_lr = config['optimizer']['lr']
    
    param_groups = []
    
    # MobileNet encoder - lower learning rate for pretrained features
    mobilenet_params = []
    for name, param in model.named_parameters():
        if 'mobilenet_encoder' in name:
            mobilenet_params.append(param)
    
    if mobilenet_params:
        param_groups.append({
            'params': mobilenet_params,
            'lr': base_lr * 0.1,  # 10x lower for pretrained MobileNet
            'name': 'mobilenet_encoder'
        })
    
    # ViT encoder - medium learning rate
    vit_params = []
    for name, param in model.named_parameters():
        if 'vit_encoder' in name:
            vit_params.append(param)
    
    if vit_params:
        param_groups.append({
            'params': vit_params,
            'lr': base_lr * 0.5,  # 2x lower for ViT
            'name': 'vit_encoder'
        })
    
    # Decoder and fusion layers - full learning rate
    other_params = []
    handled_param_ids = set()
    
    for group in param_groups:
        for param in group['params']:
            handled_param_ids.add(id(param))
    
    for param in model.parameters():
        if id(param) not in handled_param_ids:
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'decoder_fusion'
        })
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['optimizer'].get('weight_decay', 0.005),
        betas=config['optimizer'].get('betas', [0.9, 0.999])
    )
    
    return optimizer


def freeze_encoder_layers(model: nn.Module, freeze_mobilenet: bool = True, freeze_vit: bool = False):
    """Freeze encoder layers for initial fine-tuning epochs."""
    if freeze_mobilenet and hasattr(model, 'mobilenet_encoder'):
        for param in model.mobilenet_encoder.parameters():
            param.requires_grad = False
        print("Frozen MobileNet encoder")
    
    if freeze_vit and hasattr(model, 'vit_encoder'):
        # Freeze only the patch embedding and first few transformer blocks
        if hasattr(model.vit_encoder, 'patch_embed'):
            for param in model.vit_encoder.patch_embed.parameters():
                param.requires_grad = False
        
        if hasattr(model.vit_encoder, 'blocks'):
            for i, block in enumerate(model.vit_encoder.blocks[:6]):  # Freeze first 6 blocks
                for param in block.parameters():
                    param.requires_grad = False
        print("Partially frozen ViT encoder")


def unfreeze_all_layers(model: nn.Module):
    """Unfreeze all model layers."""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfrozen all layers")


def train_epoch(model, train_loader, optimizer, criterion, device, use_amp=True):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            dice = dice_coefficient(preds.float(), masks.float())
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}, Dice: {dice:.4f}')
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


def validate_epoch(model, val_loader, criterion, device, use_amp=True):
    """Validate one epoch."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            preds = torch.sigmoid(outputs) > 0.5
            dice = dice_coefficient(preds.float(), masks.float())
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DuaSkinSeg model')
    parser.add_argument('--config', type=str, 
                      default='configs/finetune_duaskinseg_advanced.json',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    print("🚀 DuaSkinSeg Fine-tuning")
    print("="*50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Modify config for fine-tuning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a fine-tuning config based on existing DuaSkinSeg config
    base_config_path = "configs/duaskinseg_advanced.json"
    
    try:
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Base config not found: {base_config_path}")
        return
    
    # Create fine-tuning modifications
    finetune_config = base_config.copy()
    finetune_config.update({
        "experiment_name": f"duaskinseg_finetune_{timestamp}",
        "pretrained_checkpoint": "runs/duaskinseg_advanced/checkpoints/best_model_20250902_211024_dice_0.8785.pth",
        "optimizer": {
            **base_config.get("optimizer", {}),
            "lr": 5e-6,  # Much lower learning rate
            "weight_decay": 0.0001
        },
        "data": {
            **base_config.get("data", {}),
            "batch_size": 2  # Smaller batch for stability
        },
        "training": {
            **base_config.get("training", {}),
            "epochs": 30  # Fewer epochs for fine-tuning
        }
    })
    
    print(f"📊 Base model: DuaSkinSeg (0.8785 Dice)")
    print(f"🎯 Target: 0.8835+ Dice (+0.5%+)")
    print(f"⚙️  Fine-tuning config created")
    
    # Initialize trainer with fine-tuning config
    trainer = LesionSegmentationTrainer(finetune_config)
    
    # Load pretrained weights if checkpoint exists
    checkpoint_path = finetune_config.get("pretrained_checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📥 Loading pretrained weights: {checkpoint_path}")
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load into model
            trainer.model.load_state_dict(state_dict, strict=False)
            print("✅ Pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained weights: {e}")
            print("🔄 Continuing with random initialization")
    
    # Modify optimizer for fine-tuning (lower learning rates)
    print("🔧 Setting up fine-tuning optimizer...")
    
    # Create parameter groups with different learning rates
    mobilenet_params = []
    vit_params = []
    other_params = []
    
    for name, param in trainer.model.named_parameters():
        if 'mobilenet_encoder' in name:
            mobilenet_params.append(param)
        elif 'vit_encoder' in name:
            vit_params.append(param)
        else:
            other_params.append(param)
    
    # Setup optimizer with layer-wise learning rates
    param_groups = []
    base_lr = finetune_config['optimizer']['lr']
    
    if mobilenet_params:
        param_groups.append({'params': mobilenet_params, 'lr': base_lr * 0.1})
    if vit_params:
        param_groups.append({'params': vit_params, 'lr': base_lr * 0.5})
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr})
    
    trainer.optimizer = optim.AdamW(
        param_groups,
        weight_decay=finetune_config['optimizer'].get('weight_decay', 0.0001)
    )
    
    print(f"📊 Training setup:")
    print(f"   • Learning rates: MobileNet={base_lr*0.1:.2e}, ViT={base_lr*0.5:.2e}, Other={base_lr:.2e}")
    print(f"   • Batch size: {finetune_config['data']['batch_size']}")
    print(f"   • Epochs: {finetune_config['training']['epochs']}")
    
    # Start training
    print("🚀 Starting fine-tuning training...")
    
    try:
        trainer.train()
        print("✅ Fine-tuning completed successfully!")
        
        # Print results
        best_dice = getattr(trainer, 'best_dice', 0.0)
        baseline_dice = 0.8785
        improvement = best_dice - baseline_dice
        
        print(f"\\n📊 Fine-tuning Results:")
        print(f"   • Baseline: {baseline_dice:.4f} Dice")
        print(f"   • Fine-tuned: {best_dice:.4f} Dice")
        print(f"   • Improvement: +{improvement:.4f} ({100*improvement/baseline_dice:.2f}%)")
        
        if improvement > 0:
            print("🎉 Fine-tuning successful!")
        else:
            print("⚠️  No improvement observed")
        
        return best_dice, improvement
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    best_dice, improvement = main()
    if best_dice is not None:
        print(f"\n🎉 Fine-tuning completed!")
        print(f"📊 Best Dice Score: {best_dice:.4f}")
        print(f"📈 Improvement: +{improvement:.4f}")
    else:
        print(f"\n❌ Fine-tuning failed")