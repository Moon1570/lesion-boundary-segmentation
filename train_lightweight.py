#!/usr/bin/env python3
"""
Lightweight models training for ensemble diversity.

This script trains smaller, diverse models that can be used in ensemble
while staying within 8GB RAM constraints.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.dataset import create_data_loaders
from utils.metrics import SegmentationMetrics
from models.advanced_losses import FocalLoss, TverskyLoss, IoULoss

class LightweightUNet(nn.Module):
    """Lightweight U-Net with fewer parameters for ensemble diversity."""
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(LightweightUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels*2)
        self.enc3 = self.conv_block(base_channels*2, base_channels*4)
        self.enc4 = self.conv_block(base_channels*4, base_channels*8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = self.upconv(base_channels*16, base_channels*8)
        self.dec4 = self.conv_block(base_channels*16, base_channels*8)
        
        self.upconv3 = self.upconv(base_channels*8, base_channels*4)
        self.dec3 = self.conv_block(base_channels*8, base_channels*4)
        
        self.upconv2 = self.upconv(base_channels*4, base_channels*2)
        self.dec2 = self.conv_block(base_channels*4, base_channels*2)
        
        self.upconv1 = self.upconv(base_channels*2, base_channels)
        self.dec1 = self.conv_block(base_channels*2, base_channels)
        
        # Output
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)


class ResidualUNet(nn.Module):
    """U-Net with residual connections for ensemble diversity."""
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(ResidualUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.residual_block(in_channels, base_channels)
        self.enc2 = self.residual_block(base_channels, base_channels*2)
        self.enc3 = self.residual_block(base_channels*2, base_channels*4)
        self.enc4 = self.residual_block(base_channels*4, base_channels*8)
        
        # Bottleneck
        self.bottleneck = self.residual_block(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = self.upconv(base_channels*16, base_channels*8)
        self.dec4 = self.residual_block(base_channels*16, base_channels*8)
        
        self.upconv3 = self.upconv(base_channels*8, base_channels*4)
        self.dec3 = self.residual_block(base_channels*8, base_channels*4)
        
        self.upconv2 = self.upconv(base_channels*4, base_channels*2)
        self.dec2 = self.residual_block(base_channels*4, base_channels*2)
        
        self.upconv1 = self.upconv(base_channels*2, base_channels)
        self.dec1 = self.residual_block(base_channels*2, base_channels)
        
        # Output
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
    def residual_block(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
    
    def forward(self, x):
        # Similar to LightweightUNet but with residual blocks
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)


class ResidualBlock(nn.Module):
    """Residual block with identity connection."""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Identity connection
        if in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.identity = nn.Identity()
    
    def forward(self, x):
        identity = self.identity(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


def get_model(model_name, base_channels=32):
    """Create model based on name."""
    if model_name == 'lightweight_unet':
        return LightweightUNet(base_channels=base_channels)
    elif model_name == 'residual_unet':
        return ResidualUNet(base_channels=base_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_loss_function(loss_name):
    """Create loss function based on name."""
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'tversky':
        return TverskyLoss()
    elif loss_name == 'iou':
        return IoULoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, metrics_calc, device, epoch):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_metrics = []
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate metrics
            preds = torch.sigmoid(outputs) > 0.5
            metrics = metrics_calc.calculate_batch_metrics(preds, masks)
            all_metrics.append(metrics)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}',
                'IoU': f'{metrics["iou"]:.4f}'
            })
    
    # Average metrics
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return total_loss / len(val_loader), avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Lightweight Models for Ensemble')
    parser.add_argument('--model', default='lightweight_unet', 
                       choices=['lightweight_unet', 'residual_unet'],
                       help='Model architecture to use')
    parser.add_argument('--loss', default='focal',
                       choices=['bce', 'focal', 'tversky', 'iou'],
                       help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--base-channels', type=int, default=32, help='Base number of channels')
    parser.add_argument('--data-dir', default='data/ISIC2018', help='Data directory')
    parser.add_argument('--output-dir', default='runs/lightweight_models', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.loss}_ch{args.base_channels}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training: {run_name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Loss: {args.loss}")
    logger.info(f"Base channels: {args.base_channels}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # Create model
    model = get_model(args.model, base_channels=args.base_channels)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function, optimizer, and metrics
    criterion = get_loss_function(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    metrics_calc = SegmentationMetrics(device=device)
    
    # Training loop
    best_dice = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, metrics_calc, device, epoch)
        
        # Update scheduler
        scheduler.step(val_metrics['dice'])
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        if val_metrics:
            logger.info(f"Val Dice: {val_metrics['dice']:.4f}")
            logger.info(f"Val IoU: {val_metrics['iou']:.4f}")
            logger.info(f"Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
        
        # Save best model
        if val_metrics and val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': vars(args)
            }, output_dir / 'best_model.pth')
            logger.info(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best Dice Score: {best_dice:.4f}")
    logger.info(f"Model saved at: {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
