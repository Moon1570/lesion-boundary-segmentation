#!/usr/bin/env python3
"""
Training pipeline for lesion boundary segmentation models.

This script implements a comprehensive training pipeline with:
- Multiple model architectures (Custom U-Net, MONAI U-Net)
- Various loss functions (BCE, Dice, Focal, Boundary, Combined)
- Mixed precision training (AMP)
- Learning rate scheduling
- Comprehensive validation metrics
- Model checkpointing and resuming
- TensorBoard logging
- Early stopping

Usage:
    python train.py --config configs/unet_config.yaml
    python train.py --model custom_unet --loss combined --epochs 100
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.dataset import create_data_loaders, ISIC2018Dataset
from models.unet import UNet, UNetMonai
from models.losses import DiceLoss, FocalLoss, BoundaryLoss, CombinedLoss
from utils.metrics import SegmentationMetrics
from utils.visualization import TrainingVisualizer
from utils.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class LesionSegmentationTrainer:
    """
    Comprehensive trainer for lesion boundary segmentation.
    
    Features:
    - Multiple model architectures
    - Various loss functions
    - Mixed precision training
    - Comprehensive metrics tracking
    - Model checkpointing
    - Early stopping
    - Learning rate scheduling
    """
    
    @staticmethod
    def safe_print(message: str):
        """Safely print messages without unicode encoding issues."""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(safe_message)
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.setup_model()
        self.setup_data_loaders()
        self.setup_loss_function()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_metrics()
        self.setup_callbacks()
        
        # Mixed precision training (setup before logging)
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        self.setup_logging()
        
        print(f"Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {config['model']['name']}")
        print(f"   Loss: {config['loss']['name']}")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"   Batch Size: {config['data']['batch_size']}")
    
    def setup_directories(self):
        """Setup output directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.predictions_dir = self.output_dir / 'predictions'
        self.tensorboard_dir = self.logs_dir / 'tensorboard'
        self.monitoring_dir = self.output_dir / 'monitoring'
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.predictions_dir, 
                        self.tensorboard_dir, self.monitoring_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save training configuration
        config_path = self.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Output directories created:")
        print(f"   Checkpoints: {self.checkpoints_dir}")
        print(f"   TensorBoard: {self.tensorboard_dir}")
        print(f"   Monitoring: {self.monitoring_dir}")
    
    def setup_model(self):
        """Initialize the segmentation model."""
        model_config = self.config['model']
        model_name = model_config['name'].lower()
        
        if model_name == 'custom_unet':
            self.model = UNet(
                n_channels=model_config.get('in_channels', 3),
                n_classes=model_config.get('out_channels', 1),
                bilinear=True,
                channels=model_config.get('encoder_channels', [32, 64, 128, 256])
            )
        elif model_name == 'monai_unet':
            self.model = UNetMonai(
                spatial_dims=2,
                in_channels=model_config.get('in_channels', 3),
                out_channels=model_config.get('out_channels', 1),
                channels=model_config.get('channels', [32, 64, 128, 256, 512]),
                strides=model_config.get('strides', [2, 2, 2, 2]),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def setup_data_loaders(self):
        """Setup data loaders."""
        data_config = self.config['data']
        
        self.data_loaders = create_data_loaders(
            data_dir=data_config['data_dir'],
            splits_dir=data_config.get('splits_dir', 'splits'),
            batch_size=data_config['batch_size'],
            num_workers=data_config.get('num_workers', 4),
            image_size=data_config.get('image_size', 384),
            pin_memory=data_config.get('pin_memory', True)
        )
        
        print(f"Data Loaders:")
        for split, loader in self.data_loaders.items():
            print(f"   {split}: {len(loader)} batches ({len(loader.dataset)} samples)")
    
    def setup_loss_function(self):
        """Setup loss function."""
        loss_config = self.config['loss']
        loss_name = loss_config['name'].lower()
        
        if loss_name == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(loss_config.get('pos_weight', 1.0))
            )
        elif loss_name == 'dice':
            self.criterion = DiceLoss(
                smooth=loss_config.get('smooth', 1.0),
                sigmoid=loss_config.get('sigmoid', True)
            )
        elif loss_name == 'focal':
            self.criterion = FocalLoss(
                alpha=loss_config.get('alpha', 1.0),
                gamma=loss_config.get('gamma', 2.0),
                sigmoid=loss_config.get('sigmoid', True)
            )
        elif loss_name == 'boundary':
            self.criterion = BoundaryLoss(
                boundary_weight=loss_config.get('boundary_weight', 2.0),
                sigmoid=loss_config.get('sigmoid', True)
            )
        elif loss_name == 'combined':
            weights = loss_config.get('weights', {'bce': 0.5, 'dice': 0.3, 'boundary': 0.2})
            self.criterion = CombinedLoss(
                bce_weight=weights.get('bce', 0.5),
                dice_weight=weights.get('dice', 0.3),
                focal_weight=weights.get('focal', 0.0),
                boundary_weight=weights.get('boundary', 0.2)
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        self.criterion = self.criterion.to(self.device)
    
    def setup_optimizer(self):
        """Setup optimizer."""
        optim_config = self.config['optimizer']
        optim_name = optim_config['name'].lower()
        
        if optim_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_config['lr'],
                weight_decay=optim_config.get('weight_decay', 1e-4),
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optim_config['lr'],
                weight_decay=optim_config.get('weight_decay', 1e-2),
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optim_config['lr'],
                momentum=optim_config.get('momentum', 0.9),
                weight_decay=optim_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optim_name}")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        if 'scheduler' not in self.config:
            self.scheduler = None
            return
        
        sched_config = self.config['scheduler']
        sched_name = sched_config['name'].lower()
        
        if sched_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', self.config['training']['epochs']),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                verbose=True
            )
        else:
            self.scheduler = None
    
    def setup_metrics(self):
        """Setup evaluation metrics."""
        self.metrics = SegmentationMetrics(device=self.device)
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks_config = self.config.get('callbacks', {})
        
        # Early stopping
        if 'early_stopping' in callbacks_config:
            es_config = callbacks_config['early_stopping']
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 20),
                min_delta=es_config.get('min_delta', 1e-4),
                mode=es_config.get('mode', 'min')
            )
        else:
            self.early_stopping = None
        
        # Model checkpoint
        self.model_checkpoint = ModelCheckpoint(
            filepath=self.checkpoints_dir,
            monitor='val_dice',
            mode='max',
            save_best_only=True,
            save_last=True
        )
        
        # Learning rate plateau callback
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_plateau = ReduceLROnPlateau(self.scheduler)
        else:
            self.lr_plateau = None
    
    def setup_logging(self):
        """Setup enhanced logging and visualization."""
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.visualizer = TrainingVisualizer(self.predictions_dir)
        
        # Setup text logging
        import logging
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.logs_dir / 'training.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log model architecture to TensorBoard
        sample_input = torch.randn(1, 3, 384, 384).to(self.device)
        self.writer.add_graph(self.model, sample_input)
        
        # Log hyperparameters
        hparams = {
            'model': self.config['model']['name'],
            'loss': self.config['loss']['name'],
            'optimizer': self.config['optimizer']['name'],
            'lr': self.config['optimizer']['lr'],
            'batch_size': self.config['data']['batch_size'],
            'image_size': self.config['data']['image_size'],
            'use_amp': self.use_amp
        }
        self.writer.add_hparams(hparams, {})
        
        # Initial log
        self.logger.info("Training setup completed")
        self.logger.info(f"Model: {self.config['model']['name']}")
        self.logger.info(f"Loss: {self.config['loss']['name']}")
        self.logger.info(f"TensorBoard: {self.tensorboard_dir}")
        
        print(f"TensorBoard logging setup:")
        print(f"   Directory: {self.tensorboard_dir}")
        print(f"   Command: tensorboard --logdir {self.tensorboard_dir}")
        print(f"   URL: http://localhost:6006")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        # Progress bar
        pbar = tqdm(self.data_loaders['train'], desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.use_amp):
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            # Calculate batch metrics
            with torch.no_grad():
                pred_binary = torch.sigmoid(predictions) > 0.5
                batch_metrics = self.metrics.calculate_batch_metrics(pred_binary, masks)
                
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average metrics
        avg_loss = np.mean(epoch_losses)
        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])
        
        epoch_metrics['loss'] = avg_loss
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {}
        
        with torch.no_grad():
            pbar = tqdm(self.data_loaders['val'], desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(device_type='cuda', enabled=self.use_amp):
                    predictions = self.model(images)
                    loss = self.criterion(predictions, masks)
                
                epoch_losses.append(loss.item())
                
                # Calculate metrics
                pred_binary = torch.sigmoid(predictions) > 0.5
                batch_metrics = self.metrics.calculate_batch_metrics(pred_binary, masks)
                
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
                
                # Save predictions for visualization (first batch only)
                if batch_idx == 0 and epoch % self.config.get('vis_freq', 5) == 0:
                    self.visualizer.save_predictions(
                        images, masks, predictions, epoch, split='val'
                    )
        
        # Average metrics
        avg_loss = np.mean(epoch_losses)
        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])
        
        epoch_metrics['loss'] = avg_loss
        return epoch_metrics
    
    def log_metrics(self, train_metrics: Dict[str, float], 
                   val_metrics: Dict[str, float], epoch: int):
        """Log metrics to tensorboard and console."""
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Console output
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics.get('dice', 0):.4f}, "
              f"IoU: {train_metrics.get('iou', 0):.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics.get('dice', 0):.4f}, "
              f"IoU: {val_metrics.get('iou', 0):.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Enhanced checkpoint saving with comprehensive metadata."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Check if this is the best model and update before creating checkpoint
        val_dice = metrics.get('dice', 0)
        val_loss = metrics.get('loss', float('inf'))
        
        # Update best metrics
        is_new_best_dice = val_dice > self.best_val_dice or is_best
        is_new_best_loss = val_loss < self.best_val_loss
        
        if is_new_best_dice:
            self.best_val_dice = val_dice
        if is_new_best_loss:
            self.best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'config': self.config,
            'metrics': metrics,
            'timestamp': timestamp,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'amp_enabled': self.use_amp
        }
        
        # Save current epoch checkpoint
        checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint (always overwrite)
        latest_path = self.checkpoints_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model if this is a new best (based on Dice score primarily)
        if is_new_best_dice:
            best_path = self.checkpoints_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            
            # Also save with timestamp for history
            best_timestamped = self.checkpoints_dir / f'best_model_{timestamp}_dice_{val_dice:.4f}.pth'
            torch.save(checkpoint, best_timestamped)
            
            self.logger.info(f"New best model saved! Dice: {val_dice:.4f}")
            print(f"  New best model saved! Dice: {val_dice:.4f}")
        
        # Log if we got a new best loss (even if not saving as best model)
        if is_new_best_loss:
            self.logger.info(f"New best validation loss: {val_loss:.4f}")
            print(f"  New best loss: {val_loss:.4f}")
        
        # Clean up old checkpoints (keep last 5 epoch checkpoints)
        self._cleanup_old_checkpoints()
        
        # Log checkpoint info
        self.logger.info(f"Checkpoint saved: epoch {epoch+1}, dice: {val_dice:.4f}, loss: {val_loss:.4f}")
        print(f"  Checkpoint saved: {checkpoint_path.name}")
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Clean up old epoch checkpoints, keeping only the most recent ones."""
        epoch_checkpoints = sorted(self.checkpoints_dir.glob('checkpoint_epoch_*.pth'))
        if len(epoch_checkpoints) > keep_last:
            for old_checkpoint in epoch_checkpoints[:-keep_last]:
                try:
                    old_checkpoint.unlink()
                    self.logger.info(f"Cleaned up old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {old_checkpoint.name}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str = None) -> int:
        """
        Load model checkpoint with automatic detection.
        
        Args:
            checkpoint_path: Specific checkpoint path, or None for auto-detection
            
        Returns:
            Starting epoch number
        """
        if checkpoint_path is None:
            # Auto-detect latest checkpoint
            latest_path = self.checkpoints_dir / 'latest_checkpoint.pth'
            if latest_path.exists():
                checkpoint_path = str(latest_path)
            else:
                # Look for the highest numbered epoch checkpoint
                epoch_checkpoints = sorted(self.checkpoints_dir.glob('checkpoint_epoch_*.pth'))
                if epoch_checkpoints:
                    checkpoint_path = str(epoch_checkpoints[-1])
                else:
                    self.logger.info("No checkpoint found, starting from scratch")
                    return 0
        
        if not Path(checkpoint_path).exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        try:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load scaler state for mixed precision
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore best metrics
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
            
            start_epoch = checkpoint['epoch'] + 1
            
            self.logger.info(f"Checkpoint loaded successfully")
            self.logger.info(f"   Resuming from epoch: {start_epoch}")
            self.logger.info(f"   Best validation dice: {self.best_val_dice:.4f}")
            
            print(f"Resumed from checkpoint:")
            print(f"   Epoch: {start_epoch}")
            print(f"   Best Dice: {self.best_val_dice:.4f}")
            print(f"   Best Loss: {self.best_val_loss:.4f}")
            
            return start_epoch
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            print(f"Failed to load checkpoint: {e}")
            return 0
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch'] + 1}")
        print(f"   Best Val Dice: {self.best_val_dice:.4f}")
    
    def train(self, resume: bool = True):
        """Enhanced training loop with resumption capability."""
        
        # Try to resume from checkpoint if requested
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint()
        
        total_epochs = self.config['training']['epochs']
        
        self.logger.info(f"Starting training from epoch {start_epoch+1} to {total_epochs}")
        print(f"\nStarting training from epoch {start_epoch+1}/{total_epochs}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            for epoch in range(start_epoch, total_epochs):
                epoch_start = time.time()
                
                self.logger.info(f"Starting epoch {epoch+1}/{total_epochs}")
                
                # Training phase
                train_metrics = self.train_epoch(epoch)
                
                # Validation phase
                val_metrics = self.validate_epoch(epoch)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Enhanced TensorBoard logging
                self._log_to_tensorboard(train_metrics, val_metrics, epoch)
                
                # Log metrics to console and file
                self.log_metrics(train_metrics, val_metrics, epoch)
                
                # Save checkpoint with enhanced metadata
                is_best = val_metrics.get('dice', 0) > self.best_val_dice
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Callbacks
                if self.early_stopping:
                    if self.early_stopping(val_metrics['loss']):
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                        break
                
                # Timing and progress update
                epoch_time = time.time() - epoch_start
                total_time = time.time() - start_time
                
                self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
                print(f"  Epoch Time: {epoch_time:.2f}s, Total: {total_time/60:.1f}min")
                
                # Save monitoring plots every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self._save_monitoring_plots(epoch)
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            print("\nTraining interrupted! Saving checkpoint...")
            self.save_checkpoint(epoch, val_metrics, is_best=False)
            print("Checkpoint saved. You can resume training later.")
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            print(f"\nTraining failed: {e}")
            # Save emergency checkpoint
            try:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
                print("Emergency checkpoint saved.")
            except:
                print("Failed to save emergency checkpoint.")
            raise
        
        # Training completed successfully
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        print(f"\n🎉 Training completed in {total_time/3600:.2f} hours")
        
        # Final model summary
        self._print_training_summary()
    
    def _log_to_tensorboard(self, train_metrics: Dict[str, float], 
                           val_metrics: Dict[str, float], epoch: int):
        """Enhanced TensorBoard logging."""
        
        # Loss curves
        self.writer.add_scalars('Loss', {
            'Train': train_metrics.get('loss', 0),
            'Validation': val_metrics.get('loss', 0)
        }, epoch)
        
        # Dice coefficient
        if 'dice' in val_metrics:
            self.writer.add_scalar('Metrics/Dice_Score', val_metrics['dice'], epoch)
        
        # IoU score
        if 'iou' in val_metrics:
            self.writer.add_scalar('Metrics/IoU_Score', val_metrics['iou'], epoch)
        
        # Pixel accuracy
        if 'pixel_accuracy' in val_metrics:
            self.writer.add_scalar('Metrics/Pixel_Accuracy', val_metrics['pixel_accuracy'], epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # GPU memory usage (if available)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            self.writer.add_scalars('GPU_Memory', {
                'Allocated_GB': memory_allocated,
                'Reserved_GB': memory_reserved
            }, epoch)
        
        # Model gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                self.writer.add_scalar(f'Gradient_Norms/{name}', param.grad.norm(), epoch)
    
    def _save_monitoring_plots(self, epoch: int):
        """Save monitoring plots for current training state."""
        try:
            import matplotlib.pyplot as plt
            
            # Create a simple progress plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Placeholder for now (would need to track metrics history)
            ax1.set_title('Training Progress')
            ax1.text(0.5, 0.5, f'Epoch {epoch+1}', ha='center', va='center', transform=ax1.transAxes)
            
            ax2.set_title('GPU Status')
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                ax2.text(0.5, 0.5, f'GPU Memory: {memory_used:.1f}GB', ha='center', va='center', transform=ax2.transAxes)
            
            ax3.set_title('Current Metrics')
            ax3.text(0.5, 0.5, f'Best Dice: {self.best_val_dice:.4f}', ha='center', va='center', transform=ax3.transAxes)
            
            ax4.set_title('Training Info')
            ax4.text(0.5, 0.5, f'Device: {self.device}', ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            plot_path = self.monitoring_dir / f'progress_epoch_{epoch+1:03d}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to save monitoring plots: {e}")
    
    def _print_training_summary(self):
        """Print final training summary."""
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Best Validation Dice: {self.best_val_dice:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Final Model: {self.checkpoints_dir / 'best_checkpoint.pth'}")
        print(f"TensorBoard Logs: {self.tensorboard_dir}")
        print(f"Checkpoints: {self.checkpoints_dir}")
        print(f"{'='*60}")
        print(f"   Best Val Dice: {self.best_val_dice:.4f}")
        
        # Close writer
        self.writer.close()


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        "model": {
            "name": "custom_unet",
            "in_channels": 3,
            "out_channels": 1,
            "encoder_channels": [32, 64, 128, 256],
            "bottleneck_channels": 512,
            "decoder_activation": "relu"
        },
        "data": {
            "data_dir": "data/ISIC2018_proc",
            "splits_dir": "splits",
            "batch_size": 8,
            "num_workers": 4,
            "image_size": 384,
            "pin_memory": True
        },
        "loss": {
            "name": "combined",
            "weights": {"bce": 0.5, "dice": 0.3, "boundary": 0.2},
            "sigmoid": True
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "betas": [0.9, 0.999]
        },
        "scheduler": {
            "name": "cosine",
            "T_max": 100,
            "eta_min": 1e-6
        },
        "training": {
            "epochs": 100,
            "use_amp": True,
            "vis_freq": 5
        },
        "callbacks": {
            "early_stopping": {
                "patience": 20,
                "min_delta": 1e-4,
                "mode": "min"
            }
        },
        "output_dir": "runs/training"
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train lesion segmentation model')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='custom_unet',
                       choices=['custom_unet', 'monai_unet'],
                       help='Model architecture')
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['bce', 'dice', 'focal', 'boundary', 'combined'],
                       help='Loss function')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (or "auto" for automatic detection)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not attempt to resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='runs/ckpts',
                       help='Output directory')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Override with command line arguments
        config['model']['name'] = args.model
        config['loss']['name'] = args.loss
        config['training']['epochs'] = args.epochs
        config['data']['batch_size'] = args.batch_size
        config['optimizer']['lr'] = args.lr
        config['output_dir'] = args.output_dir
    
    # Validate CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU (training will be slow)")
        config['data']['num_workers'] = 0  # Avoid multiprocessing issues on CPU
        config['training']['use_amp'] = False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Initialize trainer
    trainer = LesionSegmentationTrainer(config)
    
    # Handle resume logic
    resume_training = not args.no_resume
    if args.resume:
        if args.resume.lower() == "auto":
            # Auto-resume from latest checkpoint
            trainer.train(resume=True)
        else:
            # Resume from specific checkpoint
            trainer.load_checkpoint(args.resume)
            trainer.train(resume=False)  # Don't auto-resume since we manually loaded
    else:
        # Start training (with auto-resume if not disabled)
        trainer.train(resume=resume_training)


if __name__ == "__main__":
    main()
