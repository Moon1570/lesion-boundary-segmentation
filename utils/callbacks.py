#!/usr/bin/env python3
"""
Training callbacks for monitoring and controlling training process.

Implements various callbacks for training:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Custom training callbacks

Author: GitHub Copilot
"""

import os
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings


class EarlyStopping:
    """
    Early stopping callback to stop training when a monitored metric stops improving.
    
    Args:
        patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in monitored quantity to qualify as an improvement
        mode: 'min' for decreasing metrics (loss), 'max' for increasing metrics (accuracy)
        restore_best_weights: Whether to restore model weights from the best epoch
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} is unknown")
    
    def __call__(self, current_value: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current value of monitored metric
            model: Model to save weights from (if restore_best_weights=True)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            
            # Restore best weights
            if self.restore_best_weights and model is not None and self.best_weights:
                model.load_state_dict(self.best_weights)
                
            return True
        
        return False
    
    def get_best_value(self) -> float:
        """Get the best monitored value."""
        return self.best


class ModelCheckpoint:
    """
    Model checkpoint callback to save the model during training.
    
    Args:
        filepath: Path to save checkpoints
        monitor: Metric to monitor for saving best model
        mode: 'min' or 'max' for the monitored metric
        save_best_only: Only save model when it's the best seen so far
        save_last: Always save the model from the last epoch
        save_weights_only: Save only model weights, not the entire model
        period: Save checkpoint every 'period' epochs
    """
    
    def __init__(self, filepath: Union[str, Path], monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True, 
                 save_last: bool = True, save_weights_only: bool = False,
                 period: int = 1):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.period = period
        
        self.epochs_since_last_save = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} is unknown")
    
    def __call__(self, epoch: int, logs: Dict[str, float], 
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None) -> bool:
        """
        Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics for current epoch
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            scaler: AMP scaler to save
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        self.epochs_since_last_save += 1
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            warnings.warn(f"Metric '{self.monitor}' not found in logs. Available metrics: {list(logs.keys())}")
            current_value = 0.0
        
        # Check if we should save
        save_checkpoint = False
        
        if self.save_best_only:
            if self.monitor_op(current_value, self.best):
                self.best = current_value
                save_checkpoint = True
        else:
            if self.epochs_since_last_save >= self.period:
                save_checkpoint = True
        
        # Always save if save_last and it's not save_best_only
        if self.save_last and not self.save_best_only:
            save_checkpoint = True
        
        if save_checkpoint:
            self._save_checkpoint(epoch, logs, model, optimizer, scheduler, scaler)
            self.epochs_since_last_save = 0
            return True
        
        return False
    
    def _save_checkpoint(self, epoch: int, logs: Dict[str, float],
                        model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """Save the actual checkpoint."""
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logs': logs,
            'best_value': self.best
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Determine filename
        if self.save_best_only:
            filename = 'best_checkpoint.pth'
        else:
            filename = f'checkpoint_epoch_{epoch:03d}.pth'
        
        filepath = self.filepath / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Also save as latest
        if self.save_last:
            latest_path = self.filepath / 'latest_checkpoint.pth'
            torch.save(checkpoint, latest_path)


class ReduceLROnPlateau:
    """
    Wrapper for PyTorch's ReduceLROnPlateau scheduler that works with our callback system.
    
    Args:
        scheduler: PyTorch ReduceLROnPlateau scheduler instance
        monitor: Metric to monitor
    """
    
    def __init__(self, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
                 monitor: str = 'val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor
    
    def __call__(self, logs: Dict[str, float]) -> bool:
        """
        Step the scheduler based on monitored metric.
        
        Args:
            logs: Dictionary of metrics for current epoch
            
        Returns:
            True if learning rate was reduced, False otherwise
        """
        current_value = logs.get(self.monitor)
        if current_value is None:
            warnings.warn(f"Metric '{self.monitor}' not found in logs")
            return False
        
        # Get current LR
        old_lr = self.scheduler.optimizer.param_groups[0]['lr']
        
        # Step scheduler
        self.scheduler.step(current_value)
        
        # Check if LR was reduced
        new_lr = self.scheduler.optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"   📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
            return True
        
        return False


class LearningRateLogger:
    """
    Callback to log learning rate during training.
    """
    
    def __init__(self):
        self.lr_history = []
    
    def __call__(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Log current learning rate.
        
        Args:
            optimizer: Optimizer to get learning rate from
            
        Returns:
            Current learning rate
        """
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        return current_lr
    
    def get_history(self) -> List[float]:
        """Get learning rate history."""
        return self.lr_history.copy()


class MetricLogger:
    """
    Callback to log and track metrics during training.
    """
    
    def __init__(self):
        self.train_metrics = {}
        self.val_metrics = {}
        self.epochs = []
    
    def __call__(self, epoch: int, train_logs: Dict[str, float], 
                 val_logs: Dict[str, float]):
        """
        Log metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            train_logs: Training metrics
            val_logs: Validation metrics
        """
        self.epochs.append(epoch)
        
        # Log training metrics
        for metric, value in train_logs.items():
            if metric not in self.train_metrics:
                self.train_metrics[metric] = []
            self.train_metrics[metric].append(value)
        
        # Log validation metrics
        for metric, value in val_logs.items():
            if metric not in self.val_metrics:
                self.val_metrics[metric] = []
            self.val_metrics[metric].append(value)
    
    def get_best_metric(self, metric: str, split: str = 'val', mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value for a specific metric.
        
        Args:
            metric: Metric name
            split: 'train' or 'val'
            mode: 'max' or 'min'
            
        Returns:
            (best_value, best_epoch)
        """
        metrics_dict = self.val_metrics if split == 'val' else self.train_metrics
        
        if metric not in metrics_dict:
            raise ValueError(f"Metric '{metric}' not found in {split} metrics")
        
        values = metrics_dict[metric]
        
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        best_value = values[best_idx]
        best_epoch = self.epochs[best_idx]
        
        return best_value, best_epoch
    
    def get_metric_history(self, metric: str, split: str = 'val') -> List[float]:
        """Get history for a specific metric."""
        metrics_dict = self.val_metrics if split == 'val' else self.train_metrics
        return metrics_dict.get(metric, []).copy()


class CustomCallback:
    """
    Base class for custom callbacks.
    
    Subclass this to create custom callbacks for training.
    """
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.
    """
    
    def __init__(self, callbacks: Optional[List[CustomCallback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: CustomCallback):
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


def test_callbacks():
    """Test the callback implementations."""
    print("🧪 Testing Training Callbacks")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='min')
    
    print("\n📊 Testing Early Stopping:")
    values = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]  # Should stop after patience
    for i, val in enumerate(values):
        should_stop = early_stopping(val)
        print(f"   Epoch {i+1}: Loss={val:.3f}, Stop={should_stop}")
        if should_stop:
            break
    
    # Test model checkpoint (minimal test)
    checkpoint = ModelCheckpoint('test_checkpoints', monitor='val_dice', mode='max')
    print(f"\n💾 ModelCheckpoint created with monitor='val_dice'")
    
    # Test metric logger
    logger = MetricLogger()
    
    # Simulate some training epochs
    for epoch in range(5):
        train_logs = {'loss': 1.0 - epoch * 0.1, 'dice': 0.5 + epoch * 0.1}
        val_logs = {'loss': 1.1 - epoch * 0.1, 'dice': 0.45 + epoch * 0.1}
        logger(epoch, train_logs, val_logs)
    
    # Get best metrics
    best_dice, best_epoch = logger.get_best_metric('dice', 'val', 'max')
    print(f"\n📈 Best validation Dice: {best_dice:.3f} at epoch {best_epoch}")
    
    print("\n✅ Callbacks test completed!")


if __name__ == "__main__":
    test_callbacks()
