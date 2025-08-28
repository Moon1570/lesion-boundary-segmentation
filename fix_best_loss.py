#!/usr/bin/env python3
"""
Check the validation loss from the best epoch and fix the best_val_loss
"""
import torch
from pathlib import Path

def fix_best_loss():
    """Fix the best_val_loss in checkpoints"""
    checkpoints_dir = Path("runs/ckpts/checkpoints")
    
    # Load the epoch 100 checkpoint (our best Dice model)
    epoch_100_path = checkpoints_dir / "checkpoint_epoch_100.pth"
    
    if not epoch_100_path.exists():
        print(f"ERROR: {epoch_100_path} does not exist!")
        return
    
    # Load the epoch 100 checkpoint
    print(f"Loading: {epoch_100_path}")
    checkpoint = torch.load(epoch_100_path, map_location='cpu', weights_only=False)
    
    # Get the actual loss from this epoch
    actual_loss = checkpoint['metrics']['loss']
    print(f"Validation loss from epoch 100: {actual_loss}")
    print(f"Current best_val_loss in checkpoint: {checkpoint['best_val_loss']}")
    
    # Update the best_val_loss to the correct value
    checkpoint['best_val_loss'] = float(actual_loss)
    
    # Save as the corrected best checkpoint
    best_path = checkpoints_dir / "best_checkpoint.pth"
    torch.save(checkpoint, best_path)
    print(f"Corrected best checkpoint saved to: {best_path}")
    
    # Also update the latest checkpoint
    latest_path = checkpoints_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    print(f"Corrected latest checkpoint saved to: {latest_path}")
    
    # Verify the fix
    print("\nVerification:")
    corrected = torch.load(best_path, map_location='cpu', weights_only=False)
    print(f"New best_val_loss: {corrected['best_val_loss']}")
    print(f"New best_val_dice: {corrected['best_val_dice']}")
    print(f"Metrics loss: {corrected['metrics']['loss']}")
    print(f"Metrics dice: {corrected['metrics']['dice']}")

if __name__ == "__main__":
    fix_best_loss()
