#!/usr/bin/env python3
"""
Fix the best checkpoint to have the correct best_val_dice value
"""
import torch
from pathlib import Path

def fix_best_checkpoint():
    """Fix the best checkpoint to have the correct best_val_dice value"""
    checkpoints_dir = Path("runs/ckpts/checkpoints")
    
    # Load the checkpoint that should be the best (epoch 100 with 0.8630)
    epoch_100_path = checkpoints_dir / "checkpoint_epoch_100.pth"
    
    if not epoch_100_path.exists():
        print(f"ERROR: {epoch_100_path} does not exist!")
        return
    
    # Load the epoch 100 checkpoint
    print(f"Loading: {epoch_100_path}")
    checkpoint = torch.load(epoch_100_path, map_location='cpu', weights_only=False)
    
    # Get the actual Dice score from this epoch
    actual_dice = checkpoint['metrics']['dice']
    print(f"Actual Dice from epoch 100: {actual_dice}")
    print(f"Current best_val_dice in checkpoint: {checkpoint['best_val_dice']}")
    
    # Update the best_val_dice to the correct value
    checkpoint['best_val_dice'] = float(actual_dice)
    
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
    print(f"New best_val_dice: {corrected['best_val_dice']}")
    print(f"Metrics dice: {corrected['metrics']['dice']}")
    print(f"Epoch: {corrected['epoch']}")

if __name__ == "__main__":
    fix_best_checkpoint()
