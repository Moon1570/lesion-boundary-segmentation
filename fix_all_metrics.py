#!/usr/bin/env python3
"""
Comprehensive fix for both best_val_dice and best_val_loss in checkpoints
"""
import torch
from pathlib import Path

def fix_all_checkpoint_metrics():
    """Fix both best_val_dice and best_val_loss in all relevant checkpoints"""
    checkpoints_dir = Path("runs/ckpts/checkpoints")
    
    # Load the epoch 100 checkpoint (our actual best model)
    epoch_100_path = checkpoints_dir / "checkpoint_epoch_100.pth"
    
    if not epoch_100_path.exists():
        print(f"ERROR: {epoch_100_path} does not exist!")
        return
    
    # Load the epoch 100 checkpoint
    print(f"Loading: {epoch_100_path}")
    checkpoint = torch.load(epoch_100_path, map_location='cpu', weights_only=False)
    
    # Get the actual metrics from epoch 100
    actual_dice = checkpoint['metrics']['dice']
    actual_loss = checkpoint['metrics']['loss']
    
    print(f"Epoch 100 metrics:")
    print(f"  Dice: {actual_dice}")
    print(f"  Loss: {actual_loss}")
    print(f"Current checkpoint values:")
    print(f"  best_val_dice: {checkpoint['best_val_dice']}")
    print(f"  best_val_loss: {checkpoint['best_val_loss']}")
    
    # Update both values to the correct ones
    checkpoint['best_val_dice'] = float(actual_dice)
    checkpoint['best_val_loss'] = float(actual_loss)
    
    print(f"\nUpdating to:")
    print(f"  best_val_dice: {checkpoint['best_val_dice']}")
    print(f"  best_val_loss: {checkpoint['best_val_loss']}")
    
    # Save corrected checkpoints
    checkpoints_to_update = [
        "best_checkpoint.pth",
        "latest_checkpoint.pth"
    ]
    
    for checkpoint_name in checkpoints_to_update:
        checkpoint_path = checkpoints_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Updated: {checkpoint_path}")
    
    # Verify the fixes
    print("\n=== Verification ===")
    for checkpoint_name in checkpoints_to_update:
        checkpoint_path = checkpoints_dir / checkpoint_name
        corrected = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"{checkpoint_name}:")
        print(f"  best_val_dice: {corrected['best_val_dice']}")
        print(f"  best_val_loss: {corrected['best_val_loss']}")
        print(f"  metrics dice: {corrected['metrics']['dice']}")
        print(f"  metrics loss: {corrected['metrics']['loss']}")
        print()

if __name__ == "__main__":
    fix_all_checkpoint_metrics()
