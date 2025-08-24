#!/usr/bin/env python3
"""
Quick script to inspect checkpoint contents
"""
import torch
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and print its contents"""
    print(f"Inspecting: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: File does not exist!")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"Checkpoint contents:")
        for key, value in checkpoint.items():
            if key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict']:
                print(f"  {key}: <state_dict>")
            else:
                print(f"  {key}: {value}")
        
        print()
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")

if __name__ == "__main__":
    checkpoints_dir = Path("runs/ckpts/checkpoints")
    
    print("=== Checkpoint Inspection ===\n")
    
    # Check latest checkpoint
    latest_path = checkpoints_dir / "latest_checkpoint.pth"
    inspect_checkpoint(latest_path)
    
    # Check best checkpoint
    best_path = checkpoints_dir / "best_checkpoint.pth"
    inspect_checkpoint(best_path)
    
    # Check the 0.8630 best model
    best_0863_path = checkpoints_dir / "best_model_20250824_063507_dice_0.8630.pth"
    inspect_checkpoint(best_0863_path)
