#!/usr/bin/env python3
"""
Cleanup old checkpoints to free up disk space.
Keeps only the essential checkpoints.
"""

import os
import glob
import re
from pathlib import Path
from typing import List, Tuple

def get_checkpoint_info(filepath: str) -> Tuple[str, float]:
    """Extract dice score from checkpoint filename."""
    filename = os.path.basename(filepath)
    if 'best_model_' in filename and '_dice_' in filename:
        match = re.search(r'dice_([\d.]+)', filename)
        if match:
            try:
                dice_str = match.group(1).rstrip('.')  # Remove trailing dot if present
                return filename, float(dice_str)
            except ValueError:
                return filename, 0.0
    return filename, 0.0

def cleanup_checkpoints(runs_dir: str = "runs", dry_run: bool = True):
    """
    Clean up old checkpoints from all model directories.
    
    Args:
        runs_dir: Base runs directory
        dry_run: If True, only show what would be deleted
    """
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    for model_dir in Path(runs_dir).glob("*/"):
        ckpt_dir = model_dir / "checkpoints"
        if ckpt_dir.exists():
            checkpoint_dirs.append(ckpt_dir)
    
    total_size_freed = 0
    
    for ckpt_dir in checkpoint_dirs:
        print(f"\n🔍 Processing: {ckpt_dir}")
        
        # Get all checkpoint files
        all_files = list(ckpt_dir.glob("*.pth"))
        if not all_files:
            print("   No checkpoint files found")
            continue
        
        files_to_keep = set()
        files_to_delete = []
        
        # Always keep these essential files
        essential_files = ["best_checkpoint.pth", "latest_checkpoint.pth"]
        for essential in essential_files:
            essential_path = ckpt_dir / essential
            if essential_path.exists():
                files_to_keep.add(str(essential_path))
        
        # Find the best model file (highest dice score)
        best_model_files = []
        epoch_files = []
        
        for file_path in all_files:
            filename = file_path.name
            if filename.startswith("best_model_") and "_dice_" in filename:
                _, dice_score = get_checkpoint_info(str(file_path))
                best_model_files.append((str(file_path), dice_score))
            elif filename.startswith("checkpoint_epoch_"):
                epoch_files.append(str(file_path))
        
        # Keep only the best model file
        if best_model_files:
            best_model_files.sort(key=lambda x: x[1], reverse=True)
            best_file = best_model_files[0][0]
            files_to_keep.add(best_file)
            print(f"   ✅ Keeping best model: {os.path.basename(best_file)} (Dice: {best_model_files[0][1]:.4f})")
            
            # Mark older best models for deletion
            for file_path, dice_score in best_model_files[1:]:
                files_to_delete.append(file_path)
        
        # Mark epoch checkpoints for deletion (keep only latest 2)
        if epoch_files:
            epoch_files.sort()  # Sort by name (which includes epoch number)
            files_to_keep.update(epoch_files[-2:])  # Keep last 2 epoch checkpoints
            files_to_delete.extend(epoch_files[:-2])  # Delete older ones
        
        # Calculate sizes and perform deletion
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                total_size_freed += file_size
                
                if dry_run:
                    print(f"   🗑️  Would delete: {os.path.basename(file_path)} ({file_size / 1024 / 1024:.1f} MB)")
                else:
                    os.remove(file_path)
                    print(f"   🗑️  Deleted: {os.path.basename(file_path)} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Show what's being kept
        kept_files = [f for f in all_files if str(f) in files_to_keep]
        if kept_files:
            print(f"   ✅ Keeping {len(kept_files)} files:")
            for kept_file in sorted(kept_files):
                file_size = os.path.getsize(kept_file)
                print(f"      • {kept_file.name} ({file_size / 1024 / 1024:.1f} MB)")
    
    print(f"\n📊 Summary:")
    print(f"   Total space {'would be' if dry_run else ''} freed: {total_size_freed / 1024 / 1024:.1f} MB")
    
    if dry_run:
        print(f"\n⚠️  This was a DRY RUN. No files were actually deleted.")
        print(f"   Run with dry_run=False to perform actual cleanup.")

if __name__ == "__main__":
    print("🧹 Checkpoint Cleanup Tool")
    print("=" * 50)
    
    # First, show what would be deleted
    print("\n🔍 DRY RUN - Showing what would be deleted:")
    cleanup_checkpoints(dry_run=True)
    
    # Ask for confirmation
    print("\n" + "=" * 50)
    response = input("Do you want to proceed with the cleanup? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n🗑️  Performing actual cleanup...")
        cleanup_checkpoints(dry_run=False)
        print("\n✅ Cleanup completed!")
    else:
        print("\n❌ Cleanup cancelled.")
