#!/usr/bin/env python3
"""
Monitor training progress by checking the latest log files.
"""

import os
from pathlib import Path
import json

def check_training_progress():
    """Check current training progress."""
    
    # Check for recent training runs
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ No runs directory found")
        return
    
    # Find latest training runs
    training_dirs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and any(run_dir.glob("logs/**/*.log")):
            training_dirs.append(run_dir)
    
    if not training_dirs:
        print("❌ No active training runs found")
        return
    
    # Sort by modification time (most recent first)
    training_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("🔍 Active Training Runs:")
    print("=" * 50)
    
    for i, run_dir in enumerate(training_dirs[:3]):  # Show top 3
        print(f"\n📊 {run_dir.name}:")
        
        # Check for checkpoints
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                epoch_num = latest_checkpoint.stem.split('_')[-1]
                print(f"  Latest Checkpoint: Epoch {epoch_num}")
        
        # Check for logs
        log_files = list(run_dir.glob("logs/**/*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-5:] if len(lines) >= 5 else lines
                    print(f"  Recent Log Entries:")
                    for line in recent_lines:
                        if line.strip():
                            print(f"    {line.strip()}")
            except:
                print(f"  Log file: {latest_log.name}")
        
        # Check for predictions
        pred_dir = run_dir / "predictions" / "predictions"
        if pred_dir.exists():
            predictions = list(pred_dir.glob("*.png"))
            if predictions:
                latest_pred = max(predictions, key=lambda x: x.stat().st_mtime)
                print(f"  Latest Prediction: {latest_pred.name}")

if __name__ == "__main__":
    check_training_progress()
