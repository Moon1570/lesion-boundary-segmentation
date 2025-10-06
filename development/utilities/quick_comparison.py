#!/usr/bin/env python3
"""
Quick test to compare original vs advanced preprocessed data using existing models.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json

from scripts.dataset import create_data_loaders
from models.enhanced_unet import AttentionUNet


def quick_train_test(model: nn.Module, data_dir: str, epochs: int = 5, batch_size: int = 6) -> dict:
    """
    Quick training test to compare datasets.
    
    Args:
        model: Model to train
        data_dir: Dataset directory
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Dictionary with results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    try:
        train_loader, val_loader, _ = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2
        )
        print(f"Loaded data from {data_dir}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error loading data from {data_dir}: {e}")
        return {'error': str(e)}
    
    # Move model to device
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    print(f"Starting training on {data_dir}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # Limit to 10 batches for quick test
                break
            
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 5:  # Limit to 5 batches for quick test
                    break
                
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                else:
                    images, masks = batch
                    images = images.to(device)
                    masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    training_time = (time.time() - start_time) / 60  # Convert to minutes
    
    return {
        'dataset': Path(data_dir).name,
        'best_val_loss': best_val_loss,
        'final_val_loss': avg_val_loss,
        'training_time_minutes': training_time,
        'epochs': epochs
    }


def main():
    """Main comparison function."""
    print("🔬 Quick Dataset Comparison Test")
    print("=" * 50)
    
    # Datasets to compare
    datasets = [
        "data/ISIC2018_proc",
        "data/ISIC2018_advanced"
    ]
    
    # Check if datasets exist
    for dataset in datasets:
        if not Path(dataset).exists():
            print(f"❌ Dataset not found: {dataset}")
            return
    
    results = []
    
    for dataset_dir in datasets:
        print(f"\n📊 Testing dataset: {dataset_dir}")
        print("-" * 30)
        
        # Create fresh model for each test
        model = AttentionUNet(n_channels=3, n_classes=1)
        
        result = quick_train_test(
            model=model,
            data_dir=dataset_dir,
            epochs=5,
            batch_size=6
        )
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Completed: Best Val Loss: {result['best_val_loss']:.4f}")
            print(f"   Training Time: {result['training_time_minutes']:.2f} minutes")
        
        results.append(result)
    
    # Compare results
    print("\n📈 Comparison Results:")
    print("=" * 50)
    
    successful_results = [r for r in results if 'error' not in r]
    
    if len(successful_results) >= 2:
        orig_result = next((r for r in successful_results if 'proc' in r['dataset'] and 'advanced' not in r['dataset']), None)
        adv_result = next((r for r in successful_results if 'advanced' in r['dataset']), None)
        
        if orig_result and adv_result:
            print(f"Original Dataset ({orig_result['dataset']}):")
            print(f"   Best Val Loss: {orig_result['best_val_loss']:.4f}")
            print(f"   Training Time: {orig_result['training_time_minutes']:.2f} min")
            
            print(f"\nAdvanced Dataset ({adv_result['dataset']}):")
            print(f"   Best Val Loss: {adv_result['best_val_loss']:.4f}")
            print(f"   Training Time: {adv_result['training_time_minutes']:.2f} min")
            
            loss_improvement = ((orig_result['best_val_loss'] - adv_result['best_val_loss']) 
                              / orig_result['best_val_loss']) * 100
            
            print(f"\n🎯 Performance Change:")
            if loss_improvement > 0:
                print(f"   📈 Advanced preprocessing improved loss by {loss_improvement:.1f}%")
            else:
                print(f"   📉 Advanced preprocessing increased loss by {abs(loss_improvement):.1f}%")
            
            time_diff = adv_result['training_time_minutes'] - orig_result['training_time_minutes']
            print(f"   ⏱️ Time difference: {time_diff:+.2f} minutes")
        
        # Save detailed results
        with open("quick_comparison_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: quick_comparison_results.json")
    
    else:
        print("❌ Not enough successful results for comparison")
    
    print("\n✅ Quick comparison completed!")


if __name__ == "__main__":
    main()
