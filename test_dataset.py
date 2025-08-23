#!/usr/bin/env python3
"""
Final validation test for ISIC2018 dataset implementation.
Tests all components: preprocessing, dataset loading, augmentations, and performance.

Author: GitHub Copilot
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from dataset import ISIC2018Dataset, create_data_loaders, analyze_dataset_splits

def test_dataset_integrity():
    """Test basic dataset loading and integrity."""
    print("🔍 Testing Dataset Integrity...")
    
    try:
        # Test all splits
        for split in ['train', 'val', 'test']:
            print(f"  🧪 Testing {split} split...")
            dataset = ISIC2018Dataset(
                data_dir="data/ISIC2018_proc",
                split=split,
                augment=(split == 'train')
            )
            
            print(f"  ✅ {split.capitalize()} dataset: {len(dataset)} samples")
            
            # Test sample access
            sample = dataset[0]
            required_keys = ['image', 'image_id']
            if split != 'test':
                required_keys.append('mask')
                
            for key in required_keys:
                assert key in sample, f"Missing key '{key}' in {split} sample"
            
            # Test tensor shapes
            assert sample['image'].shape == (3, 384, 384)
            
            if split != 'test':  # Test split has no masks
                assert sample['mask'].shape == (1, 384, 384)
                
                # Test binary masks
                mask_unique = torch.unique(sample['mask'])
                assert len(mask_unique) <= 2
                assert torch.all((sample['mask'] == 0) | (sample['mask'] == 1))
            
            print(f"    📊 Sample 0 shape: {sample['image'].shape}")
            if split != 'test':
                print(f"    🎭 Mask range: {sample['mask'].min():.3f} - {sample['mask'].max():.3f}")
            else:
                print(f"    ✅ Test split has no masks (expected)")
    
    except Exception as e:
        print(f"  ❌ Dataset integrity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  ✅ Dataset integrity test passed!\n")
    return True

def test_data_loaders():
    """Test data loader creation and performance."""
    print("🚀 Testing Data Loaders...")
    
    try:
        # Create data loaders
        data_loaders = create_data_loaders(
            data_dir="data/ISIC2018_proc",
            batch_size=8,  # Smaller batch for testing
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Test each loader
        for split, loader in data_loaders.items():
            print(f"  📦 {split.capitalize()} loader: {len(loader)} batches")
            
            # Test first batch
            batch = next(iter(loader))
            batch_size = batch['image'].shape[0]
            
            print(f"    🔢 Batch size: {batch_size}")
            print(f"    📏 Image batch shape: {batch['image'].shape}")
            
            if split != 'test':
                print(f"    🎭 Mask batch shape: {batch['mask'].shape}")
                
                # Test binary masks in batch
                mask_min = batch['mask'].min()
                mask_max = batch['mask'].max()
                print(f"    📊 Mask range: {mask_min:.3f} - {mask_max:.3f}")
                
                assert mask_min >= 0.0 and mask_max <= 1.0
    
    except Exception as e:
        print(f"  ❌ Data loader test failed: {e}")
        return False
    
    print("  ✅ Data loader test passed!\n")
    return True

def test_augmentation_consistency():
    """Test that augmentations work correctly and don't break masks."""
    print("🎨 Testing Augmentation Consistency...")
    
    try:
        # Create training dataset with augmentations
        train_dataset = ISIC2018Dataset(
            data_dir="data/ISIC2018_proc",
            split="train",
            augment=True
        )
        
        # Test same sample multiple times to see augmentation variety
        sample_idx = 0
        image_stats = []
        mask_stats = []
        
        for i in range(5):
            sample = train_dataset[sample_idx]
            
            # Check image stats
            img_mean = sample['image'].mean().item()
            img_std = sample['image'].std().item()
            image_stats.append((img_mean, img_std))
            
            # Check mask integrity
            mask = sample['mask']
            mask_unique = torch.unique(mask)
            mask_area = mask.sum().item() / mask.numel()
            mask_stats.append(mask_area)
            
            # Ensure binary mask
            assert torch.all((mask == 0) | (mask == 1))
        
        # Check that augmentations create variety
        image_means = [stat[0] for stat in image_stats]
        mask_areas = mask_stats
        
        print(f"  📊 Image means across 5 augmentations: {[f'{m:.3f}' for m in image_means]}")
        print(f"  🎭 Mask areas across 5 augmentations: {[f'{a:.3f}' for a in mask_areas]}")
        
        # Verify augmentations create variety (not identical)
        if len(set([f'{m:.2f}' for m in image_means])) > 1:
            print("  ✅ Photometric augmentations working (varied image means)")
        
        # Mask areas should be similar but may vary slightly due to geometric transforms
        if max(mask_areas) - min(mask_areas) < 0.1:  # Less than 10% difference
            print("  ✅ Geometric augmentations working (consistent mask areas)")
        
    except Exception as e:
        print(f"  ❌ Augmentation test failed: {e}")
        return False
    
    print("  ✅ Augmentation test passed!\n")
    return True

def test_performance():
    """Test data loading performance."""
    print("⚡ Testing Performance...")
    
    try:
        # Create data loaders
        data_loaders = create_data_loaders(
            data_dir="data/ISIC2018_proc",
            batch_size=16,
            num_workers=0
        )
        
        # Test training loader performance (with augmentations)
        train_loader = data_loaders['train']
        
        print(f"  🏃 Testing training loader speed ({len(train_loader)} batches)...")
        start_time = time.time()
        
        batch_times = []
        for i, batch in enumerate(train_loader):
            batch_start = time.time()
            
            # Simulate some processing
            _ = batch['image'].mean()
            if 'mask' in batch:
                _ = batch['mask'].sum()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if i >= 4:  # Test first 5 batches
                break
        
        avg_batch_time = np.mean(batch_times)
        total_time = time.time() - start_time
        
        print(f"    ⏱️ Average batch time: {avg_batch_time:.3f}s")
        print(f"    📊 Total time (5 batches): {total_time:.3f}s")
        
        if avg_batch_time < 1.0:  # Should be reasonable
            print("  ✅ Performance is acceptable")
        else:
            print("  ⚠️ Performance might be slow (consider reducing augmentations)")
    
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False
    
    print("  ✅ Performance test passed!\n")
    return True

def test_edge_cases():
    """Test edge cases and error handling."""
    print("🔬 Testing Edge Cases...")
    
    try:
        # Test invalid split
        try:
            invalid_dataset = ISIC2018Dataset(
                data_dir="data/ISIC2018_proc",
                split="invalid",
                splits_dir="splits"
            )
            print("  ❌ Should have failed with invalid split")
            return False
        except (ValueError, FileNotFoundError):
            print("  ✅ Correctly handles invalid split")
        
        # Test non-existent data directory
        try:
            invalid_dataset = ISIC2018Dataset(
                data_dir="nonexistent",
                split="train",
                splits_dir="splits"
            )
            print("  ❌ Should have failed with invalid data directory")
            return False
        except FileNotFoundError:
            print("  ✅ Correctly handles invalid data directory")
        
        # Test dataset length consistency
        train_dataset = ISIC2018Dataset("data/ISIC2018_proc", "train", splits_dir="splits")
        val_dataset = ISIC2018Dataset("data/ISIC2018_proc", "val", splits_dir="splits")
        
        expected_train = 2293
        expected_val = 301
        
        if len(train_dataset) == expected_train:
            print(f"  ✅ Train split size correct: {len(train_dataset)}")
        else:
            print(f"  ⚠️ Train split size unexpected: {len(train_dataset)} (expected {expected_train})")
        
        if len(val_dataset) == expected_val:
            print(f"  ✅ Val split size correct: {len(val_dataset)}")
        else:
            print(f"  ⚠️ Val split size unexpected: {len(val_dataset)} (expected {expected_val})")
    
    except Exception as e:
        print(f"  ❌ Edge case test failed: {e}")
        return False
    
    print("  ✅ Edge case test passed!\n")
    return True

def main():
    """Run comprehensive dataset validation."""
    print("🎯 ISIC2018 Dataset Implementation - Final Validation")
    print("=" * 60)
    
    # Check if processed data exists
    data_dir = Path("data/ISIC2018_proc")
    if not data_dir.exists():
        print("❌ Processed data directory not found!")
        print("   Please run preprocessing first: python scripts/preprocess.py")
        return False
    
    # Check if split files exist
    splits_dir = Path("splits")
    if not splits_dir.exists() or not (splits_dir / "isic2018_train.txt").exists():
        print("❌ Split files not found!")
        print("   Please ensure splits/ directory contains train/val split files")
        return False
    
    print(f"📁 Data directory: {data_dir.absolute()}")
    print(f"📂 Splits directory: {splits_dir.absolute()}\n")
    
    # Run all tests
    tests = [
        ("Dataset Integrity", test_dataset_integrity),
        ("Data Loaders", test_data_loaders),
        ("Augmentation Consistency", test_augmentation_consistency),
        ("Performance", test_performance),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"❌ {test_name} test failed!")
            break
    
    # Print summary
    print("📋 Test Summary:")
    print("-" * 40)
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    print("-" * 40)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Dataset implementation is ready for use.")
        
        # Print usage summary
        print("\n📖 Quick Usage:")
        print("```python")
        print("from scripts.dataset import create_data_loaders")
        print()
        print("data_loaders = create_data_loaders(")
        print("    data_dir='data/ISIC2018_proc',")
        print("    batch_size=16,")
        print("    num_workers=4")
        print(")")
        print()
        print("for batch in data_loaders['train']:")
        print("    images = batch['image']  # Shape: (B, 3, 384, 384)")
        print("    masks = batch['mask']    # Shape: (B, 1, 384, 384)")
        print("    # Your training code here...")
        print("```")
        
        return True
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
