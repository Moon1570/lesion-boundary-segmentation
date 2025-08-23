#!/usr/bin/env python3
"""
Test data loaders with splits.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.dataset import create_data_loaders

def test_data_loaders():
    print('🧪 Testing data loaders with splits...')
    
    try:
        data_loaders = create_data_loaders(
            data_dir='data/ISIC2018_proc',
            splits_dir='splits',
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            image_size=384
        )
        
        print(f'📊 Data Loaders Created:')
        for split, loader in data_loaders.items():
            dataset = loader.dataset
            print(f'   {split}: {len(loader)} batches, {len(dataset)} samples')
            print(f'      Has masks: {dataset.masks_dir is not None}')
            
        # Test one batch from training
        train_loader = data_loaders['train']
        sample = next(iter(train_loader))
        
        print(f'\n✅ Sample batch:')
        print(f'   Image shape: {sample["image"].shape}')
        print(f'   Mask shape: {sample["mask"].shape}')
        print(f'   Image range: [{sample["image"].min():.3f}, {sample["image"].max():.3f}]')
        print(f'   Mask unique values: {sample["mask"].unique()}')
        print(f'   Sample IDs: {sample["image_id"][:2]}')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loaders()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
