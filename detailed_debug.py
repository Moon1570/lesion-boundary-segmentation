#!/usr/bin/env python3
"""
Detailed debug script to check dataset and dataloader.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.dataset import ISIC2018Dataset
from torch.utils.data import DataLoader

def debug_dataset():
    """Debug dataset directly."""
    print("Testing dataset directly...")
    
    dataset = ISIC2018Dataset(
        data_dir="data/ISIC2018_proc",
        split="train",
        image_size=384,
        augment=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
    
    if isinstance(sample, dict):
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)} - {value}")

def debug_dataloader():
    """Debug dataloader."""
    print("\nTesting dataloader...")
    
    dataset = ISIC2018Dataset(
        data_dir="data/ISIC2018_proc",
        split="train",
        image_size=384,
        augment=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0  # Important for debugging
    )
    
    print(f"Dataloader length: {len(dataloader)}")
    
    # Get first batch
    try:
        for batch in dataloader:
            print(f"Batch type: {type(batch)}")
            
            if isinstance(batch, dict):
                print(f"Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)} - {value}")
            else:
                print(f"Batch is not a dict: {type(batch)}")
                print(f"Batch content: {batch}")
            
            break  # Just check first batch
            
    except Exception as e:
        print(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()
    debug_dataloader()
