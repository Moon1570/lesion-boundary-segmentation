# Create a demo script to see augmentation effects
from dataset import ISIC2018Dataset
import matplotlib.pyplot as plt
import torch

def visualize_augmentations(dataset, idx=0, num_augmentations=8):
    """Show multiple augmented versions of the same image."""
    
    fig, axes = plt.subplots(2, num_augmentations, figsize=(20, 6))
    
    for i in range(num_augmentations):
        sample = dataset[idx]  # Get augmented sample
        
        # Denormalize for visualization
        image = sample['image']
        mean, std = 0.6042, 0.1817
        for c in range(3):
            image[c] = image[c] * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Plot image
        axes[0, i].imshow(image.permute(1, 2, 0))
        axes[0, i].set_title(f'Augmented {i+1}')
        axes[0, i].axis('off')
        
        # Plot mask
        if 'mask' in sample:
            axes[1, i].imshow(sample['mask'].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('runs/figs/augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Usage
train_dataset = ISIC2018Dataset(split='train', augment=True)
visualize_augmentations(train_dataset, idx=0, num_augmentations=6)