# ISIC2018 Dataset Implementation

A comprehensive PyTorch Dataset implementation for the ISIC2018 lesion boundary segmentation challenge with proper train/validation splits and extensive augmentation pipeline using Albumentations.

## 🎯 Key Features

### Dataset Splits
- **Deterministic Splits**: Uses predetermined splits from `splits/*.txt` files
- **Train Split**: 2,293 samples (88.4% of original training data)
- **Validation Split**: 301 samples (11.6% of original training data) 
- **Test Split**: 1,000 samples (no masks available)
- **Total**: 2,594 training samples split deterministically

### Augmentation Pipeline

#### Training Augmentations (p = probability)
```python
# Geometric transformations
- HorizontalFlip(p=0.5)
- VerticalFlip(p=0.5) 
- Rotate(limit=±15°, p=0.7)

# Elastic/Affine (small deformations)
- ElasticTransform(alpha=50, sigma=5, p=0.3)
- GridDistortion(num_steps=5, distort_limit=0.1, p=0.3)
- OpticalDistortion(distort_limit=0.05, p=0.3)

# Photometric transformations
- RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
- CLAHE(clip_limit=2.0, p=0.3)
- RandomGamma(gamma_limit=(80,120), p=0.3)

# Color jitter (light)
- ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4)

# Normalization (always applied)
- Normalize(mean=0.6042, std=0.1817)
```

#### Validation/Test Augmentations
```python
# Minimal processing
- Normalize(mean=0.6042, std=0.1817) only
```

### Data Characteristics
- **Image Size**: 384×384×3 (RGB)
- **Mask Size**: 384×384×1 (Binary)
- **Mask Values**: {0, 1} (binary segmentation)
- **Normalization**: Dataset-specific statistics (mean=0.6042, std=0.1817)
- **Format**: PNG (lossless)

## 🚀 Usage

### Basic Dataset Creation
```python
from scripts.dataset import ISIC2018Dataset

# Training dataset with augmentations
train_dataset = ISIC2018Dataset(
    data_dir="data/ISIC2018_proc",
    split="train", 
    augment=True
)

# Validation dataset without augmentations  
val_dataset = ISIC2018Dataset(
    data_dir="data/ISIC2018_proc",
    split="val",
    augment=False
)
```

### Data Loaders
```python
from scripts.dataset import create_data_loaders

# Create all data loaders at once
data_loaders = create_data_loaders(
    data_dir="data/ISIC2018_proc",
    batch_size=16,
    num_workers=4
)

train_loader = data_loaders['train']  # 286 batches (16 samples each)
val_loader = data_loaders['val']      # 38 batches  
test_loader = data_loaders['test']    # 125 batches
```

### Sample Access
```python
# Get a single sample
sample = train_dataset[0]

# Sample structure:
{
    'image': torch.Tensor,  # Shape: (3, 384, 384), Range: normalized
    'mask': torch.Tensor,   # Shape: (1, 384, 384), Range: {0, 1}
    'image_id': str         # e.g., "ISIC_0000000"
}
```

### Batch Processing
```python
for batch in train_loader:
    images = batch['image']    # Shape: (B, 3, 384, 384)
    masks = batch['mask']      # Shape: (B, 1, 384, 384)  
    ids = batch['image_id']    # List of strings
    
    # Your training code here
    predictions = model(images)
    loss = criterion(predictions, masks)
```

## 📊 Dataset Statistics

### Split Distribution
```
Training:   2,293 samples (88.4%)
Validation:   301 samples (11.6%)
Test:       1,000 samples (no masks)
Total:      3,594 samples
```

### Lesion Area Distribution
```
Training Split:
  Mean lesion area: 36.38% of image
  Median: 33.30%
  Range: 3.01% - 82.99%
  Std: 18.91%

Validation Split:
  Mean lesion area: 35.23% of image
  Median: 29.11%  
  Range: 4.33% - 99.53%
  Std: 26.27%
```

### Performance Metrics
```
Loading Speed (16 batch size):
  Training (with augmentations): ~0.4s per batch
  Validation (no augmentations): ~0.2s per batch
  
Memory Usage:
  Per batch (16 samples): ~50MB
  Full dataset: ~1-2GB
```

## 🔧 Configuration Options

### Dataset Parameters
```python
ISIC2018Dataset(
    data_dir="data/ISIC2018_proc",     # Processed data directory
    split="train",                      # "train", "val", or "test"
    splits_dir="splits",               # Directory with split files
    image_size=384,                    # Target image size
    augment=None,                      # Auto: True for train, False for val/test
    stats_file=None                    # Auto: uses dataset_stats.json
)
```

### Data Loader Parameters
```python
create_data_loaders(
    data_dir="data/ISIC2018_proc",
    splits_dir="splits", 
    batch_size=16,
    num_workers=4,
    image_size=384,
    pin_memory=True
)
```

## 🎨 Visualization & Analysis

### Sample Visualization
```python
from scripts.dataset import visualize_samples

# Visualize dataset samples
visualize_samples(
    dataset=train_dataset,
    num_samples=4,
    save_path="dataset_samples.png"
)
```

### Augmentation Demonstration
```python
# Run comprehensive demo
python demo_dataset.py

# Generates:
# - augmentation_demonstration.png: Shows augmentation effects
# - dataset_statistics.png: Lesion area distribution
```

### Dataset Analysis
```python
from scripts.dataset import analyze_dataset_splits

# Print comprehensive dataset information
analyze_dataset_splits(
    data_dir="data/ISIC2018_proc",
    splits_dir="splits"
)
```

## 🔄 Data Pipeline Flow

```
Raw ISIC2018 → Preprocessing → Split Files → Dataset → DataLoader → Model
     ↓              ↓             ↓           ↓          ↓
   Various      384×384×3     train.txt   Augmented   Batched
   sizes         PNG         val.txt     samples     tensors
                Binary        test.txt    Binary      Binary
                masks         (2293)      masks       masks
                             (301)       {0,1}       {0,1}
```

## ⚙️ Implementation Details

### Binary Mask Enforcement
```python
def _binarize_mask(self, mask):
    """Ensure mask contains only {0, 1} values."""
    if mask.max() > 1:
        mask = mask / 255.0  # Handle 0-255 range
    return (mask > 0.5).astype(np.uint8)  # Threshold at 0.5
```

### Synchronized Transformations
```python
# Albumentations ensures image and mask are transformed identically
transformed = self.transforms(image=image, mask=mask)
# Both image and mask undergo same geometric transformations
# Only image gets photometric transformations
```

### Mask Interpolation
- **Geometric transforms**: Nearest-neighbor interpolation for masks
- **Preserves binary values**: No interpolation artifacts
- **Synchronized**: Same transform applied to image and mask

## 🧪 Validation & Testing

### Quality Checks
```python
# Verify binary masks
assert torch.all((sample['mask'] == 0) | (sample['mask'] == 1))

# Check dimensions
assert sample['image'].shape == (3, 384, 384)
assert sample['mask'].shape == (1, 384, 384)

# Verify normalization
mean_val = sample['image'].mean()  # Should be around 0 after normalization
```

### Performance Testing
```python
# Run speed tests
python demo_dataset.py

# Expected performance:
# - Training batches: ~0.4s each (with augmentations)
# - Validation batches: ~0.2s each (minimal processing)
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure albumentations is installed
pip install albumentations

# Check version compatibility
pip list | grep albumentations  # Should be >= 2.0.8
```

**Memory Issues**
```python
# Reduce batch size
data_loaders = create_data_loaders(batch_size=8)

# Reduce num_workers
data_loaders = create_data_loaders(num_workers=0)
```

**Slow Loading**
```python
# Increase num_workers (if not on Windows)
data_loaders = create_data_loaders(num_workers=4)

# Enable pin_memory for GPU training
data_loaders = create_data_loaders(pin_memory=True)
```

### Validation Commands
```python
# Test dataset integrity
python -c "
from scripts.dataset import ISIC2018Dataset
dataset = ISIC2018Dataset('train')
print(f'✅ Dataset loaded: {len(dataset)} samples')
"

# Verify split files
ls -la splits/
# Should show: isic2018_train.txt, isic2018_val.txt

# Check processed data
ls -la data/ISIC2018_proc/
# Should show: train_images/, train_masks/, test_images/, dataset_stats.json
```

## 📈 Research Integration

### Training Loop Integration
```python
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    for batch in train_loader:
        images, masks = batch['image'].cuda(), batch['mask'].cuda()
        
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Evaluation Integration
```python
def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch['image'].cuda(), batch['mask'].cuda()
            predictions = model(images)
            
            # Threshold predictions at 0.5 for binary evaluation
            pred_binary = (predictions > 0.5).float()
            
            # Compute metrics
            iou = compute_iou(pred_binary, masks)
```

## 📚 References

- **ISIC 2018 Challenge**: [Skin Lesion Analysis Toward Melanoma Detection](https://challenge2018.isic-archive.com/)
- **Albumentations**: [Fast Image Augmentation Library](https://albumentations.ai/)
- **PyTorch Datasets**: [Custom Dataset Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

---

**Status**: ✅ **Production Ready**  
**Last Updated**: August 23, 2025  
**Version**: 1.0
