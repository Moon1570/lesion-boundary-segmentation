# ISIC2018 Dataset Preprocessing

This module provides comprehensive preprocessing functionality for the ISIC2018 lesion boundary segmentation dataset, implementing deterministic preprocessing pipelines as described in the research methodology.

## Features

### 🔄 Canonical Preprocessing
- **Resize Strategy**: Shorter side → 384px while preserving aspect ratio
- **Padding**: Center-pad to 384×384 square format
- **Format**: Save as PNG (lossless compression)
- **Mask Handling**: Nearest-neighbor interpolation for segmentation masks

### 📊 Normalization Pipeline
- **Per-image**: Min-max normalization to [0,1]
- **Dataset-level**: Standardization using computed mean/std from training set
- **Statistics**: Automatically computed and cached for reproducibility

### 🧹 Optional Hair Removal
- **DullRazor Implementation**: Classical morphological approach
- **Method**: Morphological closing + bilinear inpainting
- **Reference**: Widely cited method in dermoscopic image analysis
- **Usage**: Optional `--hair-removal dullrazor` flag

## Quick Start

### Basic Preprocessing
```bash
# Standard preprocessing without hair removal
python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc

# With custom target size
python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc --target_size 512
```

### With Hair Removal (Ablation Study)
```bash
# DullRazor hair removal for ablation analysis
python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc --hair-removal dullrazor
```

### Helper Scripts
```bash
# Test on small subset (50 images)
python scripts/test_preprocessing.py

# Test with hair removal
python scripts/test_preprocessing.py --hair-removal

# Run full preprocessing
python scripts/run_preprocessing.py
python scripts/run_preprocessing.py --hair-removal

# Demo preprocessing pipeline
python scripts/demo_preprocess.py
```

## Input/Output Structure

### Input Directory Structure
```
data/ISIC2018/
├── train_images/          # Training images (.jpg)
├── train_masks/           # Training masks (.png) 
├── val_images/            # Validation images (.jpg)
└── test_images/           # Test images (.jpg)
```

### Output Directory Structure
```
data/ISIC2018_proc/
├── train_images/          # Processed training images (.png)
├── train_masks/           # Processed training masks (.png)
├── val_images/            # Processed validation images (.png)
├── test_images/           # Processed test images (.png)
└── dataset_stats.json     # Dataset normalization statistics
```

## Technical Details

### Canonical Resizing Algorithm
1. **Scale Calculation**: `scale = target_size / min(height, width)`
2. **Resize**: Apply scale to both dimensions
3. **Crop or Pad**: If larger than target → center crop, if smaller → center pad
4. **Interpolation**: Linear for images, nearest-neighbor for masks

### Dataset Statistics
The preprocessing computes dataset-level statistics from training images:
```json
{
  "mean": 0.6016,
  "std": 0.2069
}
```

### DullRazor Hair Removal
Implementation details:
- **Hair Detection**: Black top-hat morphological operation
- **Kernel Sizes**: Elliptical kernels (17×17, 11×11, 7×7)
- **Threshold**: Binary threshold at value 10
- **Inpainting**: OpenCV TELEA algorithm for hair region filling

## Performance

### Processing Speed
- **Without Hair Removal**: ~30-35 images/second
- **With Hair Removal**: ~2-14 images/second (depends on image complexity)
- **Memory Usage**: ~1-2GB for full dataset processing

### Output Sizes
- **Original Images**: Various sizes (typically ~767×1022)
- **Processed Images**: Fixed 384×384×3
- **Processed Masks**: Fixed 384×384

## Command Line Options

```bash
python scripts/preprocess.py [OPTIONS]

Options:
  --input_dir PATH          Input directory containing raw ISIC2018 data
                           [default: data/ISIC2018]
  
  --output_dir PATH         Output directory for processed data  
                           [default: data/ISIC2018_proc]
  
  --target_size INT         Target image size (square)
                           [default: 384]
  
  --hair-removal {dullrazor}  Apply hair removal method
                           [default: None]
  
  --recompute_stats         Recompute dataset statistics even if they exist
                           [default: False]
  
  --help                    Show help message and exit
```

## Dependencies

Required packages (from `requirements.txt`):
- `opencv-python>=4.12.0`
- `numpy>=2.1.2`
- `pillow>=11.0.0`
- `tqdm>=4.67.1`

## Reproducibility

- **Deterministic**: All operations are deterministic for reproducible results
- **Cached Statistics**: Dataset statistics are computed once and cached
- **Version Control**: All processed data maintains traceability to preprocessing parameters

## Validation

The preprocessing includes built-in validation:
1. **Demo Mode**: Visualizes preprocessing steps on sample images
2. **Test Mode**: Processes small subset for validation
3. **Dimension Checks**: Ensures consistent output dimensions
4. **File Integrity**: Validates input/output file correspondences

## References

- **DullRazor**: Lee, T., et al. "DullRazor: A software approach to hair removal from images." Computers in Biology and Medicine, 1997.
- **Morphological Operations**: Serra, J. "Image Analysis and Mathematical Morphology." Academic Press, 1982.
- **ISIC Dataset**: Codella, N., et al. "Skin lesion analysis toward melanoma detection." ISIC 2018 Challenge.

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or process splits individually
2. **Slow Hair Removal**: Consider skipping for initial experiments
3. **Missing Masks**: Check train_masks directory structure and naming convention
4. **Permission Errors**: Ensure write permissions for output directory

### Performance Optimization

- Use SSD storage for faster I/O
- Ensure sufficient RAM (≥8GB recommended)
- Consider parallel processing for large datasets
- Monitor disk space (processed data ~2-3x larger than original)

## Future Enhancements

- [ ] **SharpRazor**: Modern alternative to DullRazor
- [ ] **Multi-scale Processing**: Support for different target sizes
- [ ] **Data Augmentation**: Integration with augmentation pipeline
- [ ] **Parallel Processing**: Multi-threading support for faster processing
