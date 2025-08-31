# Lesion Boundary Segmentation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.12.0+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Paper title: Lightweight Ensembles and Reproducible Baselines for Skin Lesion Segmentation: A Study of U-Net Variants on ISIC-2018

A deep learning framework for accurate segmentation of skin lesion boundaries in dermoscopic images using the ISIC2018 dataset. This project implements state-of-the-art preprocessing techniques and provides a comprehensive pipeline for lesion boundary analysis.

## 🎯 Project Overview

This repository contains the implementation for skin lesion boundary segmentation, a critical task in automated dermoscopic image analysis for melanoma detection. The project focuses on:

- **Deterministic Preprocessing**: Canonical resizing, normalization, and optional hair removal
- **Robust Data Pipeline**: Handles ISIC2018 dataset with comprehensive validation
- **Reproducible Results**: Standardized preprocessing with cached statistics
- **Ablation Studies**: Optional DullRazor hair removal for comparative analysis

## 📊 Dataset

**ISIC2018 Challenge Dataset**
- **Training Images**: 2,594 dermoscopic images
- **Training Masks**: 2,594 corresponding segmentation masks
- **Validation Images**: 100 images
- **Test Images**: 1,000 images
- **Original Size**: Variable (typically 767×1022 to 1944×2592)
- **Processed Size**: 384×384 pixels (canonical)

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/Moon1570/lesion-boundary-segmentation.git
cd lesion-boundary-segmentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step-by-Step Setup

1. **Environment Setup**
   ```bash
   # Ensure Python 3.12+ is installed
   python --version
   
   # Activate virtual environment
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. **Data Preparation**
   ```bash
   # Download ISIC2018 dataset from official source
   # Place data in the following structure:
   mkdir -p data/ISIC2018/{train_images,train_masks,val_images,test_images}
   ```

3. **Verify Dataset**
   ```bash
   # Check dataset integrity
   python scripts/dataset_info.py --raw_dir data/ISIC2018
   ```

4. **Test Preprocessing**
   ```bash
   # Run on small subset first
   python scripts/test_preprocessing.py
   ```

5. **Full Preprocessing**
   ```bash
   # Process complete dataset
   python scripts/preprocess.py
   ```

6. **Validate Setup** (Optional)
   ```bash
   # Run comprehensive setup validation
   python setup_check.py
   ```

### Data Setup
1. Download ISIC2018 dataset and place in `data/ISIC2018/`
2. Ensure the following structure:
```
data/ISIC2018/
├── train_images/    # 2594 .jpg files
├── train_masks/     # 2594 .png files  
├── val_images/      # 100 .jpg files
└── test_images/     # 1000 .jpg files
```

### Preprocessing

#### Standard Preprocessing
```bash
# Basic preprocessing (recommended)
python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc

# Test on small subset first
python scripts/test_preprocessing.py
```

#### With Hair Removal (Ablation Study)
```bash
# DullRazor hair removal for comparative analysis
python scripts/preprocess.py --input_dir data/ISIC2018 --output_dir data/ISIC2018_proc --hair-removal dullrazor

# Test hair removal on subset
python scripts/test_preprocessing.py --hair-removal
```

#### Advanced Options
```bash
# Custom target size
python scripts/preprocess.py --target_size 512

# Recompute dataset statistics
python scripts/preprocess.py --recompute_stats

# Full help
python scripts/preprocess.py --help
```

## 📁 Project Structure

```
lesion-boundary-segmentation/
├── data/
│   ├── ISIC2018/              # Raw dataset
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── val_images/
│   │   └── test_images/
│   └── ISIC2018_proc/         # Processed dataset
│       ├── train_images/      # 384×384 PNG images
│       ├── train_masks/       # 384×384 PNG masks
│       ├── val_images/
│       ├── test_images/
│       └── dataset_stats.json # Normalization statistics
├── scripts/
│   ├── preprocess.py          # Main preprocessing script
│   ├── test_preprocessing.py  # Test on subset
│   ├── demo_preprocess.py     # Visual demonstration
│   ├── run_preprocessing.py   # Simple wrapper
│   ├── dataset_info.py        # Dataset analysis
│   └── README_preprocessing.md # Detailed preprocessing docs
├── runs/
│   ├── ckpts/                 # Model checkpoints
│   ├── figs/                  # Training figures
│   ├── preds/                 # Predictions
│   └── scripts/               # Training scripts
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔬 Preprocessing Pipeline

### Canonical Preprocessing
1. **Resize Strategy**: Shorter side → 384px (preserves aspect ratio)
2. **Padding**: Center-pad to 384×384 square format
3. **Normalization**: 
   - Per-image min-max to [0,1]
   - Dataset standardization (mean: 0.6016, std: 0.2069)
4. **Format**: PNG lossless compression
5. **Masks**: Nearest-neighbor interpolation

### DullRazor Hair Removal (Optional)
- **Method**: Morphological closing + bilinear inpainting
- **Implementation**: Classical approach using OpenCV
- **Performance**: ~2-14 images/second (vs ~30-35 without)
- **Use Case**: Ablation studies to assess hair impact

### Dataset Statistics
```json
{
  "mean": 0.6016,
  "std": 0.2069
}
```

## � Expected Output

After successful preprocessing, you should have:

### File Structure
```
data/ISIC2018_proc/
├── train_images/          # 2,594 PNG files (384×384×3)
├── train_masks/           # 2,594 PNG files (384×384×1)
├── val_images/            # 100 PNG files (384×384×3)
├── test_images/           # 1,000 PNG files (384×384×3)
└── dataset_stats.json     # Normalization statistics
```

### Verification Checklist
- [ ] All images are 384×384 pixels
- [ ] PNG format (lossless compression)
- [ ] Dataset statistics file exists
- [ ] File counts match original dataset
- [ ] No corrupted or missing files

### Sample Validation
```bash
# Quick verification commands
find data/ISIC2018_proc/train_images -name "*.png" | wc -l  # Should be 2594
find data/ISIC2018_proc/train_masks -name "*.png" | wc -l   # Should be 2594
find data/ISIC2018_proc/val_images -name "*.png" | wc -l    # Should be 100
find data/ISIC2018_proc/test_images -name "*.png" | wc -l   # Should be 1000
```

## �🛠️ Utility Scripts

### Dataset Analysis
```bash
# Analyze raw dataset
python scripts/dataset_info.py --raw_dir data/ISIC2018

# Analyze processed dataset
python scripts/dataset_info.py --processed_dir data/ISIC2018_proc

# Compare raw vs processed
python scripts/dataset_info.py --compare
```

### Preprocessing Demo
```bash
# Visual demonstration of preprocessing steps
python scripts/demo_preprocess.py
```

### Performance Testing
```bash
# Test preprocessing speed and accuracy
python scripts/test_preprocessing.py
```

## 📊 Performance Metrics

### Preprocessing Speed
- **Standard Pipeline**: 30-35 images/second
- **With Hair Removal**: 2-14 images/second
- **Memory Usage**: 1-2GB for full dataset
- **Storage**: Processed data ~2-3x larger than original

### Dataset Coverage
- **Training**: 2,594/2,594 images (100%)
- **Validation**: 100/100 images (100%)
- **Test**: 1,000/1,000 images (100%)
- **Masks**: 2,594/2,594 masks (100%)

## � Usage Examples

### Basic Workflow
```bash
# 1. Test preprocessing on small subset
python scripts/test_preprocessing.py

# 2. Run full preprocessing
python scripts/preprocess.py

# 3. Analyze results
python scripts/dataset_info.py --compare
```

### Ablation Study Workflow
```bash
# 1. Standard preprocessing
python scripts/preprocess.py --output_dir data/ISIC2018_proc_standard

# 2. With hair removal
python scripts/preprocess.py --output_dir data/ISIC2018_proc_hairfree --hair-removal dullrazor

# 3. Compare both approaches
python scripts/dataset_info.py --processed_dir data/ISIC2018_proc_standard
python scripts/dataset_info.py --processed_dir data/ISIC2018_proc_hairfree
```

### Custom Configuration
```bash
# Different target size
python scripts/preprocess.py --target_size 512 --output_dir data/ISIC2018_proc_512

# Recompute statistics
python scripts/preprocess.py --recompute_stats
```

## �🔧 Configuration

### Command Line Options
```bash
python scripts/preprocess.py [OPTIONS]

Options:
  --input_dir PATH          Input directory [default: data/ISIC2018]
  --output_dir PATH         Output directory [default: data/ISIC2018_proc]
  --target_size INT         Target image size [default: 384]
  --hair-removal {dullrazor} Hair removal method [default: None]
  --recompute_stats         Recompute dataset statistics
  --help                    Show help message
```

### Dependencies
```
opencv-python>=4.12.0      # Image processing
numpy>=2.1.2               # Numerical computations
pillow>=11.0.0             # Image I/O
torch>=2.5.1               # Deep learning framework
torchvision>=0.20.1        # Computer vision utilities
albumentations>=2.0.8      # Data augmentation
matplotlib>=3.10.5         # Visualization
tqdm>=4.67.1               # Progress bars
scikit-image>=0.25.2       # Image processing
```

## 🧪 Validation & Testing

### Automated Testing
```bash
# Run preprocessing tests
python scripts/test_preprocessing.py

# Validate output dimensions
python scripts/dataset_info.py --compare

# Demo preprocessing pipeline
python scripts/demo_preprocess.py
```

### Quality Assurance
- ✅ **Dimension Consistency**: All outputs are 384×384
- ✅ **File Integrity**: Input-output correspondence validated
- ✅ **Mask Alignment**: Segmentation masks properly processed
- ✅ **Statistics Caching**: Reproducible normalization parameters
- ✅ **Error Handling**: Robust file processing with validation

## � Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Process smaller batches or use test mode first
python scripts/test_preprocessing.py
```

**Missing Files**
```bash
# Check dataset structure
python scripts/dataset_info.py --raw_dir data/ISIC2018
```

**Slow Processing**
```bash
# Skip hair removal for faster processing
python scripts/preprocess.py  # without --hair-removal flag
```

**Permission Errors**
```bash
# Ensure write permissions for output directory
mkdir -p data/ISIC2018_proc
chmod 755 data/ISIC2018_proc
```

### Validation Commands
```bash
# Verify preprocessing worked correctly
python scripts/dataset_info.py --compare

# Check specific split
ls -la data/ISIC2018_proc/train_images/ | wc -l  # Should show 2594

# Validate image dimensions
python -c "import cv2; img=cv2.imread('data/ISIC2018_proc/train_images/ISIC_0000000.png'); print(img.shape)"
# Should output: (384, 384, 3)
```

## �📈 Research Context

### Methodology
This preprocessing pipeline implements best practices for dermoscopic image analysis:

1. **Canonical Sizing**: Ensures consistent input dimensions for deep learning models
2. **Aspect Ratio Preservation**: Maintains image proportions during resizing
3. **Lossless Storage**: PNG format preserves image quality for training
4. **Standardized Normalization**: Dataset-level statistics for consistent training

### Ablation Studies
The optional hair removal functionality enables:
- **Comparative Analysis**: With/without hair removal performance
- **Method Validation**: Classical DullRazor vs modern approaches
- **Impact Assessment**: Hair artifact influence on segmentation accuracy

## 📚 References

- **ISIC 2018**: Codella, N., et al. "Skin lesion analysis toward melanoma detection: A challenge at the 2018 international symposium on biomedical imaging (ISBI)"
- **DullRazor**: Lee, T., et al. "DullRazor: A software approach to hair removal from images." Computers in Biology and Medicine, 1997
- **Dermoscopy**: Argenziano, G., et al. "Dermoscopy of pigmented skin lesions: results of a consensus meeting"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ISIC Archive**: International Skin Imaging Collaboration
- **Challenge Organizers**: ISIC 2018 Challenge contributors
- **Research Community**: Dermoscopic image analysis researchers

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **Project Link**: [https://github.com/Moon1570/lesion-boundary-segmentation](https://github.com/Moon1570/lesion-boundary-segmentation)

---

**Note**: This project is part of a master's thesis research on lesion boundary segmentation. For detailed preprocessing documentation, see `scripts/README_preprocessing.md`.