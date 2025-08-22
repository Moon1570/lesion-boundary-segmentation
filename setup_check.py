#!/usr/bin/env python3
"""
Quick setup script for the lesion boundary segmentation project.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 12):
        print("❌ Python 3.12 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_virtual_environment():
    """Check if running in virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment (recommended to use .venv)")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    # Package mapping: import name -> package name
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'pillow'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_data_structure():
    """Check if data directory structure exists."""
    data_dir = Path("data/ISIC2018")
    expected_dirs = ["train_images", "train_masks", "val_images", "test_images"]
    
    if not data_dir.exists():
        print("❌ Data directory not found: data/ISIC2018")
        print("Please download and extract ISIC2018 dataset")
        return False
    
    missing_dirs = []
    for dir_name in expected_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✅ {dir_name}: {file_count} files")
        else:
            print(f"❌ {dir_name}: not found")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    
    return True


def run_test_preprocessing():
    """Run test preprocessing to validate setup."""
    print("\n🧪 Running test preprocessing...")
    try:
        result = subprocess.run([
            sys.executable, "scripts/test_preprocessing.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Test preprocessing completed successfully")
            return True
        else:
            print("❌ Test preprocessing failed")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Test preprocessing timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test preprocessing: {e}")
        return False


def main():
    """Main setup validation function."""
    print("🚀 Lesion Boundary Segmentation - Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("Data Structure", check_data_structure),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Running test preprocessing...")
        if run_test_preprocessing():
            print("\n✅ Setup complete! You're ready to start preprocessing.")
            print("\nNext steps:")
            print("  1. Run full preprocessing: python scripts/preprocess.py")
            print("  2. Analyze results: python scripts/dataset_info.py --compare")
        else:
            print("\n⚠️  Setup validation failed. Please check the error messages above.")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above and run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
