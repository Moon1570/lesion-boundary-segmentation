import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import cv2
import os
from pathlib import Path
import json

# Load processed data from your existing pipeline
def load_isic_data(data_dir="../data/ISIC2018_proc", splits_dir="../splits"):
    """Load ISIC2018 data using your existing preprocessing pipeline"""
    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    
    # Load train/val splits
    train_split_file = splits_dir / "isic2018_train.txt"
    val_split_file = splits_dir / "isic2018_val.txt"
    
    if train_split_file.exists() and val_split_file.exists():
        with open(train_split_file, 'r') as f:
            train_ids = [line.strip() for line in f.readlines()]
        with open(val_split_file, 'r') as f:
            val_ids = [line.strip() for line in f.readlines()]
    else:
        # Fallback: load all training images and create split
        train_img_dir = data_dir / "train_images"
        all_ids = [f.stem for f in train_img_dir.glob("*.png")]
        train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    
    return train_ids, val_ids

# Load data splits
train_ids, val_ids = load_isic_data()

class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for imgCount in range(len(imgs)):
            imgs[imgCount] = transforms.functional.resized_crop(imgs[imgCount], i, j, h, w, self.size, self.interpolation)
        return imgs
class ISICLoader(Dataset):
    def __init__(self, image_ids, data_dir="../data/ISIC2018_proc", transform=True, typeData="train"):
        self.image_ids = image_ids
        self.data_dir = Path(data_dir)
        self.transform = transform if typeData == "train" else False  # augment data bool
        self.typeData = typeData
        
        # Set image and mask directories
        if typeData == "train":
            self.img_dir = self.data_dir / "train_images"
            self.mask_dir = self.data_dir / "train_masks"
        elif typeData == "val":
            self.img_dir = self.data_dir / "train_images"  # Val images are in train_images
            self.mask_dir = self.data_dir / "train_masks"  # Val masks are in train_masks
        else:  # test
            self.img_dir = self.data_dir / "test_images"
            self.mask_dir = None  # No masks for test
            
        # Load dataset statistics for normalization
        stats_file = self.data_dir / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.mean = stats.get('mean', 0.6042)
                self.std = stats.get('std', 0.1817)
        else:
            self.mean = 0.6042
            self.std = 0.1817
            
    def __len__(self):
        return len(self.image_ids)

    def rotate(self, image, mask, degrees=(-15,15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            if mask is not None:
                mask = mask.rotate(degree, Image.NEAREST)
        return image, mask
        
    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
        
    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask
        
    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            if mask is not None:
                image, mask = RandomCrop((192, 256), scale=(0.8, 0.95))([image, mask])
            else:
                image = RandomCrop((192, 256), scale=(0.8, 0.95))(image)
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.img_dir / f"{image_id}.png"
        image = Image.open(img_path).convert('RGB')
        
        # Load mask if available
        mask = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{image_id}.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
        
        # Apply augmentations
        if self.transform and mask is not None:
            image, mask = self.augment(image, mask)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Normalize using dataset statistics
        image = transforms.Normalize(mean=[self.mean], std=[self.std])(image)
        
        if mask is not None:
            mask = np.asarray(mask, np.float32) / 255.0  # Normalize to [0, 1]
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            return image, mask
        else:
            return image

# Create datasets and dataloaders
train_dataset_obj = ISICLoader(train_ids, typeData="train")
val_dataset_obj = ISICLoader(val_ids, typeData="val", transform=False)

train_dataset = DataLoader(train_dataset_obj, batch_size=4, pin_memory=True, shuffle=True, num_workers=2, drop_last=True)
test_dataset = DataLoader(val_dataset_obj, batch_size=1, num_workers=2)