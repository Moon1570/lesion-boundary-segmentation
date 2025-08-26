"""
Advanced data augmentation for medical image segmentation
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2


class MedicalAugmentation:
    """Advanced medical image augmentation"""
    
    def __init__(self, config):
        self.rotation_angle = config.get('rotation', 15)
        self.flip_prob = config.get('flip_prob', 0.5)
        self.brightness = config.get('brightness', 0.1)
        self.contrast = config.get('contrast', 0.1)
        self.hue = config.get('hue', 0.05)
        self.saturation = config.get('saturation', 0.1)
        self.elastic_transform = config.get('elastic_transform', True)
        self.gaussian_noise = config.get('gaussian_noise', 0.01)
        
    def __call__(self, image, mask):
        # Random rotation
        if random.random() < 0.7:
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Color jittering
        if random.random() < 0.8:
            image = self._color_jitter(image)
        
        # Elastic transformation
        if self.elastic_transform and random.random() < 0.3:
            image, mask = self._elastic_transform(image, mask)
        
        # Gaussian noise
        if random.random() < 0.3:
            image = self._add_gaussian_noise(image)
        
        # Random crop and resize
        if random.random() < 0.5:
            image, mask = self._random_crop_resize(image, mask)
        
        return image, mask
    
    def _color_jitter(self, image):
        """Apply color jittering"""
        if random.random() < 0.8:
            brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if random.random() < 0.8:
            contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
            image = TF.adjust_contrast(image, contrast_factor)
        
        if random.random() < 0.8:
            saturation_factor = 1 + random.uniform(-self.saturation, self.saturation)
            image = TF.adjust_saturation(image, saturation_factor)
        
        if random.random() < 0.8:
            hue_factor = random.uniform(-self.hue, self.hue)
            image = TF.adjust_hue(image, hue_factor)
        
        return image
    
    def _elastic_transform(self, image, mask, alpha=50, sigma=5):
        """Apply elastic transformation"""
        # Convert to numpy
        img_np = np.array(image)
        mask_np = np.array(mask)
        
        shape = img_np.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur(np.random.randn(*shape) * alpha, (0, 0), sigma)
        dy = cv2.GaussianBlur(np.random.randn(*shape) * alpha, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        
        # Apply transformation
        img_transformed = cv2.remap(img_np, indices[1], indices[0], cv2.INTER_LINEAR)
        mask_transformed = cv2.remap(mask_np, indices[1], indices[0], cv2.INTER_NEAREST)
        
        return Image.fromarray(img_transformed), Image.fromarray(mask_transformed)
    
    def _add_gaussian_noise(self, image):
        """Add Gaussian noise"""
        img_np = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(0, self.gaussian_noise, img_np.shape)
        img_noisy = np.clip(img_np + noise, 0, 1)
        return Image.fromarray((img_noisy * 255).astype(np.uint8))
    
    def _random_crop_resize(self, image, mask, scale=(0.8, 1.0)):
        """Random crop and resize"""
        w, h = image.size
        scale_factor = random.uniform(*scale)
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        image = image.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))
        
        # Resize back to original size
        image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        
        return image, mask


class TestTimeAugmentation:
    """Test Time Augmentation for inference"""
    
    def __init__(self, scales=[0.9, 1.0, 1.1], flips=[False, True]):
        self.scales = scales
        self.flips = flips
    
    def __call__(self, model, image):
        predictions = []
        
        for scale in self.scales:
            for flip in self.flips:
                # Scale image
                h, w = image.shape[-2:]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = torch.nn.functional.interpolate(
                    image, size=(new_h, new_w), mode='bilinear', align_corners=True
                )
                
                # Flip if needed
                if flip:
                    scaled_img = torch.flip(scaled_img, dims=[-1])
                
                # Predict
                with torch.no_grad():
                    pred = model(scaled_img)
                
                # Resize back
                pred = torch.nn.functional.interpolate(
                    pred, size=(h, w), mode='bilinear', align_corners=True
                )
                
                # Flip back if needed
                if flip:
                    pred = torch.flip(pred, dims=[-1])
                
                predictions.append(pred)
        
        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)
