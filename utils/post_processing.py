"""
Advanced post-processing techniques for segmentation improvement
"""
import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage import morphology, measure
from skimage.filters import gaussian


class SegmentationPostProcessor:
    """Advanced post-processing for segmentation masks"""
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'remove_small_objects': True,
                'min_object_size': 100,
                'hole_filling': True,
                'morphological_closing': True,
                'gaussian_smoothing': True,
                'contour_smoothing': True,
                'confidence_threshold': 0.5
            }
        self.config = config
    
    def __call__(self, prediction, confidence_map=None):
        """Apply post-processing pipeline"""
        # Convert to binary mask
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        
        # Apply confidence threshold
        binary_mask = prediction > self.config['confidence_threshold']
        
        # Gaussian smoothing of prediction
        if self.config.get('gaussian_smoothing', True):
            prediction = gaussian(prediction, sigma=1.0)
            binary_mask = prediction > self.config['confidence_threshold']
        
        # Remove small objects
        if self.config.get('remove_small_objects', True):
            binary_mask = morphology.remove_small_objects(
                binary_mask, 
                min_size=self.config.get('min_object_size', 100)
            )
        
        # Fill holes
        if self.config.get('hole_filling', True):
            binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Morphological closing
        if self.config.get('morphological_closing', True):
            kernel = morphology.disk(3)
            binary_mask = morphology.binary_closing(binary_mask, kernel)
        
        # Contour smoothing
        if self.config.get('contour_smoothing', True):
            binary_mask = self._smooth_contours(binary_mask)
        
        return binary_mask.astype(np.float32)
    
    def _smooth_contours(self, binary_mask):
        """Smooth contours using morphological operations"""
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create smoothed mask
        smoothed_mask = np.zeros_like(binary_mask)
        
        for contour in contours:
            # Approximate contour to smooth it
            epsilon = 0.002 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Fill the smoothed contour
            cv2.fillPoly(smoothed_mask, [smoothed_contour], 1)
        
        return smoothed_mask


class EnsemblePostProcessor:
    """Ensemble multiple model predictions with post-processing"""
    
    def __init__(self, models, weights=None, post_processor=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.post_processor = post_processor or SegmentationPostProcessor()
    
    def predict(self, image):
        """Ensemble prediction with post-processing"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(image)
                predictions.append(torch.sigmoid(pred))
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        total_weight = sum(self.weights)
        
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * (weight / total_weight)
        
        # Apply post-processing
        return self.post_processor(ensemble_pred)


class ConditionalRandomField:
    """CRF post-processing for refined segmentation"""
    
    def __init__(self, bilateral_weight=10, gaussian_weight=3):
        self.bilateral_weight = bilateral_weight
        self.gaussian_weight = gaussian_weight
    
    def __call__(self, image, prediction):
        """Apply CRF refinement (requires pydensecrf)"""
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            print("Warning: pydensecrf not installed. Skipping CRF post-processing.")
            return prediction
        
        # Convert to required format
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        
        h, w = prediction.shape[-2:]
        
        # Prepare unary potentials
        prob_fg = prediction.flatten()
        prob_bg = 1 - prob_fg
        unary = np.stack([prob_bg, prob_fg])
        unary = -np.log(unary + 1e-8)
        
        # Setup CRF
        d = dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(unary.reshape(2, -1).astype(np.float32))
        
        # Add pairwise potentials
        d.addPairwiseGaussian(sxy=self.gaussian_weight, compat=1)
        d.addPairwiseBilateral(
            sxy=self.bilateral_weight, 
            srgb=13, 
            rgbim=image.astype(np.uint8), 
            compat=1
        )
        
        # Inference
        Q = d.inference(5)
        refined = np.argmax(Q, axis=0).reshape(h, w)
        
        return refined.astype(np.float32)
