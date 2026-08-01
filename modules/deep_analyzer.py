"""
Deep Learning Analyzer
Uses neural networks for advanced feature extraction
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class DeepAnalyzer:
    """
    Deep learning-based image analysis for geolocation
    Uses pre-trained CNNs to extract high-level features
    """
    
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.feature_dim = 512
        
        if TORCH_AVAILABLE:
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Model load error: {e}")
                self.model = None
    
    def _load_model(self):
        try:
            resnet = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
            self.model.eval()
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            logger.warning(f"Model load error: {e}")
            self.model = None
    
    def extract_features(self, image) -> np.ndarray:
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(self.feature_dim)
        
        try:
            from PIL import Image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(tensor)
                features = features.flatten()
                
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim)
    
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return float(np.clip((similarity + 1) / 2, 0, 1))
