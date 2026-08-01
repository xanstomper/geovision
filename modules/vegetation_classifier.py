import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class VegetationClassifier:
    def __init__(self):
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for VegetationClassifier.")
            
    def classify(self, image_path: str) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {'status': 'unavailable'}
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'status': 'error'}
                
            # Convert to float for indices
            img_float = img.astype(float) / 255.0
            B, G, R = cv2.split(img_float)
            
            # VARI (Visible Atmospherically Resistant Index): (G-R)/(G+R-B)
            denom = (G + R - B)
            denom[denom == 0] = 1e-6
            vari = (G - R) / denom
            
            # Green fraction
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_fraction = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
            
            density = 'none'
            if green_fraction > 0.6: density = 'forest'
            elif green_fraction > 0.3: density = 'dense'
            elif green_fraction > 0.1: density = 'moderate'
            elif green_fraction > 0.02: density = 'sparse'
            
            # Climate estimation based on HSV saturation/value in green regions
            climate = 'unknown'
            if green_fraction > 0.1:
                green_pixels = hsv[green_mask > 0]
                avg_sat = np.mean(green_pixels[:, 1])
                if avg_sat > 180: climate = 'tropical'
                elif avg_sat < 100: climate = 'arid'
                else: climate = 'temperate'
                
            return {
                'status': 'success',
                'vegetation_density': density,
                'estimated_climate': climate,
                'vegetation_indices': {
                    'vari_mean': float(np.mean(vari)),
                    'green_fraction': float(green_fraction)
                },
                'confidence': float(green_fraction) if green_fraction > 0 else 0.1
            }
            
        except Exception as e:
            logger.error(f"Vegetation classification failed: {e}")
            return {'status': 'error', 'error': str(e)}
