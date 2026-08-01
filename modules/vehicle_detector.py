import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class VehicleDetector:
    def __init__(self):
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available.")
            return
            
    def detect_vehicles(self, image_path: str) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {'status': 'unavailable'}
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'status': 'error', 'error': 'Could not read image'}
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Simple contour based fallback
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicles = []
            height, width = img.shape[:2]
            left_count = 0
            right_count = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1000 < area < 20000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    
                    # Rough heuristic for vehicles
                    if 0.5 < aspect_ratio < 2.5:
                        v_type = 'sedan'
                        if h > w: v_type = 'truck/bus'
                        elif aspect_ratio > 1.8: v_type = 'SUV'
                            
                        center_x = x + w/2
                        if center_x < width/2:
                            left_count += 1
                        else:
                            right_count += 1
                            
                        vehicles.append({
                            'box': [x, y, w, h],
                            'type': v_type
                        })
                        
            driving_side = 'unknown'
            if len(vehicles) > 0:
                if left_count > right_count: driving_side = 'left'
                elif right_count > left_count: driving_side = 'right'
                
            return {
                'status': 'success',
                'vehicles_detected': len(vehicles),
                'driving_side_estimate': driving_side,
                'vehicle_types': [v['type'] for v in vehicles],
                'confidence': 0.5 if len(vehicles) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Vehicle detection failed: {e}")
            return {'status': 'error', 'error': str(e)}
