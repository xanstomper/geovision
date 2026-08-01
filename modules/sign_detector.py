import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    import easyocr
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class SignDetector:
    def __init__(self):
        if not CV2_AVAILABLE:
            logger.warning("OpenCV/EasyOCR not available. SignDetector disabled.")
            self.reader = None
            return
        
        try:
            self.reader = easyocr.Reader(['en'])
        except Exception as e:
            logger.warning(f"Failed to load easyocr reader: {e}")
            self.reader = None
            
        self.color_ranges = {
            'highway_green': (np.array([40, 50, 50]), np.array([80, 255, 255])),
            'info_blue': (np.array([100, 100, 50]), np.array([130, 255, 255])),
            'stop_red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
            'warning_yellow': (np.array([20, 100, 100]), np.array([30, 255, 255]))
        }

    def detect_signs(self, image_path: str) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {'status': 'unavailable'}
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'status': 'error', 'error': 'Image not found'}
                
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            detected_signs = []
            
            for sign_type, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 500:
                        continue
                        
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    
                    if 0.2 < aspect_ratio < 5.0:
                        roi = img[y:y+h, x:x+w]
                        text, conf = "", 0.0
                        
                        if self.reader:
                            results = self.reader.readtext(roi)
                            if results:
                                text = " ".join([res[1] for res in results])
                                conf = float(results[0][2])
                                
                        detected_signs.append({
                            'sign_type': sign_type,
                            'bounding_box': [x, y, w, h],
                            'text': text,
                            'confidence': conf
                        })
                        
            return {
                'status': 'success',
                'detected_signs': detected_signs,
                'sign_count': len(detected_signs)
            }
            
        except Exception as e:
            logger.error(f"Failed to detect signs: {e}")
            return {'status': 'error', 'error': str(e)}
