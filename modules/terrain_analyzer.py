import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class TerrainAnalyzer:
    def __init__(self):
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for TerrainAnalyzer.")
            
    def analyze(self, image_path: str) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {'status': 'unavailable'}
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'status': 'error'}
                
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Simple horizon detection: find top-most strong horizontal lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width//4, maxLineGap=20)
            horizon_y = height // 2
            
            if lines is not None:
                h_lines = [l[0] for l in lines if abs(l[0][1] - l[0][3]) < 20]
                if h_lines:
                    horizon_y = min([l[1] for l in h_lines])
                    
            terrain_type = 'flat' if horizon_y > height * 0.3 else 'hilly/mountainous'
            
            # Analyze ground color (bottom 20% of image)
            ground_roi = img[int(height*0.8):height, :]
            avg_color_bgr = np.mean(ground_roi, axis=(0, 1))
            b, g, r = avg_color_bgr
            
            ground_color = 'unknown'
            if r > g and r > b and r - g > 30: ground_color = 'red soil'
            elif r > g and b > g and r < 100: ground_color = 'dark earth'
            elif r > 150 and g > 150 and b > 100: ground_color = 'sand/concrete'
            elif r < 100 and g < 100 and b < 100: ground_color = 'asphalt'
            
            # Water detection via blue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
            water_fraction = np.sum(blue_mask > 0) / (height * width)
            water_detected = water_fraction > 0.05
            
            elevation = 'low/sea_level' if water_detected else 'unknown'
            if terrain_type == 'hilly/mountainous' and not water_detected:
                elevation = 'high_elevation'
                
            return {
                'status': 'success',
                'terrain_type': terrain_type,
                'ground_color': ground_color,
                'elevation_estimate': elevation,
                'water_detected': water_detected,
                'confidence': 0.7
            }
            
        except Exception as e:
            logger.error(f"Terrain analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
