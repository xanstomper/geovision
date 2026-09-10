import cv2
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class RoadAnalyzer:
    """Analyzes road markings and driving side from OpenCV image analysis."""
    
    def __init__(self):
        pass

    def analyze_road_features(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return {
                "driving_side": "unknown",
                "line_color": "unknown",
                "road_surface": "unknown",
                "region_indicators": [],
                "confidence": 0.0
            }
            
        height, width = image.shape[:2]
        bottom_third = image[height - height//3:, :]
        hsv_image = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        # Yellow lines (North America, etc)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # White lines
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 20, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        white_pixels = cv2.countNonZero(white_mask)
        
        total_pixels = bottom_third.shape[0] * bottom_third.shape[1]
        yellow_ratio = yellow_pixels / total_pixels
        white_ratio = white_pixels / total_pixels
        
        line_color = "none"
        if yellow_ratio > 0.001 and yellow_ratio > white_ratio * 0.2:
            line_color = "yellow"
        elif white_ratio > 0.001:
            line_color = "white"
            
        region_indicators = []
        if line_color == "yellow":
            region_indicators.append("North America / South America")
        elif line_color == "white":
            region_indicators.append("Europe / Asia / Oceania")
            
        # Basic driving side detection: looking at line positions in the lower half
        height, width = image.shape[:2]
        lower_half = image[height//2:, :]
        gray_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_lower, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
        
        left_count = 0
        right_count = 0
        if lines is not None:
            for line in lines:
                pts = line.reshape(-1) if hasattr(line, "reshape") else list(line)
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in pts[:4]]
                if x1 < width // 2 and x2 < width // 2:
                    left_count += 1
                elif x1 > width // 2 and x2 > width // 2:
                    right_count += 1
                    
        driving_side = "unknown"
        if left_count > right_count * 1.5:
            # Lines mostly on left might indicate left-hand traffic? It's heuristic.
            driving_side = "left"
        elif right_count > left_count * 1.5:
            driving_side = "right"
            
        if driving_side == "left":
            region_indicators.append("UK / Japan / Australia / India / South Africa")
            
        # Road surface
        bottom_quarter = image[height - height//4:, :]
        gray_image = cv2.cvtColor(bottom_quarter, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_image)
        road_surface = "dark asphalt" if mean_intensity < 100 else "light concrete"
            
        return {
            "driving_side": driving_side,
            "line_color": line_color,
            "road_surface": road_surface,
            "region_indicators": region_indicators,
            "confidence": 0.6 if line_color != "none" else 0.2
        }
