"""
GeoVision Core Engine
Advanced geolocation analysis using multi-layer computer vision
"""

import cv2
import numpy as np
from PIL import Image
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import colorsys

logger = logging.getLogger(__name__)


@dataclass
class VisionFeatures:
    """Extracted vision features from an image"""
    
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    color_names: List[str] = field(default_factory=list)
    material_hues: Dict[str, float] = field(default_factory=dict)
    building_type: Optional[str] = None
    floors_estimate: Optional[int] = None
    facade_material: Optional[str] = None
    roof_type: Optional[str] = None
    window_density: float = 0.0
    building_density: float = 0.0
    road_type: Optional[str] = None
    has_traffic_signs: bool = False
    has_sidewalk: bool = False
    has_street_lights: bool = False
    vegetation_density: float = 0.0
    tree_types: List[str] = field(default_factory=list)
    leaf_color_variance: float = 0.0
    has_grass: bool = False
    detected_text: List[str] = field(default_factory=list)
    sign_types: List[str] = field(default_factory=list)
    license_plates: List[str] = field(default_factory=list)
    shadow_direction: Optional[float] = None
    shadow_length_ratio: float = 0.0
    estimated_time_of_day: Optional[str] = None
    architectural_style: Optional[str] = None
    era_estimate: Optional[str] = None
    region_indicators: List[str] = field(default_factory=list)
    sky_percentage: float = 0.0
    weather_condition: Optional[str] = None
    cloud_density: float = 0.0
    edge_density: float = 0.0
    texture_complexity: float = 0.0
    perspective_lines: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "dominant_colors": [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in self.dominant_colors],
            "color_names": self.color_names,
            "material_hues": self.material_hues,
            "building_type": self.building_type,
            "floors_estimate": self.floors_estimate,
            "facade_material": self.facade_material,
            "roof_type": self.roof_type,
            "window_density": self.window_density,
            "building_density": self.building_density,
            "road_type": self.road_type,
            "has_traffic_signs": self.has_traffic_signs,
            "has_sidewalk": self.has_sidewalk,
            "has_street_lights": self.has_street_lights,
            "vegetation_density": self.vegetation_density,
            "tree_types": self.tree_types,
            "leaf_color_variance": self.leaf_color_variance,
            "has_grass": self.has_grass,
            "detected_text": self.detected_text,
            "sign_types": self.sign_types,
            "license_plates": self.license_plates,
            "shadow_direction": self.shadow_direction,
            "shadow_length_ratio": self.shadow_length_ratio,
            "estimated_time_of_day": self.estimated_time_of_day,
            "architectural_style": self.architectural_style,
            "era_estimate": self.era_estimate,
            "region_indicators": self.region_indicators,
            "sky_percentage": self.sky_percentage,
            "weather_condition": self.weather_condition,
            "cloud_density": self.cloud_density,
            "edge_density": self.edge_density,
            "texture_complexity": self.texture_complexity,
            "perspective_lines": self.perspective_lines
        }


class VisionEngine:
    """
    Multi-modal computer vision analysis engine for geolocation
    """
    
    MATERIAL_COLORS = {}
    ARCHITECTURAL_REGIONS = {}
    HSV_THRESHOLDS = {}
    
    def __init__(self):
        self.features = VisionFeatures()
        self.width = 0
        self.height = 0
        self._init_models()
    
    def _init_models(self):
        self.yolo = None
        
        config_path = Path(__file__).parent.parent / "config" / "vision_config.json"
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                
                # Material colors are listed as lists of RGB values, convert to tuples
                if "material_colors" in data:
                    self.MATERIAL_COLORS = {
                        k: [tuple(v[0]), tuple(v[1])] 
                        for k, v in data["material_colors"].items()
                    }
                
                if "architectural_regions" in data:
                    self.ARCHITECTURAL_REGIONS = data["architectural_regions"]
                
                if "hsv_thresholds" in data:
                    self.HSV_THRESHOLDS = data["hsv_thresholds"]
        except Exception as e:
            logger.error(f"Failed to load vision config: {e}")
    
    def analyze_image(self, image_path: str) -> VisionFeatures:
        self.features = VisionFeatures()
        image = self._load_and_preprocess(image_path)
        if image is None:
            return self.features
        
        self._analyze_colors_and_materials(image)
        self._analyze_architecture(image)
        self._analyze_infrastructure(image)
        self._analyze_vegetation(image)
        self._analyze_shadows_and_lighting(image)
        self._detect_text_and_signs(image_path)
        self._analyze_sky_and_weather(image)
        self._analyze_texture_and_geometry(image)
        self._cross_correlate_features()
        
        return self.features
    
    def _load_and_preprocess(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img_np = np.array(img)
            self.height, self.width = img_np.shape[:2]
            img_np = cv2.resize(img_np, (1024, 768), interpolation=cv2.INTER_AREA)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return img_np
        except Exception as e:
            logger.error(f"Image loading error: {e}")
            return None
    
    def _analyze_colors_and_materials(self, image: np.ndarray):
        pixels = image.reshape(-1, 3)
        Z = np.float32(pixels[np.random.choice(pixels.shape[0], min(10000, pixels.shape[0]))])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(-counts)
        self.features.dominant_colors = [tuple(centers[i]) for i in sorted_indices[:6]]
        self.features.color_names = [self._color_to_name(c) for c in self.features.dominant_colors]
        
        for material, (lower_rgb, upper_rgb) in self.MATERIAL_COLORS.items():
            mask = cv2.inRange(image, np.array(lower_rgb), np.array(upper_rgb))
            ratio = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
            if ratio > 0.02:
                self.features.material_hues[material] = float(ratio)
        
        if self.features.material_hues:
            self.features.facade_material = max(self.features.material_hues, key=self.features.material_hues.get)
    
    def _color_to_name(self, rgb: Tuple[int, int, int]) -> str:
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if v < 0.2:
            return "black"
        elif v > 0.9 and s < 0.1:
            return "white"
        elif s < 0.15:
            return "gray"
        h_deg = h * 360
        if h_deg < 30 or h_deg > 330:
            return "red"
        elif h_deg < 60:
            return "orange"
        elif h_deg < 90:
            return "yellow"
        elif h_deg < 150:
            return "green"
        elif h_deg < 210:
            return "cyan"
        elif h_deg < 270:
            return "blue"
        elif h_deg < 300:
            return "purple"
        return "magenta"
    
    def _analyze_architecture(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        self.features.edge_density = float(np.sum(edges > 0) / edges.size)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        line_counts = np.sum(horizontal_lines > 0, axis=1)
        peaks = self._find_peaks(line_counts, height=50, distance=30)
        self.features.floors_estimate = max(len(peaks), 2) if peaks else 3
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        building_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
        self.features.building_density = building_area / (self.width * self.height)
        
        roof_horizontal = np.sum(horizontal_lines[:self.height//4] > 0)
        roof_percentage = roof_horizontal / (np.sum(edges > 0) + 1)
        self.features.roof_type = "flat" if roof_percentage > 0.3 else "pitched"
        
        window_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        windows = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, window_kernel)
        self.features.window_density = float(np.sum(windows > 0) / windows.size * 10)
        
        self._classify_architectural_style()
    
    def _find_peaks(self, arr: np.ndarray, height: int = 1, distance: int = 1) -> List[int]:
        peaks = []
        for i in range(distance, len(arr) - distance):
            if arr[i] > height and all(arr[i] > arr[j] for j in range(max(0, i-distance), i)) and \
               all(arr[i] > arr[j] for j in range(i+1, min(len(arr), i+distance+1))):
                peaks.append(i)
        return peaks
    
    def _classify_architectural_style(self):
        style_scores = {}
        if "red" in self.features.color_names and self.features.floors_estimate and self.features.floors_estimate >= 2:
            style_scores["red_brick_apartments"] = 0.8
        if "orange" in self.features.color_names or "yellow" in self.features.color_names:
            if self.features.facade_material in ["tan_brick", "stucco"]:
                style_scores["tan_stucco"] = 0.7
        
        best_style = max(style_scores, key=style_scores.get) if style_scores else None
        if best_style and best_style in self.ARCHITECTURAL_REGIONS:
            style_info = self.ARCHITECTURAL_REGIONS[best_style]
            self.features.architectural_style = style_info["style"]
            self.features.era_estimate = style_info["era"]
    
    def _analyze_infrastructure(self, image: np.ndarray):
        bottom_half = image[self.height//2:, :]
        lower = self.HSV_THRESHOLDS.get("road_asphalt", {}).get("lower", [20, 20, 20])
        upper = self.HSV_THRESHOLDS.get("road_asphalt", {}).get("upper", [100, 100, 100])
        road_mask = cv2.inRange(bottom_half, np.array(lower), np.array(upper))
        road_ratio = np.sum(road_mask > 0) / (bottom_half.shape[0] * bottom_half.shape[1])
        if road_ratio > 0.1:
            self.features.road_type = "asphalt_road"
        
        lower_sw = self.HSV_THRESHOLDS.get("sidewalk_concrete", {}).get("lower", [150, 140, 130])
        upper_sw = self.HSV_THRESHOLDS.get("sidewalk_concrete", {}).get("upper", [230, 225, 215])
        sidewalk_mask = cv2.inRange(bottom_half, np.array(lower_sw), np.array(upper_sw))
        sidewalk_ratio = np.sum(sidewalk_mask > 0) / (bottom_half.shape[0] * bottom_half.shape[1])
        self.features.has_sidewalk = sidewalk_ratio > 0.05
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        street_light_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lights = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, street_light_kernel)
        self.features.has_street_lights = np.sum(lights > 0) > 50
    
    def _analyze_vegetation(self, image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = self.HSV_THRESHOLDS.get("vegetation_green", {}).get("lower", [30, 30, 30])
        upper_green = self.HSV_THRESHOLDS.get("vegetation_green", {}).get("upper", [90, 255, 255])
        green_lower = np.array(lower_green)
        green_upper = np.array(upper_green)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        self.features.vegetation_density = float(np.sum(green_mask > 0) / (self.height * self.width))
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_green = [c for c in contours if cv2.contourArea(c) > 10000]
        self.features.tree_types = ["deciduous" if len(large_green) > 2 else "sparse"]
        
        grass_mask = green_mask[self.height//2:, :]
        self.features.has_grass = np.sum(grass_mask > 0) / grass_mask.size > 0.05
        
        if len(large_green) > 0:
            green_areas = []
            for c in large_green[:5]:
                mask = cv2.drawContours(np.zeros_like(green_mask), [c], -1, 255, -1)
                mean_color = cv2.mean(image, mask=mask)
                green_areas.append(mean_color[:3])
            if green_areas:
                green_means = np.array(green_areas)
                self.features.leaf_color_variance = float(np.mean(np.var(green_means, axis=0)))
    
    def _analyze_shadows_and_lighting(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, shadow_candidates = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                # OpenCV5 HoughLinesP returns (N,4); cv2<4.4 returns (N,1,4).
                line = line.reshape(-1)
                if line.size < 4:
                    continue
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                if x2 == x1:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            if angles:
                self.features.shadow_direction = float(np.median(angles))
        
        self.features.shadow_length_ratio = float(np.sum(shadow_candidates > 0) / (self.height * self.width))
        if self.features.shadow_length_ratio > 0.15:
            self.features.estimated_time_of_day = "morning_or_evening"
        elif self.features.shadow_length_ratio > 0.05:
            self.features.estimated_time_of_day = "afternoon"
        else:
            self.features.estimated_time_of_day = "midday"
    
    def _detect_text_and_signs(self, image_path: str):
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(image_path, detail=0, paragraph=True)
            self.features.detected_text = [r for r in results if len(r) > 2]
            sign_keywords = ["stop", "yield", "parking", "exit", "entrance", "no", "speed", "limit"]
            self.features.sign_types = [t for t in self.features.detected_text if any(k in t.lower() for k in sign_keywords)]
            plate_patterns = [t for t in self.features.detected_text if 6 <= len(t) <= 10]
            self.features.license_plates = plate_patterns
        except Exception as e:
            logger.warning(f"OCR unavailable: {e}")
    
    def _analyze_sky_and_weather(self, image: np.ndarray):
        top_third = image[:self.height//3, :]
        hsv_top = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
        lower_sky = self.HSV_THRESHOLDS.get("sky_blue", {}).get("lower", [80, 20, 100])
        upper_sky = self.HSV_THRESHOLDS.get("sky_blue", {}).get("upper", [130, 255, 255])
        sky_mask = cv2.inRange(hsv_top, np.array(lower_sky), np.array(upper_sky))
        sky_ratio = np.sum(sky_mask > 0) / (top_third.shape[0] * top_third.shape[1])
        self.features.sky_percentage = float(sky_ratio)
        
        lower_cloud = self.HSV_THRESHOLDS.get("cloud_bright", {}).get("lower", [200, 200, 200])
        upper_cloud = self.HSV_THRESHOLDS.get("cloud_bright", {}).get("upper", [255, 255, 255])
        bright_mask = cv2.inRange(top_third, np.array(lower_cloud), np.array(upper_cloud))
        cloud_ratio = np.sum(bright_mask > 0) / (top_third.shape[0] * top_third.shape[1])
        self.features.cloud_density = float(cloud_ratio)
        
        if sky_ratio > 0.3 and cloud_ratio > 0.2:
            self.features.weather_condition = "partly_cloudy"
        elif sky_ratio > 0.5:
            self.features.weather_condition = "clear"
        elif cloud_ratio > 0.5:
            self.features.weather_condition = "overcast"
        else:
            self.features.weather_condition = "unclear"
    
    def _analyze_texture_and_geometry(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = self._compute_glcm(gray)
        contrast = np.sum(glcm ** 2)
        self.features.texture_complexity = float(contrast)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
        self.features.perspective_lines = len(lines) if lines is not None else 0
    
    def _compute_glcm(self, gray: np.ndarray) -> np.ndarray:
        glcm = np.zeros((16, 16), dtype=np.float32)
        quantized = (gray / 16).astype(np.uint8)
        for i in range(quantized.shape[0] - 1):
            for j in range(quantized.shape[1] - 1):
                a = quantized[i, j]
                b = quantized[i+1, j+1]
                glcm[a, b] += 1
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
        return glcm
    
    def _cross_correlate_features(self):
        if self.features.facade_material == "red_brick" and self.features.floors_estimate and self.features.floors_estimate >= 2:
            self.features.building_type = "multi-unit residential"
        
        if self.features.vegetation_density > 0.3 and self.features.leaf_color_variance > 100:
            self.features.tree_types.append("temperate_deciduous_forest")
        
        if self.features.has_sidewalk and self.features.road_type and self.features.has_street_lights:
            self.features.region_indicators.append("developed_urban_area")
    
    def get_analysis_json(self) -> str:
        return json.dumps(self.features.to_dict(), indent=2)
    
    def visualize_features(self, image_path: str, output_path: str):
        img = self._load_and_preprocess(image_path)
        if img is None:
            return
        vis = img.copy()
        materials = list(self.features.material_hues.keys())
        for idx, (material, ratio) in enumerate(self.features.material_hues.items()):
            if ratio > 0.05:
                cv2.putText(vis, f"{material}: {ratio:.0%}", 
                           (10, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(output_path, vis)
