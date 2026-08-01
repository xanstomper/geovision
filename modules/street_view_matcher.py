"""
Street View Matcher
Matches ground-level images to satellite and street view imagery
for cross-view verification and geolocation confirmation.
"""

import numpy as np
import cv2
import requests
import logging
import math
import hashlib
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class ImageComparisonFeatures:
    """Feature vectors extracted from an image for comparison."""
    histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    edge_density: float = 0.0
    edge_orientation_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    building_footprint_ratio: float = 0.0
    vegetation_index: float = 0.0
    color_moments: np.ndarray = field(default_factory=lambda: np.array([]))
    texture_features: np.ndarray = field(default_factory=lambda: np.array([]))
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    mean_brightness: float = 0.0
    contrast: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of a full location verification."""
    latitude: float
    longitude: float
    overall_similarity: float
    histogram_similarity: float
    edge_similarity: float
    building_footprint_similarity: float
    vegetation_similarity: float
    texture_similarity: float
    ground_image_path: str
    satellite_tile_used: Optional[str] = None
    street_view_used: Optional[str] = None
    is_verified: bool = False
    confidence_label: str = "unknown"
# ------------------------------------------------------------------
    # Tile Coordinate Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lat_lng_to_tile(lat: float, lng: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lng to OSM tile coordinates (x, y)."""
        lat_rad = math.radians(lat)
        n = 2 ** zoom
        x = int((lng + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @staticmethod
    def _tile_to_lat_lng(x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert OSM tile coordinates back to lat/lng (top-left corner)."""
        n = 2 ** zoom
        lng = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lng

    @staticmethod
    def _meters_per_pixel(lat: float, zoom: int) -> float:
        """Calculate ground resolution in meters per pixel."""
        return 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)

    def _get_tile_url(self, server_key: str, x: int, y: int, z: int) -> str:
        """Build a tile URL from a server key and tile coordinates."""
        template = self.TILE_SERVERS.get(server_key)
        if not template:
            raise ValueError(f"Unknown tile server: {server_key}")
        return template.format(z=z, y=y, x=x)

    # ------------------------------------------------------------------
    # Image Fetching
    # ------------------------------------------------------------------

    def _generate_cache_key(self, *args: Any) -> str:
        """Generate a deterministic cache filename from arguments (MD5)."""
        raw = ":".join(str(a) for a in args)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _fetch_tile(
        self,
        url: str,
        cache_path: Path,
        timeout: int = 15,
        retries: int = 2,
    ) -> Optional[np.ndarray]:
        """
        Fetch a single tile from a URL, with caching and retry logic.
        """
        # Check cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                img_array = np.frombuffer(cache_path.read_bytes(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    logger.debug(f"Cache hit: {cache_path.name}")
                    return img
                else:
                    logger.warning(f"Corrupt cache file, re-fetching: {cache_path}")
                    cache_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Cache read error: {e}, re-fetching")
                cache_path.unlink(missing_ok=True)

        # Fetch with retries
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self._session.get(url, timeout=timeout)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type and not response.content[:4] in (
                    b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF",
                ):
                    logger.warning(f"Non-image response from {url[:60]}...")
                    continue

                cache_path.write_bytes(response.content)
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    return img

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.debug(f"Tile fetch attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    import time
                    time.sleep(1 * (attempt + 1))

        logger.error(f"Failed to fetch tile after {retries + 1} attempts: {last_error}")
        return None
def fetch_satellite_tile(
        self,
        lat: float,
        lng: float,
        zoom: Optional[int] = None,
        tile_size: Optional[int] = None,
        source: str = "esri_satellite",
    ) -> Optional[np.ndarray]:
        """
        Fetch satellite imagery for a given lat/lng from free tile sources.

        Constructs a composite image from multiple 256px tiles when the
        requested tile_size exceeds the standard tile size. Uses ESRI
        ArcGIS World Imagery by default (no API key required).

        Args:
            lat: Latitude in decimal degrees
            lng: Longitude in decimal degrees
            zoom: Zoom level (0-19, default: self.zoom_level)
            tile_size: Output image size in pixels (default: self.tile_size)
            source: Tile server key from self.TILE_SERVERS

        Returns:
            RGB satellite image as numpy array, or None on failure
        """
        zoom = zoom or self.zoom_level
        tile_size = tile_size or self.tile_size
        zoom = max(1, min(19, zoom))

        cache_key = self._generate_cache_key("satellite", lat, lng, zoom, tile_size, source)
        cache_path = self.satellite_cache_dir / f"{cache_key}.png"

        # Try cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                img_array = np.frombuffer(cache_path.read_bytes(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    logger.info(f"Satellite tile cache hit: ({lat:.4f}, {lng:.4f}) z={zoom}")
                    return img
                else:
                    cache_path.unlink(missing_ok=True)
            except Exception:
                cache_path.unlink(missing_ok=True)

        logger.info(f"Fetching satellite tile: ({lat:.4f}, {lng:.4f}) z={zoom} size={tile_size} source={source}")

        tiles_needed = math.ceil(tile_size / self.TILE_SIZE)
        center_tile_x, center_tile_y = self._lat_lng_to_tile(lat, lng, zoom)
        start_x = center_tile_x - tiles_needed // 2
        start_y = center_tile_y - tiles_needed // 2

        # Fetch tiles in parallel
        tile_images: Dict[Tuple[int, int], np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tile = {}
            for dx in range(tiles_needed):
                for dy in range(tiles_needed):
                    tx = start_x + dx
                    ty = start_y + dy
                    tile_url = self._get_tile_url(source, tx, ty, zoom)
                    tile_cache = self.satellite_cache_dir / f"tile_{source}_{zoom}_{tx}_{ty}.png"
                    future = executor.submit(self._fetch_tile, tile_url, tile_cache)
                    future_to_tile[future] = (tx, ty)

            for future in as_completed(future_to_tile):
                tx, ty = future_to_tile[future]
                try:
                    img = future.result()
                    if img is not None:
                        tile_images[(tx, ty)] = img
                except Exception as e:
                    logger.warning(f"Tile ({tx}, {ty}) fetch failed: {e}")

        if not tile_images:
            if source != "osm_standard":
                logger.warning(f"Primary source {source} failed, trying OSM fallback")
                return self.fetch_satellite_tile(lat, lng, zoom, tile_size, "osm_standard")
            logger.error("All tile sources failed")
            return None

        # Stitch tiles together
        min_tx = min(tx for tx, _ in tile_images)
        max_tx = max(tx for tx, _ in tile_images)
        min_ty = min(ty for _, ty in tile_images)
        max_ty = max(ty for _, ty in tile_images)

        canvas_width = (max_tx - min_tx + 1) * self.TILE_SIZE
        canvas_height = (max_ty - min_ty + 1) * self.TILE_SIZE

        sample = next(iter(tile_images.values()))
        actual_tile_h, actual_tile_w = sample.shape[:2]
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for (tx, ty), img in tile_images.items():
            px = (tx - min_tx) * actual_tile_w
            py = (ty - min_ty) * actual_tile_h
            h, w = img.shape[:2]
            canvas[py:py + h, px:px + w] = img[:h, :w]

        # Crop to requested size (centered)
        if canvas.shape[1] > tile_size or canvas.shape[0] > tile_size:
            cy = canvas.shape[0] // 2
            cx = canvas.shape[1] // 2
            half = tile_size // 2
            y1 = max(0, cy - half)
            y2 = min(canvas.shape[0], cy + half + (tile_size % 2))
            x1 = max(0, cx - half)
            x2 = min(canvas.shape[1], cx + half + (tile_size % 2))
            canvas = canvas[y1:y2, x1:x2]

        if canvas.shape[0] != tile_size or canvas.shape[1] != tile_size:
            canvas = cv2.resize(canvas, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(cache_path), canvas)
        logger.info(f"Satellite tile cached: {canvas.shape} ({len(tile_images)} tiles)")
        return canvas
def fetch_street_view(
        self,
        lat: float,
        lng: float,
        size: Tuple[int, int] = (640, 640),
        fov: int = 90,
        heading: Optional[float] = None,
        pitch: float = 0,
    ) -> Optional[np.ndarray]:
        """
        Attempt to fetch street view imagery for a given location.

        Primary: Google Street View API (if key configured).
        Fallback: Mapillary free API, then cached approximation data.
        """
        if heading is None:
            heading = 0

        cache_key = self._generate_cache_key("streetview", lat, lng, size[0], size[1], fov, heading, pitch)
        cache_path = self.street_view_cache_dir / f"{cache_key}.png"

        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                img_array = np.frombuffer(cache_path.read_bytes(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    logger.info(f"Street view cache hit: ({lat:.4f}, {lng:.4f})")
                    return img
                else:
                    cache_path.unlink(missing_ok=True)
            except Exception:
                cache_path.unlink(missing_ok=True)

        # Try Google Street View API
        if self.api_key:
            try:
                params = {"location": f"{lat},{lng}", "size": f"{size[0]}x{size[1]}",
                          "fov": fov, "heading": heading, "pitch": pitch, "key": self.api_key}
                url = f"https://maps.googleapis.com/maps/api/streetview?{urlencode(params)}"
                response = self._session.get(url, timeout=15)
                response.raise_for_status()
                if len(response.content) > 1000:
                    img_array = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        cache_path.write_bytes(response.content)
                        return img
                else:
                    self._mark_unavailable(cache_path)
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Google Street View API error: {e}")

        # Fallback: Mapillary
        try:
            mapillary_img = self._fetch_mapillary_image(lat, lng)
            if mapillary_img is not None:
                cv2.imwrite(str(cache_path), mapillary_img)
                return mapillary_img
        except Exception as e:
            logger.debug(f"Mapillary fetch failed: {e}")

        # Fallback: cached approximation
        approx = self._load_street_view_approximation(lat, lng)
        if approx is not None:
            cv2.imwrite(str(cache_path), approx)
            return approx

        self._mark_unavailable(cache_path)
        return None

    def _fetch_mapillary_image(self, lat: float, lng: float) -> Optional[np.ndarray]:
        """Fetch a street-level image from Mapillary's free API."""
        search_url = (
            f"https://graph.mapillary.com/images?"
            f"access_token=MLY|4627724718021126|f3b0afb0d9a0daa542e496b403e40a73"
            f"&bbox={lng - 0.001},{lat - 0.001},{lng + 0.001},{lat + 0.001}"
            f"&fields=id,geometry&limit=5"
        )
        try:
            resp = self._session.get(search_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            images = data.get("data", [])
            if not images:
                return None
            best_id = images[0]["id"]
            img_url = f"https://images.mapillary.com/{best_id}/thumb-640.jpg"
            img_resp = self._session.get(img_url, timeout=10)
            img_resp.raise_for_status()
            if len(img_resp.content) > 1000:
                img_array = np.frombuffer(img_resp.content, np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.debug(f"Mapillary fetch error: {e}")
        return None

    def _mark_unavailable(self, cache_path: Path) -> None:
        """Write a marker file indicating no imagery is available."""
        try:
            placeholder = np.zeros((1, 1, 3), dtype=np.uint8)
            cv2.imwrite(str(cache_path.with_suffix(".nopic")), placeholder)
        except Exception:
            pass

    def _load_street_view_approximation(self, lat: float, lng: float) -> Optional[np.ndarray]:
        """Load a cached street view approximation near the given coordinates."""
        approx_dir = self.street_view_cache_dir / "approximations"
        if not approx_dir.exists():
            return None
        best_match = None
        best_distance = float("inf")
        for fpath in approx_dir.glob("*.png"):
            try:
                parts = fpath.stem.split("_")
                if len(parts) >= 3 and parts[0] == "sv":
                    f_lat = float(parts[1])
                    f_lng = float(parts[2])
                    dist = math.sqrt((f_lat - lat) ** 2 + (f_lng - lng) ** 2)
                    if dist < best_distance and dist < 0.01:
                        best_distance = dist
                        img_array = np.frombuffer(fpath.read_bytes(), np.uint8)
                        best_match = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except (ValueError, IndexError, OSError):
                continue
# ------------------------------------------------------------------
    # Image Feature Extraction
    # ------------------------------------------------------------------

    def extract_features(self, image: np.ndarray) -> ImageComparisonFeatures:
        """Extract comprehensive visual features from an image for comparison."""
        if image is None or image.size == 0:
            raise ValueError("Cannot extract features from empty image")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        features = ImageComparisonFeatures()
        features.histogram = self._extract_color_histogram(image)
        features.color_moments = self._extract_color_moments(image)
        features.edge_density, features.edge_orientation_histogram = self._extract_edge_features(image)
        features.building_footprint_ratio = self._estimate_building_footprint(image)
        features.vegetation_index = self._compute_vegetation_index(image)
        features.texture_features = self._extract_texture_features(image)
        features.dominant_colors = self._extract_dominant_colors(image, k=5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.mean_brightness = float(np.mean(gray))
        features.contrast = float(np.std(gray))
        features.metadata = {"image_shape": image.shape, "num_pixels": image.shape[0] * image.shape[1]}
        return features

    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract normalized 3D color histogram (32 bins per BGR channel)."""
        hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
        return hist.flatten()

    def _extract_color_moments(self, image: np.ndarray) -> np.ndarray:
        """Extract color moments (mean, std, skew) for each BGR channel."""
        moments = []
        for channel in cv2.split(image):
            flat = channel.astype(np.float32).flatten()
            mean = np.mean(flat)
            std = np.std(flat)
            skew = np.mean(((flat - mean) / (std + 1e-10)) ** 3)
            moments.extend([mean, std, skew])
        return np.array(moments, dtype=np.float32)

    def _extract_edge_features(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute edge density and orientation histogram via Canny + Sobel."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        med = np.median(blurred)
        edges = cv2.Canny(blurred, int(max(0, 0.66*med)), int(min(255, 1.33*med)), apertureSize=3)
        total = edges.shape[0] * edges.shape[1]
        edge_density = np.count_nonzero(edges) / max(total, 1)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        orient = np.arctan2(sobely, sobelx + 1e-10)
        mask = mag > (np.mean(mag) + 0.5*np.std(mag))
        hist, _ = np.histogram(orient[mask], bins=36, range=(-np.pi, np.pi), density=True)
        return edge_density, hist
def _estimate_building_footprint(self, image: np.ndarray) -> float:
        """Estimate ratio of image covered by building-like structures."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 4)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        total_area = image.shape[0] * image.shape[1]
        building_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < total_area * 0.005:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)
            vertices = len(approx)
            if 4 <= vertices <= 12 and circularity < 0.6:
                building_area += area
            elif vertices > 12 and circularity < 0.3:
                building_area += area * 0.5
        return min(building_area / max(total_area, 1), 1.0)

    def _compute_vegetation_index(self, image: np.ndarray) -> float:
        """Compute vegetation index from visible bands (VARI, GRVI, ExG)."""
        b, g, r = cv2.split(image.astype(np.float32))
        eps = 1e-10
        vari = (g - r) / (g + r - b + eps)
        grvi = (g - r) / (g + r + eps)
        exg = 2 * g - r - b
        exg_norm = (exg - np.min(exg)) / (np.max(exg) - np.min(exg) + eps)
        veg_mask = (vari > 0.05) | (grvi > 0.05) | (exg_norm > 0.5)
        if np.any(veg_mask):
            combined = 0.4 * vari[veg_mask] + 0.3 * grvi[veg_mask] + 0.3 * exg_norm[veg_mask]
            return float(np.mean(np.clip(combined, 0, 1)))
        return 0.0

    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns (LBP) histogram."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 1
        lbp = np.zeros_like(gray, dtype=np.uint8)
        h, w = gray.shape
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                center = gray[y, x]
                code = 0
                neighbors = [gray[y-1,x-1], gray[y-1,x], gray[y-1,x+1],
                             gray[y,x+1], gray[y+1,x+1], gray[y+1,x],
                             gray[y+1,x-1], gray[y,x-1]]
                for i, val in enumerate(neighbors):
                    if val >= center:
                        code |= (1 << i)
                lbp[y, x] = code
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        s = hist.sum()
        if s > 0:
            hist = hist.astype(np.float32) / s
        return hist

    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means clustering."""
        pixels = image.reshape(-1, 3).astype(np.float32)
        if len(pixels) > 50000:
            idx = np.random.choice(len(pixels), 50000, replace=False)
            pixels = pixels[idx]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        sorted_idx = np.argsort(counts)[::-1]
        return [(int(centers[i][0]), int(centers[i][1]), int(centers[i][2])) for i in sorted_idx]
# ------------------------------------------------------------------
    # Image Comparison Methods
    # ------------------------------------------------------------------

    def compare_images(self, ground_img: np.ndarray, satellite_img: np.ndarray) -> Dict[str, float]:
        """Compare ground and satellite images across multiple visual dimensions."""
        if ground_img is None or satellite_img is None:
            raise ValueError("Both images must be provided for comparison")
        gh, gw = ground_img.shape[:2]
        sh, sw = satellite_img.shape[:2]
        if gh * gw > sh * sw:
            ground_resized = cv2.resize(ground_img, (sw, sh), interpolation=cv2.INTER_AREA)
            gf, sf = self.extract_features(ground_resized), self.extract_features(satellite_img)
        else:
            sat_resized = cv2.resize(satellite_img, (gw, gh), interpolation=cv2.INTER_AREA)
            gf, sf = self.extract_features(ground_img), self.extract_features(sat_resized)
        scores = {
            "histogram_similarity": self._compare_histograms(gf.histogram, sf.histogram),
            "edge_similarity": self._compare_edge_features(gf.edge_density, gf.edge_orientation_histogram, sf.edge_density, sf.edge_orientation_histogram),
            "building_footprint_similarity": self._compare_building_footprints(gf.building_footprint_ratio, sf.building_footprint_ratio),
            "vegetation_similarity": self._compare_vegetation_indices(gf.vegetation_index, sf.vegetation_index),
            "texture_similarity": self._compare_texture_features(gf.texture_features, sf.texture_features),
            "color_moment_similarity": self._compare_color_moments(gf.color_moments, sf.color_moments),
        }
        scores["overall_similarity"] = self.compute_similarity_score({
            "histogram": scores["histogram_similarity"], "edge": scores["edge_similarity"],
            "building_footprint": scores["building_footprint_similarity"],
            "vegetation": scores["vegetation_similarity"], "texture": scores["texture_similarity"],
            "color_moments": scores["color_moment_similarity"],
        })
        return scores

    def _compare_histograms(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Compare histograms: correlation + intersection + chi-square."""
        if h1.size == 0 or h2.size == 0:
            return 0.0
        if h1.size != h2.size:
            ms = min(h1.size, h2.size); h1, h2 = h1[:ms], h2[:ms]
        a, b = h1.reshape(-1, 1).astype(np.float32), h2.reshape(-1, 1).astype(np.float32)
        corr = max(0.0, (cv2.compareHist(a, b, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
        inter = min(1.0, cv2.compareHist(a, b, cv2.HISTCMP_INTERSECT))
        chi = max(0.0, 1.0 / (1.0 + cv2.compareHist(a, b, cv2.HISTCMP_CHISQR) * 0.01))
        return float(np.clip(0.5*corr + 0.3*inter + 0.2*chi, 0.0, 1.0))

    def _compare_edge_features(self, d1: float, o1: np.ndarray, d2: float, o2: np.ndarray) -> float:
        """Compare edge density and orientation histograms."""
        density_sim = max(0.0, 1.0 - abs(d1 - d2) * 3.0)
        if o1.size > 0 and o2.size > 0:
            if o1.size != o2.size:
                ms = min(o1.size, o2.size); o1, o2 = o1[:ms], o2[:ms]
            a = o1.reshape(-1, 1).astype(np.float32) / (np.sum(o1) + 1e-10)
            b = o2.reshape(-1, 1).astype(np.float32) / (np.sum(o2) + 1e-10)
            orient_sim = max(0.0, (cv2.compareHist(a, b, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
        else:
            orient_sim = 0.5
        return float(0.5 * density_sim + 0.5 * orient_sim)

    def _compare_building_footprints(self, r1: float, r2: float) -> float:
        return float(math.exp(-5.0 * abs(r1 - r2)))

    def _compare_vegetation_indices(self, v1: float, v2: float) -> float:
        return float(math.exp(-4.0 * abs(v1 - v2)))

    def _compare_texture_features(self, t1: np.ndarray, t2: np.ndarray) -> float:
        if t1.size == 0 or t2.size == 0:
            return 0.5
        if t1.size != t2.size:
            ms = min(t1.size, t2.size); t1, t2 = t1[:ms], t2[:ms]
        a = t1.reshape(-1, 1).astype(np.float32) / (np.sum(t1) + 1e-10)
        b = t2.reshape(-1, 1).astype(np.float32) / (np.sum(t2) + 1e-10)
        chi = np.sum(((a - b) ** 2) / (a + b + 1e-10))
        return float(max(0.0, 1.0 / (1.0 + chi * 0.1)))

    def _compare_color_moments(self, m1: np.ndarray, m2: np.ndarray) -> float:
        if m1.size == 0 or m2.size == 0:
            return 0.5
        if m1.size != m2.size:
            ms = min(m1.size, m2.size); m1, m2 = m1[:ms], m2[:ms]
        c1, c2 = m1.copy().astype(np.float32), m2.copy().astype(np.float32)
        for idx in [0, 3, 6]:
            if idx < len(c1):
                c1[idx] /= 255.0; c2[idx] /= 255.0
        return float(max(0.0, 1.0 / (1.0 + np.sqrt(np.sum((c1 - c2) ** 2)))))
# ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compute_similarity_score(self, feature_scores: Dict[str, float],
                                 weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute aggregate similarity from individual feature scores.

        Default weights: histogram 0.25, edge 0.20, building_footprint 0.15,
        vegetation 0.15, texture 0.10, color_moments 0.15.
        """
        default_weights = {
            "histogram": 0.25, "edge": 0.20, "building_footprint": 0.15,
            "vegetation": 0.15, "texture": 0.10, "color_moments": 0.15,
        }
        if weights is None:
            weights = default_weights
        total = 0.0
        w_sum = 0.0
        for feat, score in feature_scores.items():
            w = weights.get(feat, 0.0)
            total += w * score
            w_sum += w
        return float(np.clip(total / max(w_sum, 1e-10), 0.0, 1.0))

    def _determine_confidence_label(self, score: float) -> str:
        """Convert numerical similarity score to human-readable confidence label."""
        if score >= 0.9: return "very_high"
        elif score >= 0.8: return "high"
        elif score >= 0.7: return "moderate_high"
        elif score >= self.similarity_threshold: return "moderate"
        elif score >= 0.4: return "low"
        else: return "very_low"

    # ------------------------------------------------------------------
    # Full Verification Pipeline
    # ------------------------------------------------------------------

    def verify_location(self, lat: float, lng: float, ground_image_path: str,
                        zoom: Optional[int] = None, use_street_view: bool = True) -> VerificationResult:
        """
        Full verification pipeline.

        Given a candidate (lat, lng) and ground image path, fetches satellite
        imagery, compares features, optionally fetches street view, and
        returns a VerificationResult with similarity scores and verdict.
        """
        zoom = zoom or self.zoom_level
        logger.info(f"Verifying: ({lat:.6f}, {lng:.6f}) image={ground_image_path}")

        ground_img = cv2.imread(ground_image_path)
        if ground_img is None:
            raise FileNotFoundError(f"Cannot load ground image: {ground_image_path}")

        satellite_img = self.fetch_satellite_tile(lat, lng, zoom=zoom)
        if satellite_img is None:
            logger.warning("Satellite tile unavailable")
            return VerificationResult(lat, lng, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      ground_image_path, is_verified=False,
                                      confidence_label="no_satellite_data",
                                      details={"error": "Satellite tile fetch failed"})

        comparison = self.compare_images(ground_img, satellite_img)

        street_view_img = None
        if use_street_view:
            street_view_img = self.fetch_street_view(lat, lng)

        street_view_comparison = None
        if street_view_img is not None:
            street_view_comparison = self.compare_images(ground_img, street_view_img)
            logger.info(f"Ground vs Street View: {street_view_comparison['overall_similarity']:.3f}")

        overall = comparison["overall_similarity"]
        if street_view_comparison and street_view_comparison["overall_similarity"] > 0.5:
            overall = 0.4 * overall + 0.6 * street_view_comparison["overall_similarity"]

        is_verified = overall >= self.similarity_threshold
        label = self._determine_confidence_label(overall)

        details = {
            "zoom_level": zoom,
            "satellite_tile_shape": list(satellite_img.shape) if satellite_img is not None else None,
            "ground_image_shape": list(ground_img.shape),
            "street_view_available": street_view_img is not None,
            "feature_breakdown": {k: comparison[k] for k in comparison if k != "overall_similarity"},
        }
        if street_view_comparison:
            details["street_view_comparison"] = street_view_comparison

        result = VerificationResult(lat, lng, overall, comparison["histogram_similarity"],
                                    comparison["edge_similarity"],
                                    comparison["building_footprint_similarity"],
                                    comparison["vegetation_similarity"],
                                    comparison["texture_similarity"],
                                    ground_image_path,
                                    is_verified=is_verified, confidence_label=label, details=details)
        logger.info(f"Verification complete: overall={overall:.3f} verified={is_verified} label={label}")
        return result
# ------------------------------------------------------------------
    # Batch Verification
    # ------------------------------------------------------------------

    def verify_multiple_candidates(self, candidates: List[Tuple[float, float]],
                                   ground_image_path: str,
                                   zoom: Optional[int] = None) -> List[VerificationResult]:
        """
        Verify a list of candidate (lat, lng) pairs against a single ground image.

        Useful for narrowing down a set of possible locations.
        Returns results sorted by overall_similarity descending.
        """
        results = []
        for lat, lng in candidates:
            try:
                result = self.verify_location(lat, lng, ground_image_path, zoom=zoom, use_street_view=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Verification failed for ({lat:.4f}, {lng:.4f}): {e}")
                results.append(VerificationResult(lat, lng, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  ground_image_path, is_verified=False,
                                                  confidence_label="error", details={"error": str(e)}))
        results.sort(key=lambda r: r.overall_similarity, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Cache Management
    # ------------------------------------------------------------------

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Clear the tile cache, optionally only files older than a threshold."""
        import time
        removed = 0
        now = time.time()
        max_age = older_than_days * 86400 if older_than_days else 0
        for subdir in [self.satellite_cache_dir, self.street_view_cache_dir, self.feature_cache_dir]:
            for fpath in subdir.glob("*"):
                if fpath.is_file():
                    if older_than_days is None:
                        fpath.unlink(missing_ok=True); removed += 1
                    elif now - fpath.stat().st_mtime > max_age:
                        fpath.unlink(missing_ok=True); removed += 1
        logger.info(f"Cache cleared: {removed} files removed")
        return removed

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the tile cache."""
        stats = {
            "cache_dir": str(self.cache_dir),
            "satellite_cache": {"count": 0, "size_bytes": 0},
            "street_view_cache": {"count": 0, "size_bytes": 0},
            "feature_cache": {"count": 0, "size_bytes": 0},
        }
        for key, subdir in [("satellite_cache", self.satellite_cache_dir),
                            ("street_view_cache", self.street_view_cache_dir),
                            ("feature_cache", self.feature_cache_dir)]:
            if subdir.exists():
                for fpath in subdir.glob("*"):
                    if fpath.is_file():
                        stats[key]["count"] += 1
                        stats[key]["size_bytes"] += fpath.stat().st_size
        stats["total_size_mb"] = sum(s["size_bytes"] for s in stats.values()) / (1024 * 1024)
        return stats

    # ------------------------------------------------------------------
    # Utility: Visualize Comparison
    # ------------------------------------------------------------------

    def create_comparison_montage(self, ground_image_path: str, satellite_img: np.ndarray,
                                  street_view_img: Optional[np.ndarray] = None,
                                  result: Optional[VerificationResult] = None) -> np.ndarray:
        """Create a visual montage of ground, satellite, and street view images."""
        ground_img = cv2.imread(ground_image_path)
        if ground_img is None:
            raise FileNotFoundError(f"Cannot load: {ground_image_path}")
        target_h = 400
        images = []
        for img, label in [(ground_img, "Ground Level"), (satellite_img, "Satellite View"),
                           (street_view_img, "Street View")]:
            if img is not None:
                aspect = img.shape[1] / img.shape[0]
                nw = int(target_h * aspect)
                resized = cv2.resize(img, (nw, target_h))
                label_img = np.zeros((40, nw, 3), dtype=np.uint8)
                cv2.putText(label_img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                images.append(np.vstack([label_img, resized]))
        if not images:
            raise ValueError("No images to display")
        min_h = min(img.shape[0] for img in images)
        resized = []
        for img in images:
            aspect = img.shape[1] / img.shape[0]
            resized.append(cv2.resize(img, (int(min_h * aspect), min_h)))
        montage = np.hstack(resized)
        if result:
            annotation = f"Similarity: {result.overall_similarity:.2f} | Verified: {result.is_verified} | Confidence: {result.confidence_label}"
            cv2.putText(montage, annotation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return montage

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()
        return False

    def close(self):
        """Close the HTTP session."""
        self._session.close()
        return best_match
    details: Dict = field(default_factory=dict)