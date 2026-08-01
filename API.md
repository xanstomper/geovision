# GeoVision API Documentation

This document outlines the main classes and methods for the newly integrated OSINT and visual reasoning modules in GeoVision.

## `modules.scene_classifier`

### `class SceneClassifier`
Utilizes PyTorch and ResNet50 to classify the overall scene environment of an image.

*   **`__init__(self)`**
    Initializes the ResNet50 model and loads ImageNet categories. Maps general ImageNet keywords to broad scene types (e.g., Urban, Rural, Coastal).
*   **`classify(self, image_path: str) -> Dict[str, Any]`**
    Analyzes the provided image and returns a dictionary containing:
    *   `status`: 'success' or 'error'
    *   `scene_type`: The mapped scene category (e.g., 'Urban/city', 'Mountain/alpine')
    *   `confidence`: The probability of the top class.
    *   `top_5`: A list of the top 5 raw ImageNet classes and their probabilities.

## `modules.vegetation_classifier`

### `class VegetationClassifier`
Uses OpenCV to calculate vegetation indices and estimate climate based on color data.

*   **`__init__(self)`**
    Initializes the classifier and checks for OpenCV availability.
*   **`classify(self, image_path: str) -> Dict[str, Any]`**
    Processes the image to calculate the Visible Atmospherically Resistant Index (VARI) and green pixel fractions. Returns a dictionary containing:
    *   `vegetation_density`: Estimated density ('forest', 'dense', 'moderate', 'sparse', 'none').
    *   `estimated_climate`: Derived climate ('tropical', 'arid', 'temperate', 'unknown').
    *   `vegetation_indices`: Raw calculated indices (`vari_mean`, `green_fraction`).
    *   `confidence`: Confidence score based on the proportion of green pixels.

## `modules.osm_feature_matcher`

### `class OSMFeatureMatcher`
Provides an interface to the OpenStreetMap Overpass API for querying local geographical features.

*   **`find_pois(self, lat: float, lon: float, radius_m: int = 500) -> dict`**
    Queries OSM for amenities and shops within a given radius.
    *   **Returns:** A dictionary containing the `count` of POIs and a list of `pois` (name, type, lat, lon).
*   **`get_road_characteristics(self, lat: float, lon: float) -> dict`**
    Queries OSM for highway and surface types nearby.
    *   **Returns:** A dictionary containing lists of `highway_types` and `surface_types`.
*   **`get_building_density(self, lat: float, lon: float, radius_m: int = 500) -> dict`**
    Queries OSM for the count of buildings in the radius to estimate urban density.
    *   **Returns:** A dictionary with `building_count` and a normalized `density_score`.

## `modules.evidence_fusion`

### `class EvidenceFusion`
A probabilistic aggregator that fuses multiple geographical clues to calculate a final location estimate and uncertainty radius.

*   **`__init__(self)`**
    Initializes internal lists for evidence points and sets default reliability weights for different sources (e.g., 'exif': 1.0, 'telecom': 0.9, 'vegetation': 0.3).
*   **`add_evidence(self, source: str, lat: float, lon: float, confidence: float, radius_km: float = 50.0)`**
    Adds a specific coordinate point as evidence.
    *   `source`: The name of the module that generated the clue.
    *   `confidence`: The internal confidence of the module (0.0 to 1.0).
*   **`add_region_evidence(self, source: str, country: Optional[str] = None, state: Optional[str] = None, confidence: float = 0.5)`**
    Adds broader regional evidence (used for contextual weighting).
*   **`get_prediction(self) -> dict`**
    Calculates the weighted average of all evidence points.
    *   **Returns:** A dictionary containing `lat`, `lon`, `confidence` (overall), and `radius_km` (derived from the spatial variance of the evidence points).
*   **`get_candidates(self, top_k: int = 10) -> List[dict]`**
    Returns a sorted list of the strongest individual evidence points based on their effective weight.
