from .satellite_matcher import SatelliteMatcher, SatelliteMatch
from .location_reasoner import LocationReasoner, LocationEstimate
from .deep_analyzer import DeepAnalyzer
from .google_maps_controller import GoogleMapsController, MapsVerificationResult
from .park_finder import find_nearby_parks
from .browser_automation import BrowserAutomation

try:
    from .scene_classifier import SceneClassifier
except ImportError:
    SceneClassifier = None

try:
    from .sign_detector import SignDetector
except ImportError:
    SignDetector = None

try:
    from .vehicle_detector import VehicleDetector
except ImportError:
    VehicleDetector = None

try:
    from .vegetation_classifier import VegetationClassifier
except ImportError:
    VegetationClassifier = None

try:
    from .terrain_analyzer import TerrainAnalyzer
except ImportError:
    TerrainAnalyzer = None

try:
    from .geonames_client import GeoNamesClient
except ImportError:
    GeoNamesClient = None

try:
    from .elevation_client import ElevationClient
except ImportError:
    ElevationClient = None

try:
    from .climate_analyzer import ClimateAnalyzer
except ImportError:
    ClimateAnalyzer = None

try:
    from .osm_feature_matcher import OSMFeatureMatcher
except ImportError:
    OSMFeatureMatcher = None

try:
    from .image_embeddings import ImageEmbedder
except ImportError:
    ImageEmbedder = None

try:
    from .evidence_fusion import EvidenceFusion
except ImportError:
    EvidenceFusion = None

try:
    from .visual_geo_engine import VisualGeoEngine
except ImportError:
    VisualGeoEngine = None

try:
    from .geoclip_predictor import GeoCLIPPredictor
except ImportError:
    GeoCLIPPredictor = None

try:
    from .streetclip_predictor import StreetCLIPPredictor
except ImportError:
    StreetCLIPPredictor = None

try:
    from .vlm_geo_analyzer import vlm_analyze, vlm_geo_estimates
except ImportError:
    vlm_analyze = None
    vlm_geo_estimates = None

try:
    from .country_matcher import get_iso_code, countries_match, extract_country_from_location
except ImportError:
    get_iso_code = countries_match = extract_country_from_location = None

try:
    from .geo_math import haversine_distance, bounding_box, weighted_centroid, validate_coordinates
except ImportError:
    haversine_distance = bounding_box = weighted_centroid = validate_coordinates = None

try:
    from .evidence_chain import Evidence, EvidenceChain, EvidenceSource
except ImportError:
    Evidence = EvidenceChain = EvidenceSource = None

try:
    from .web_search_providers import search_web, get_available_providers, SerperClient, BraveClient, SearXNGClient
except ImportError:
    search_web = get_available_providers = SerperClient = BraveClient = SearXNGClient = None

try:
    from .geo_scorer import GeoScorer
    from .scoring_config import ScoringConfig
except ImportError:
    GeoScorer = ScoringConfig = None

try:
    from .grounding import GroundingEngine, GroundingResult, GroundingVerdict
except ImportError:
    GroundingEngine = GroundingResult = GroundingVerdict = None

try:
    from .geo_hierarchy import GeoHierarchyResolver
except ImportError:
    GeoHierarchyResolver = None