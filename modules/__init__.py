# Optional heavy/ML submodules are imported defensively so a broken optional
# dependency (e.g. the numpy2/scipy binary mismatch in easyocr) degrades to
# None for that one module instead of crashing the whole package.
from .satellite_matcher import SatelliteMatcher, SatelliteMatch
from .location_reasoner import LocationReasoner, LocationEstimate
from .deep_analyzer import DeepAnalyzer
from .google_maps_controller import GoogleMapsController, MapsVerificationResult
from .park_finder import find_nearby_parks
from .browser_automation import BrowserAutomation

try:
    from .scene_classifier import SceneClassifier
except Exception:
    SceneClassifier = None

try:
    from .sign_detector import SignDetector
except Exception:
    SignDetector = None

try:
    from .vehicle_detector import VehicleDetector
except Exception:
    VehicleDetector = None

try:
    from .vegetation_classifier import VegetationClassifier
except Exception:
    VegetationClassifier = None

try:
    from .terrain_analyzer import TerrainAnalyzer
except Exception:
    TerrainAnalyzer = None

try:
    from .geonames_client import GeoNamesClient
except Exception:
    GeoNamesClient = None

try:
    from .elevation_client import ElevationClient
except Exception:
    ElevationClient = None

try:
    from .climate_analyzer import ClimateAnalyzer
except Exception:
    ClimateAnalyzer = None

try:
    from .osm_feature_matcher import OSMFeatureMatcher
except Exception:
    OSMFeatureMatcher = None

try:
    from .image_embeddings import ImageEmbedder
except Exception:
    ImageEmbedder = None

try:
    from .evidence_fusion import EvidenceFusion
except Exception:
    EvidenceFusion = None

try:
    from .visual_geo_engine import VisualGeoEngine
except Exception:
    VisualGeoEngine = None

try:
    from .geoclip_predictor import GeoCLIPPredictor
except Exception:
    GeoCLIPPredictor = None

try:
    from .streetclip_predictor import StreetCLIPPredictor
except Exception:
    StreetCLIPPredictor = None

try:
    from .vlm_geo_analyzer import vlm_analyze, vlm_geo_estimates
except Exception:
    vlm_analyze = None
    vlm_geo_estimates = None

try:
    from .country_matcher import get_iso_code, countries_match, extract_country_from_location
except Exception:
    get_iso_code = countries_match = extract_country_from_location = None

try:
    from .geo_math import haversine_distance, bounding_box, weighted_centroid, validate_coordinates
except Exception:
    haversine_distance = bounding_box = weighted_centroid = validate_coordinates = None

try:
    from .evidence_chain import Evidence, EvidenceChain, EvidenceSource
except Exception:
    Evidence = EvidenceChain = EvidenceSource = None

try:
    from .web_search_providers import search_web, get_available_providers, SerperClient, BraveClient, SearXNGClient
except Exception:
    search_web = get_available_providers = SerperClient = BraveClient = SearXNGClient = None

try:
    from .geo_scorer import GeoScorer
    from .scoring_config import ScoringConfig
except Exception:
    GeoScorer = ScoringConfig = None

try:
    from .grounding import GroundingEngine, GroundingResult, GroundingVerdict
except Exception:
    GroundingEngine = GroundingResult = GroundingVerdict = None

try:
    from .geo_hierarchy import HierarchicalResolver
except Exception:
    HierarchicalResolver = None

try:
    from .browser_stealth import StealthConfig
except Exception:
    StealthConfig = None

try:
    from .visual_similarity import VisualSimilarityScorer
except Exception:
    VisualSimilarityScorer = None

try:
    from .osv5m_predictor import OSV5MPredictor
except Exception:
    OSV5MPredictor = None

try:
    from .candidate_visual_verification import verify_candidates, fetch_reference_photos
except Exception:
    verify_candidates = fetch_reference_photos = None