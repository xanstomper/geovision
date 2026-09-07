"""
Vision Clue Resolver — Bridge from Cloud Vision AI Models to Deterministic GIS
=============================================================================
Allows any multimodal AI (Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro, Hermes)
that possesses computer vision to pass its visual deductions (spotted street signs,
store names, city/country hints, driving side, architectural cues, park proximity)
into GeoVision's deterministic GIS engine.

GeoVision then executes:
1. City/region geocoding and bounding box grounding (Nominatim + GeoNames)
2. Specific street/intersection/landmark forward resolution
3. Overpass QL spatial queries for amenities and chain stores within the area
4. Park proximity spatial join
5. Offline GeoNames KD-tree city snap validation
6. Satellite landcover (ESRI/ArcGIS World Imagery) vegetation & density verification
7. Honest calibrated confidence scoring and structured evidence chain
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance in km between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 6371.0088 * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class VisionClueResolver:
    """Resolves visual clues extracted by an external vision model to exact GPS coordinates."""

    def __init__(self):
        from modules.nominatim_geocoder import NominatimGeocoder
        from modules.geonames_city_snap import get_city_index
        from modules.overpass_client import OverpassClient
        from modules.satellite_matcher import SatelliteMatcher
        from modules.chain_store_locator import ChainStoreLocator

        self.geocoder = NominatimGeocoder()
        self.city_index = get_city_index()
        self.overpass = OverpassClient()
        self.sat_matcher = SatelliteMatcher()
        self.chain_locator = ChainStoreLocator()

    def resolve(self, clues: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a dictionary of visual observations into ranked GPS coordinates."""
        country_hint = (clues.get("country_hint") or clues.get("country") or "").strip()
        city_hint = (clues.get("city_hint") or clues.get("city") or "").strip()
        state_hint = (clues.get("state_or_province") or clues.get("state") or clues.get("region") or "").strip()
        near_park = bool(clues.get("near_park", False))
        driving_side = (clues.get("driving_side") or "").lower().strip()

        # Handle street names / landmarks
        streets = clues.get("street_names") or clues.get("streets") or []
        if isinstance(streets, str):
            streets = [streets]

        # Handle detected text
        text_lines = clues.get("detected_text") or clues.get("text") or []
        if isinstance(text_lines, str):
            text_lines = [text_lines]

        # Handle chain stores
        chain_stores = clues.get("chain_stores") or []
        if isinstance(chain_stores, str):
            chain_stores = [chain_stores]

        # Handle amenities
        amenities = clues.get("amenities") or []
        if isinstance(amenities, str):
            amenities = [amenities]

        search_radius_m = int(clues.get("radius_meters") or clues.get("search_radius_m") or 35000)

        # -------------------------------------------------------------------
        # Phase 1: Ground City / Region
        # -------------------------------------------------------------------
        city_coords = None
        city_meta = {}
        city_query_parts = [p for p in [city_hint, state_hint, country_hint] if p]
        city_query_str = ", ".join(city_query_parts)

        if city_query_str:
            geo_hit = self.geocoder.forward_geocode(city_query_str)
            if geo_hit:
                city_coords = (float(geo_hit["latitude"]), float(geo_hit["longitude"]))
                city_meta = {
                    "display_name": geo_hit.get("display_name"),
                    "latitude": city_coords[0],
                    "longitude": city_coords[1],
                    "boundingbox": geo_hit.get("boundingbox"),
                }
            elif city_hint:
                # Fallback to city name only
                geo_hit = self.geocoder.forward_geocode(city_hint)
                if geo_hit:
                    city_coords = (float(geo_hit["latitude"]), float(geo_hit["longitude"]))
                    city_meta = {
                        "display_name": geo_hit.get("display_name"),
                        "latitude": city_coords[0],
                        "longitude": city_coords[1],
                        "boundingbox": geo_hit.get("boundingbox"),
                    }

        candidates: List[Dict[str, Any]] = []

        # -------------------------------------------------------------------
        # Phase 2: Street & Landmark Forward Geocoding
        # -------------------------------------------------------------------
        for street in streets:
            street_str = street.strip()
            if not street_str:
                continue
            queries_to_try = []
            if city_hint:
                queries_to_try.append(f"{street_str}, {city_hint}")
            if city_hint and country_hint:
                queries_to_try.append(f"{street_str}, {city_hint}, {country_hint}")
            queries_to_try.append(street_str)

            for q in queries_to_try:
                hit = self.geocoder.forward_geocode(q)
                if hit:
                    lat, lon = float(hit["latitude"]), float(hit["longitude"])
                    # Check distance if city coords known
                    dist_to_city = _haversine_km(city_coords[0], city_coords[1], lat, lon) if city_coords else 0.0
                    if not city_coords or dist_to_city <= (search_radius_m / 1000.0 * 1.5):
                        candidates.append({
                            "latitude": lat,
                            "longitude": lon,
                            "source": "street_geocode",
                            "matched_query": q,
                            "display_name": hit.get("display_name"),
                            "raw_confidence": 0.82,
                            "details": f"Street / intersection match for '{street_str}'",
                        })
                        break

        # -------------------------------------------------------------------
        # Phase 3: Chain Store & Business Search via Overpass
        # -------------------------------------------------------------------
        all_store_queries = list(set(chain_stores + [t for t in text_lines if len(t) > 3 and not any(t in s for s in streets)]))
        if all_store_queries and city_coords:
            try:
                cs_hits = self.chain_locator.locate_chain_stores(
                    all_store_queries[:4],
                    region_hint_lat=city_coords[0],
                    region_hint_lon=city_coords[1],
                )
                for cs in cs_hits[:5]:
                    candidates.append({
                        "latitude": cs["latitude"],
                        "longitude": cs["longitude"],
                        "source": "chain_store_osm",
                        "matched_query": cs["store_name"],
                        "display_name": f"{cs['osm_name']} ({cs['store_name']})",
                        "raw_confidence": 0.88,
                        "details": f"OSM exact store entity for '{cs['store_name']}'",
                    })
            except Exception as e:
                logger.warning("Chain store lookup failed: %s", e)

        # -------------------------------------------------------------------
        # Phase 4: Amenity Search near City Center
        # -------------------------------------------------------------------
        if amenities and city_coords and len(candidates) < 3:
            for am in amenities[:3]:
                try:
                    am_hits = self.overpass.search_amenities_by_name(
                        am, city_coords[0], city_coords[1], radius_meters=search_radius_m
                    )
                    for h in am_hits[:3]:
                        candidates.append({
                            "latitude": h["lat"],
                            "longitude": h["lon"],
                            "source": "amenity_osm",
                            "matched_query": am,
                            "display_name": f"{h['name']} ({h['type']})",
                            "raw_confidence": 0.65,
                            "details": f"Amenity '{am}' found in target city",
                        })
                except Exception as e:
                    logger.warning("Overpass amenity search error: %s", e)

        # -------------------------------------------------------------------
        # Phase 5: Fallback to City Center if no micro-location found
        # -------------------------------------------------------------------
        if not candidates and city_coords:
            candidates.append({
                "latitude": city_coords[0],
                "longitude": city_coords[1],
                "source": "city_centroid",
                "matched_query": city_query_str,
                "display_name": city_meta.get("display_name", city_query_str),
                "raw_confidence": 0.45,
                "details": f"City-level grounding for '{city_query_str}'",
            })

        # -------------------------------------------------------------------
        # Phase 6: Validate & Enhance Candidates with GIS Evidence
        # -------------------------------------------------------------------
        enhanced_candidates = []
        for cand in candidates:
            lat = cand["latitude"]
            lon = cand["longitude"]
            conf = cand["raw_confidence"]
            evidence = [cand["details"]]

            # 1. Offline GeoNames city snap
            nearest_city_snap = None
            try:
                snap = self.city_index.snap(lat, lon, max_km=75.0)
                if snap:
                    nearest_city_snap = {
                        "name": snap["name"],
                        "country_code": snap["country_code"],
                        "distance_km": snap["distance_km"],
                        "population": snap.get("population", 0),
                    }
                    if city_hint and snap["name"].lower() in city_hint.lower():
                        conf = min(0.98, conf + 0.05)
                        evidence.append(f"GeoNames snap confirmed city '{snap['name']}' ({snap['distance_km']}km)")
                    elif country_hint and snap["country_code"].lower() == country_hint.lower():
                        evidence.append(f"GeoNames snap confirmed country '{snap['country_code']}'")
            except Exception:
                pass

            # 2. Park proximity corroboration
            nearby_parks = []
            if near_park:
                try:
                    parks = self.overpass.find_nearby_parks(lat, lon, radius_meters=1500)
                    if parks:
                        nearby_parks = [{"name": p["name"], "distance_m": round(_haversine_km(lat, lon, p["lat"], p["lon"]) * 1000)} for p in parks[:3]]
                        conf = min(0.98, conf + 0.10)
                        evidence.append(f"Corroborated: {len(parks)} park(s) within 1.5km (nearest: {parks[0]['name']})")
                    else:
                        conf = max(0.20, conf - 0.15)
                        evidence.append("Warning: no park found within 1.5km despite visual observation")
                except Exception:
                    pass

            # 3. Satellite landcover texture check
            sat_info = {}
            try:
                sat_tile = self.sat_matcher.fetch_satellite_tile(lat, lon, zoom=18)
                if sat_tile is not None:
                    import cv2
                    import numpy as np
                    hsv = cv2.cvtColor(sat_tile, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
                    green_ratio = float(np.count_nonzero(mask) / (sat_tile.shape[0] * sat_tile.shape[1]))
                    landcover = "urban_built_up"
                    if green_ratio > 0.40:
                        landcover = "rural_or_park"
                    elif green_ratio > 0.15:
                        landcover = "suburban"
                    sat_info = {
                        "classification": landcover,
                        "green_ratio": round(green_ratio, 3),
                        "verified": True,
                    }
                    evidence.append(f"Satellite landcover: {landcover} (green ratio: {green_ratio:.1%})")
            except Exception:
                sat_info = {"verified": False}

            # Final calibrated confidence
            conf = round(min(0.98, max(0.10, conf)), 3)

            enhanced_candidates.append({
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "confidence": conf,
                "display_name": cand.get("display_name"),
                "source": cand.get("source"),
                "matched_query": cand.get("matched_query"),
                "nearest_city": nearest_city_snap,
                "satellite": sat_info,
                "nearby_parks": nearby_parks,
                "evidence": evidence,
                "google_maps_url": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
                "osm_url": f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}",
            })

        # Apply spatial constraint solver (elimination of physical impossibilities)
        try:
            from modules.spatial_constraint_solver import SpatialConstraintSolver
            solver = SpatialConstraintSolver()
            enhanced_candidates = solver.filter_and_rerank_estimates(enhanced_candidates, clues)
        except Exception as e:
            logger.warning(f"SpatialConstraintSolver in vision clue resolver failed: {e}")

        enhanced_candidates.sort(key=lambda c: c["confidence"], reverse=True)

        return {
            "status": "success" if enhanced_candidates else "no_match",
            "query_clues": clues,
            "city_grounding": city_meta,
            "candidate_count": len(enhanced_candidates),
            "best_estimate": enhanced_candidates[0] if enhanced_candidates else None,
            "candidates": enhanced_candidates[:5],
        }
