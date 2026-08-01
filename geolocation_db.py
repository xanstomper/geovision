#!/usr/bin/env python3
"""
geolocation_db.py — Comprehensive Geolocation Database for GeoVision.

Provides a rich dataset of North American cities, their architectural DNA,
building materials, vegetation, and public parks.  Designed to match visual
features observed in street-level imagery to probable city / neighbourhood
matches.
"""

import math
from typing import Dict, List, Optional, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

R_EARTH_KM = 6371.0  # mean Earth radius, km


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two points."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    )
    return R_EARTH_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _walk_time_minutes(distance_km: float, pace_kmh: float = 5.0) -> float:
    """Convert distance to approximate walking minutes at given pace."""
    return (distance_km / pace_kmh) * 60.0


# ── core data ────────────────────────────────────────────────────────────────

CITY_DATA: Dict[str, dict] = {
    "Toronto": {
        "province": "ON",
        "country": "Canada",
        "lat": 43.65,
        "lng": -79.38,
        "timezone": "America/Toronto",
        "climate_zone": "Humid continental (Dfb)",
        "architectural_styles": {
            "primary": ["Victorian", "Edwardian", "Toronto brick apartments", "modern condos"],
            "secondary": ["Georgian Revival", "Art Deco", "International Style", "Neo-Gothic"],
        },
        "building_materials": {
            "common": ["red brick", "limestone", "concrete", "glass"],
            "roofing": ["asphalt shingle", "slate", "green roof"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["Norway maple", "sugar maple", "red oak", "honey locust", "white spruce", "eastern white cedar"],
            "common_plants": ["hosta", "daylily", "hydrangea", "black-eyed Susan"],
            "climate_zone": "USDA 5b-6a",
        },
        "road_infrastructure": {
            "signage_style": "green highway signs with white border, standard Ontario MUTCD",
            "utility_poles": "wooden, brown/grey, often with multiple crossarms",
            "streetlights": "cobra-head LEDs on metal poles; heritage-style decorative in older neighbourhoods",
            "crosswalk_signs": "standard Ontario pedestrian crossing signs (white figure on blue)",
        },
        "parks": [
            {"name": "High Park", "lat": 43.6465, "lng": -79.4637, "area_ha": 161},
            {"name": "Trinity Bellwoods Park", "lat": 43.6470, "lng": -79.4110, "area_ha": 14},
            {"name": "Queen's Park", "lat": 43.6623, "lng": -79.3914, "area_ha": 10},
            {"name": "Riverdale Park", "lat": 43.6660, "lng": -79.3500, "area_ha": 22},
            {"name": "Allan Gardens", "lat": 43.6550, "lng": -79.3810, "area_ha": 4},
            {"name": "Christie Pits", "lat": 43.6660, "lng": -79.4230, "area_ha": 7},
            {"name": "Dufferin Grove Park", "lat": 43.6560, "lng": -79.4290, "area_ha": 5},
            {"name": "Withrow Park", "lat": 43.6700, "lng": -79.3530, "area_ha": 5},
            {"name": "Centre Island (Toronto Islands)", "lat": 43.6200, "lng": -79.3700, "area_ha": 230},
            {"name": "Cedarvale Park", "lat": 43.6890, "lng": -79.4190, "area_ha": 12},
        ],
    },
    "Chicago": {
        "province": "IL",
        "country": "USA",
        "lat": 41.88,
        "lng": -87.63,
        "timezone": "America/Chicago",
        "climate_zone": "Humid continental (Dfa)",
        "architectural_styles": {
            "primary": ["Chicago bungalow", "greystone", "brick two-flat", "high-rise modernist"],
            "secondary": ["Prairie School", "Chicago School", "Art Deco", "Neo-Gothic"],
        },
        "building_materials": {
            "common": ["red brick", "limestone", "concrete", "terracotta", "glass"],
            "roofing": ["asphalt shingle", "flat tar roof", "slate (historic)"],
            "sidewalk": ["concrete", "flagstone"],
        },
        "vegetation": {
            "tree_types": ["honey locust", "green ash", "silver maple", "London plane", "crabapple", "littleleaf linden"],
            "common_plants": ["purple coneflower", "black-eyed Susan", "hosta", "daylily"],
            "climate_zone": "USDA 5b-6a",
        },
        "road_infrastructure": {
            "signage_style": "green federal highway signs, white reflective text, Illinois MUTCD",
            "utility_poles": "wooden, grey/brown, often with crossarms and transformers",
            "streetlights": "cobra-head LEDs on metal or concrete poles; decorative in downtown",
            "crosswalk_signs": "standard US pedestrian crossing signs (white figure on yellow-green)",
        },
        "parks": [
            {"name": "Millennium Park", "lat": 41.8826, "lng": -87.6226, "area_ha": 10},
            {"name": "Lincoln Park", "lat": 41.9210, "lng": -87.6340, "area_ha": 490},
            {"name": "Grant Park", "lat": 41.8760, "lng": -87.6188, "area_ha": 129},
            {"name": "Jackson Park", "lat": 41.7830, "lng": -87.5800, "area_ha": 210},
            {"name": "Humboldt Park", "lat": 41.9040, "lng": -87.7010, "area_ha": 85},
            {"name": "Garfield Park", "lat": 41.8830, "lng": -87.7140, "area_ha": 46},
            {"name": "Douglass Park", "lat": 41.8620, "lng": -87.6940, "area_ha": 55},
            {"name": "Burnham Park", "lat": 41.8300, "lng": -87.6120, "area_ha": 240},
            {"name": "Washington Park", "lat": 41.7880, "lng": -87.6160, "area_ha": 150},
            {"name": "Marquette Park", "lat": 41.7710, "lng": -87.7020, "area_ha": 86},
        ],
    },
    "New York": {
        "province": "NY",
        "country": "USA",
        "lat": 40.71,
        "lng": -74.00,
        "timezone": "America/New_York",
        "climate_zone": "Humid subtropical (Cfa)",
        "architectural_styles": {
            "primary": ["Brownstone", "pre-war apartment", "modern high-rise", "brick row house"],
            "secondary": ["Art Deco", "Beaux-Arts", "Neo-Gothic", "International Style"],
        },
        "building_materials": {
            "common": ["brownstone (sandstone)", "brick", "limestone", "glass", "steel", "concrete"],
            "roofing": ["flat tar/gravel", "green roof (newer)", "slate (historic)"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["London plane", "Callery pear", "honey locust", "red maple", "pin oak", "Japanese zelkova"],
            "common_plants": ["ivy", "boxwood", "hydrangea", "hosta", "fern"],
            "climate_zone": "USDA 7a-7b",
        },
        "road_infrastructure": {
            "signage_style": "green NYC DOT street signs (white text on green), yellow pedestrian crossing signs",
            "utility_poles": "metal or wooden, often crowded with wires; many underground utilities in Manhattan",
            "streetlights": "decorative cast-iron (historic), cobra-head LEDs, or modern LED columns",
            "crosswalk_signs": "standard NYC pedestrian signal (orange/white hand and figure)",
        },
        "parks": [
            {"name": "Central Park", "lat": 40.7829, "lng": -73.9654, "area_ha": 341},
            {"name": "Prospect Park", "lat": 40.6602, "lng": -73.9690, "area_ha": 212},
            {"name": "Bryant Park", "lat": 40.7536, "lng": -73.9832, "area_ha": 3.9},
            {"name": "Madison Square Park", "lat": 40.7420, "lng": -73.9876, "area_ha": 2.8},
            {"name": "Washington Square Park", "lat": 40.7308, "lng": -73.9973, "area_ha": 3.9},
            {"name": "Hudson River Park", "lat": 40.7290, "lng": -74.0120, "area_ha": 220},
            {"name": "Riverside Park", "lat": 40.7950, "lng": -73.9720, "area_ha": 100},
            {"name": "Battery Park", "lat": 40.7033, "lng": -74.0170, "area_ha": 10},
            {"name": "Flushing Meadows-Corona Park", "lat": 40.7500, "lng": -73.8450, "area_ha": 359},
            {"name": "Van Cortlandt Park", "lat": 40.8970, "lng": -73.8870, "area_ha": 460},
        ],
    },
    "Montreal": {
        "province": "QC",
        "country": "Canada",
        "lat": 45.50,
        "lng": -73.57,
        "timezone": "America/Montreal",
        "climate_zone": "Humid continental (Dfb)",
        "architectural_styles": {
            "primary": ["Plexes (triplex/quadruplex)", "grey stone row houses", "brick", "modern condos"],
            "secondary": ["Art Deco", "Second Empire", "Neo-Gothic", "International Style"],
        },
        "building_materials": {
            "common": ["grey limestone", "red brick", "concrete", "glass", "stucco"],
            "roofing": ["tin sloped roofs (historic)", "asphalt shingle", "green roof"],
            "sidewalk": ["concrete", "stone slabs", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["Norway maple", "silver maple", "hackberry", "littleleaf linden", "Colorado spruce", "white birch"],
            "common_plants": ["impatiens", "begonia", "hydrangea", "lilac", "daylily"],
            "climate_zone": "USDA 4b-5b",
        },
        "road_infrastructure": {
            "signage_style": "green Quebec highway signs; blue street signs with white text (arrondissement-specific)",
            "utility_poles": "wooden, often grey; some neighbourhoods with buried hydro",
            "streetlights": "cobra-head on metal poles; heritage-style in Old Montreal (gas-lamp replicas)",
            "crosswalk_signs": "standard Quebec pedestrian crossing (white figure on blue, bilingual labels)",
        },
        "parks": [
            {"name": "Mount Royal Park", "lat": 45.5089, "lng": -73.5877, "area_ha": 190},
            {"name": "Parc La Fontaine", "lat": 45.5240, "lng": -73.5670, "area_ha": 14},
            {"name": "Parc du Mont-Royal", "lat": 45.5080, "lng": -73.5870, "area_ha": 280},
            {"name": "Parc Jean-Drapeau", "lat": 45.5080, "lng": -73.5300, "area_ha": 205},
            {"name": "Parc Jarry", "lat": 45.5340, "lng": -73.6270, "area_ha": 15},
            {"name": "Parc Laurier", "lat": 45.5260, "lng": -73.5760, "area_ha": 5},
            {"name": "Parc Angrignon", "lat": 45.4420, "lng": -73.6000, "area_ha": 97},
            {"name": "Parc du Plateau", "lat": 45.5180, "lng": -73.5750, "area_ha": 3},
            {"name": "Parc Outremont", "lat": 45.5200, "lng": -73.6070, "area_ha": 6},
            {"name": "Parc Ahuntsic", "lat": 45.5500, "lng": -73.6700, "area_ha": 18},
        ],
    },
    "Vancouver": {
        "province": "BC",
        "country": "Canada",
        "lat": 49.28,
        "lng": -123.12,
        "timezone": "America/Vancouver",
        "climate_zone": "Oceanic (Cfb)",
        "architectural_styles": {
            "primary": ["West Coast style", "stucco box", "glass tower", "Vancouver special"],
            "secondary": ["Craftsman", "Queen Anne Revival", "Modernist", "Heritage"],
        },
        "building_materials": {
            "common": ["stucco", "glass", "concrete", "wood", "brick", "stone veneer"],
            "roofing": ["asphalt shingle", "cedar shake", "green roof", "flat membrane"],
            "sidewalk": ["concrete", "pavers", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["Douglas fir", "western red cedar", "arbutus", "Japanese maple", "flowering cherry", "bigleaf maple"],
            "common_plants": ["rhododendron", "azalea", "ferns", "heather", "hydrangea", "bamboo"],
            "climate_zone": "USDA 8a-8b",
        },
        "road_infrastructure": {
            "signage_style": "green BC highway signs; Vancouver-style green street signs with white text",
            "utility_poles": "wooden, often stained dark brown; some areas with buried utilities",
            "streetlights": "modern LED cobra-head on metal poles; decorative gas-lamp-style in heritage areas",
            "crosswalk_signs": "standard BC pedestrian crossing signs",
        },
        "parks": [
            {"name": "Stanley Park", "lat": 49.3043, "lng": -123.1445, "area_ha": 405},
            {"name": "Queen Elizabeth Park", "lat": 49.2450, "lng": -123.1130, "area_ha": 52},
            {"name": "Vanier Park", "lat": 49.2770, "lng": -123.1320, "area_ha": 10},
            {"name": "Pacific Spirit Regional Park", "lat": 49.2630, "lng": -123.2000, "area_ha": 874},
            {"name": "Jericho Beach Park", "lat": 49.2720, "lng": -123.1920, "area_ha": 18},
            {"name": "Kitsilano Beach Park", "lat": 49.2720, "lng": -123.1530, "area_ha": 7},
            {"name": "David Lam Park", "lat": 49.2740, "lng": -123.1210, "area_ha": 4},
            {"name": "Coal Harbour Park", "lat": 49.2900, "lng": -123.1200, "area_ha": 3},
            {"name": "Trout Lake (John Hendry Park)", "lat": 49.2600, "lng": -123.0650, "area_ha": 10},
            {"name": "New Brighton Park", "lat": 49.2900, "lng": -123.0350, "area_ha": 8},
        ],
    },
    "Boston": {
        "province": "MA",
        "country": "USA",
        "lat": 42.36,
        "lng": -71.06,
        "timezone": "America/New_York",
        "climate_zone": "Humid subtropical/continental (Cfa/Dfa)",
        "architectural_styles": {
            "primary": ["Colonial", "brick row house", "brownstone", "triple-decker"],
            "secondary": ["Federal", "Greek Revival", "Victorian", "Modernist"],
        },
        "building_materials": {
            "common": ["red brick", "brownstone", "granite", "wood clapboard", "concrete", "glass"],
            "roofing": ["asphalt shingle", "slate (historic)", "flat membrane"],
            "sidewalk": ["brick", "concrete", "asphalt", "granite curbs"],
        },
        "vegetation": {
            "tree_types": ["red maple", "sugar maple", "oak", "American elm", "London plane", "flowering dogwood"],
            "common_plants": ["hosta", "impatiens", "ivy", "rhododendron", "azalea"],
            "climate_zone": "USDA 6a-6b",
        },
        "road_infrastructure": {
            "signage_style": "green Massachusetts highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/grey; many historic districts with buried power",
            "streetlights": "decorative cast-iron (historic district), cobra-head on main roads",
            "crosswalk_signs": "standard Massachusetts pedestrian crossing signs",
        },
        "parks": [
            {"name": "Boston Common", "lat": 42.3550, "lng": -71.0656, "area_ha": 20},
            {"name": "Public Garden", "lat": 42.3540, "lng": -71.0700, "area_ha": 10},
            {"name": "Charles River Esplanade", "lat": 42.3570, "lng": -71.0770, "area_ha": 26},
            {"name": "Arnold Arboretum", "lat": 42.2980, "lng": -71.1220, "area_ha": 107},
            {"name": "Franklin Park", "lat": 42.3050, "lng": -71.0870, "area_ha": 230},
            {"name": "Jamaica Pond", "lat": 42.3200, "lng": -71.1150, "area_ha": 28},
            {"name": "Christopher Columbus Park", "lat": 42.3610, "lng": -71.0500, "area_ha": 3},
            {"name": "Rose Kennedy Greenway", "lat": 42.3600, "lng": -71.0530, "area_ha": 6},
            {"name": "Castle Island (Pleasure Bay)", "lat": 42.3370, "lng": -71.0120, "area_ha": 10},
            {"name": "Millennium Park", "lat": 42.3030, "lng": -71.1650, "area_ha": 40},
        ],
    },
    "Detroit": {
        "province": "MI",
        "country": "USA",
        "lat": 42.33,
        "lng": -83.05,
        "timezone": "America/Detroit",
        "climate_zone": "Humid continental (Dfa)",
        "architectural_styles": {
            "primary": ["brick bungalow", "abandoned industrial", "Art Deco", "mid-century"],
            "secondary": ["Victorian", "Queen Anne", "Colonial Revival", "Modernist"],
        },
        "building_materials": {
            "common": ["red brick", "concrete", "steel", "glass", "limestone"],
            "roofing": ["asphalt shingle", "flat membrane", "slate (historic)"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["silver maple", "Norway maple", "American elm", "red oak", "honey locust", "crabapple"],
            "common_plants": ["hosta", "daylily", "lilac", "black-eyed Susan"],
            "climate_zone": "USDA 5b-6a",
        },
        "road_infrastructure": {
            "signage_style": "green Michigan highway signs; white-on-green street signs",
            "utility_poles": "wooden, grey, often with multiple crossarms",
            "streetlights": "cobra-head on metal poles; some historic decorative in downtown",
            "crosswalk_signs": "standard Michigan pedestrian crossing signs",
        },
        "parks": [
            {"name": "Belle Isle Park", "lat": 42.3420, "lng": -82.9750, "area_ha": 400},
            {"name": "Campus Martius Park", "lat": 42.3290, "lng": -83.0450, "area_ha": 2},
            {"name": "Hart Plaza", "lat": 42.3260, "lng": -83.0420, "area_ha": 5},
            {"name": "Palmer Park", "lat": 42.4180, "lng": -83.1120, "area_ha": 113},
            {"name": "Rouge Park", "lat": 42.3610, "lng": -83.2450, "area_ha": 493},
            {"name": "River Rouge Park", "lat": 42.3530, "lng": -83.2370, "area_ha": 400},
            {"name": "Eliza Howell Park", "lat": 42.4200, "lng": -83.2400, "area_ha": 97},
            {"name": "Chandler Park", "lat": 42.4100, "lng": -82.9650, "area_ha": 73},
            {"name": "Clark Park", "lat": 42.3130, "lng": -83.1430, "area_ha": 10},
            {"name": "Grand Circus Park", "lat": 42.3350, "lng": -83.0500, "area_ha": 2},
        ],
    },
    "Philadelphia": {
        "province": "PA",
        "country": "USA",
        "lat": 39.95,
        "lng": -75.17,
        "timezone": "America/New_York",
        "climate_zone": "Humid subtropical (Cfa)",
        "architectural_styles": {
            "primary": ["brick row house", "brownstone", "colonial", "modern high-rise"],
            "secondary": ["Federal", "Greek Revival", "Victorian", "Art Deco"],
        },
        "building_materials": {
            "common": ["red brick", "brownstone", "concrete", "glass", "limestone", "granite"],
            "roofing": ["asphalt shingle", "slate (historic)", "flat membrane"],
            "sidewalk": ["brick", "concrete", "asphalt", "flagstone"],
        },
        "vegetation": {
            "tree_types": ["London plane", "red maple", "pin oak", "American elm", "flowering dogwood", "kousa dogwood"],
            "common_plants": ["hosta", "ivy", "boxwood", "hydrangea", "daylily"],
            "climate_zone": "USDA 6b-7a",
        },
        "road_infrastructure": {
            "signage_style": "green Pennsylvania highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/grey; some buried in historic districts",
            "streetlights": "decorative cobra-head or LED on metal poles; historic gas-lamp replicas in Society Hill",
            "crosswalk_signs": "standard Pennsylvania pedestrian crossing signs",
        },
        "parks": [
            {"name": "Fairmount Park", "lat": 39.9800, "lng": -75.1950, "area_ha": 810},
            {"name": "Rittenhouse Square", "lat": 39.9490, "lng": -75.1720, "area_ha": 2.5},
            {"name": "Washington Square", "lat": 39.9470, "lng": -75.1520, "area_ha": 2.5},
            {"name": "Franklin Square", "lat": 39.9560, "lng": -75.1480, "area_ha": 2.5},
            {"name": "Logan Square", "lat": 39.9580, "lng": -75.1710, "area_ha": 3},
            {"name": "FDR Park", "lat": 39.9020, "lng": -75.1800, "area_ha": 104},
            {"name": "Clark Park", "lat": 39.9450, "lng": -75.2100, "area_ha": 3.5},
            {"name": "Penn Treaty Park", "lat": 39.9660, "lng": -75.1280, "area_ha": 2.5},
            {"name": "Wissahickon Valley Park", "lat": 40.0500, "lng": -75.2100, "area_ha": 720},
            {"name": "Bartram's Garden", "lat": 39.9310, "lng": -75.2120, "area_ha": 18},
        ],
    },
    "San Francisco": {
        "province": "CA",
        "country": "USA",
        "lat": 37.77,
        "lng": -122.42,
        "timezone": "America/Los_Angeles",
        "climate_zone": "Mediterranean (Csb)",
        "architectural_styles": {
            "primary": ["Victorian (Painted Lady)", "Edwardian", "stucco", "modern glass"],
            "secondary": ["Spanish Colonial Revival", "Art Deco", "Mid-century Modern", "Futurist"],
        },
        "building_materials": {
            "common": ["wood siding", "stucco", "glass", "concrete", "brick", "stone"],
            "roofing": ["asphalt shingle", "clay tile (Spanish)", "slate", "flat membrane"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["Monterey cypress", "coast redwood", "eucalyptus", "flowering plum", "Japanese maple", "magnolia"],
            "common_plants": ["agapanthus", "birds of paradise", "bougainvillea", "fern", "ivy", "succulents"],
            "climate_zone": "USDA 10a-10b",
        },
        "road_infrastructure": {
            "signage_style": "green California highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/grey; many buried utilities in newer areas",
            "streetlights": "decorative historic (cast-iron) in downtown; cobra-head LEDs on major roads",
            "crosswalk_signs": "standard California pedestrian crossing signs",
        },
        "parks": [
            {"name": "Golden Gate Park", "lat": 37.7694, "lng": -122.4862, "area_ha": 412},
            {"name": "Dolores Park", "lat": 37.7596, "lng": -122.4270, "area_ha": 6},
            {"name": "Buena Vista Park", "lat": 37.7680, "lng": -122.4410, "area_ha": 15},
            {"name": "Crissy Field", "lat": 37.8050, "lng": -122.4540, "area_ha": 40},
            {"name": "Alamo Square Park", "lat": 37.7760, "lng": -122.4350, "area_ha": 2.5},
            {"name": "The Presidio", "lat": 37.7980, "lng": -122.4660, "area_ha": 600},
            {"name": "Yerba Buena Gardens", "lat": 37.7850, "lng": -122.4020, "area_ha": 2.5},
            {"name": "Washington Square (North Beach)", "lat": 37.8000, "lng": -122.4100, "area_ha": 1.5},
            {"name": "Lafayette Park", "lat": 37.7910, "lng": -122.4270, "area_ha": 5},
            {"name": "Glen Canyon Park", "lat": 37.7390, "lng": -122.4390, "area_ha": 26},
        ],
    },
    "Los Angeles": {
        "province": "CA",
        "country": "USA",
        "lat": 34.05,
        "lng": -118.24,
        "timezone": "America/Los_Angeles",
        "climate_zone": "Mediterranean (Csa)",
        "architectural_styles": {
            "primary": ["Spanish Colonial Revival", "mid-century modern", "stucco", "modern glass"],
            "secondary": ["Art Deco", "Mission Revival", "Googie", "Contemporary"],
        },
        "building_materials": {
            "common": ["stucco", "concrete", "glass", "wood", "brick", "stone veneer"],
            "roofing": ["clay tile (Spanish)", "asphalt shingle", "flat membrane", "tar and gravel"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["palm (Washingtonia)", "jacaranda", "eucalyptus", "ficus", "oleander", "coral tree"],
            "common_plants": ["bougainvillea", "birds of paradise", "agave", "succulents", "lantana", "plumbago"],
            "climate_zone": "USDA 9b-10b",
        },
        "road_infrastructure": {
            "signage_style": "green California highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/grey; some areas with buried utilities",
            "streetlights": "cobra-head LEDs on metal poles; decorative in downtown and Hollywood",
            "crosswalk_signs": "standard California pedestrian crossing signs",
        },
        "parks": [
            {"name": "Griffith Park", "lat": 34.1340, "lng": -118.2930, "area_ha": 1747},
            {"name": "Runyon Canyon Park", "lat": 34.1040, "lng": -118.3520, "area_ha": 53},
            {"name": "Echo Park", "lat": 34.0730, "lng": -118.2600, "area_ha": 12},
            {"name": "Grand Park", "lat": 34.0560, "lng": -118.2460, "area_ha": 5},
            {"name": "Pershing Square", "lat": 34.0500, "lng": -118.2520, "area_ha": 2},
            {"name": "Exposition Park", "lat": 34.0160, "lng": -118.2860, "area_ha": 64},
            {"name": "MacArthur Park", "lat": 34.0580, "lng": -118.2880, "area_ha": 13},
            {"name": "Barnsdall Art Park", "lat": 34.0990, "lng": -118.2940, "area_ha": 4},
            {"name": "Kenneth Hahn State Recreation Area", "lat": 33.9920, "lng": -118.3700, "area_ha": 103},
            {"name": "Pan Pacific Park", "lat": 34.0640, "lng": -118.3510, "area_ha": 10},
        ],
    },
    "Seattle": {
        "province": "WA",
        "country": "USA",
        "lat": 47.61,
        "lng": -122.33,
        "timezone": "America/Los_Angeles",
        "climate_zone": "Oceanic (Cfb)",
        "architectural_styles": {
            "primary": ["Craftsman", "modern", "NW contemporary", "mid-century"],
            "secondary": ["Victorian", "Queen Anne", "Tudor Revival", "International Style"],
        },
        "building_materials": {
            "common": ["wood (cedar, fir)", "stucco", "brick", "concrete", "glass", "stone"],
            "roofing": ["asphalt shingle", "cedar shake", "flat membrane", "green roof"],
            "sidewalk": ["concrete", "asphalt", "pavers"],
        },
        "vegetation": {
            "tree_types": ["Douglas fir", "western red cedar", "hemlock", "Japanese maple", "flowering cherry", "madrone"],
            "common_plants": ["rhododendron", "azalea", "fern", "salal", "oregon grape", "hosta"],
            "climate_zone": "USDA 8a-8b",
        },
        "road_infrastructure": {
            "signage_style": "green Washington state highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/dark grey; some buried utilities in newer areas",
            "streetlights": "cobra-head LEDs on metal poles; decorative in historic districts",
            "crosswalk_signs": "standard Washington pedestrian crossing signs",
        },
        "parks": [
            {"name": "Discovery Park", "lat": 47.6620, "lng": -122.4080, "area_ha": 215},
            {"name": "Volunteer Park", "lat": 47.6300, "lng": -122.3160, "area_ha": 18},
            {"name": "Green Lake Park", "lat": 47.6800, "lng": -122.3280, "area_ha": 104},
            {"name": "Gas Works Park", "lat": 47.6460, "lng": -122.3340, "area_ha": 8},
            {"name": "Kerry Park", "lat": 47.6290, "lng": -122.3590, "area_ha": 1},
            {"name": "Seward Park", "lat": 47.5470, "lng": -122.2600, "area_ha": 120},
            {"name": "Lincoln Park", "lat": 47.5300, "lng": -122.3930, "area_ha": 75},
            {"name": "Golden Gardens Park", "lat": 47.6890, "lng": -122.4030, "area_ha": 22},
            {"name": "Alki Beach Park", "lat": 47.5800, "lng": -122.4130, "area_ha": 5},
            {"name": "Cal Anderson Park", "lat": 47.6130, "lng": -122.3180, "area_ha": 2.5},
        ],
    },
    "Minneapolis": {
        "province": "MN",
        "country": "USA",
        "lat": 44.98,
        "lng": -93.27,
        "timezone": "America/Chicago",
        "climate_zone": "Humid continental (Dfa)",
        "architectural_styles": {
            "primary": ["brick", "wood frame", "Victorian", "mid-century modern"],
            "secondary": ["Colonial Revival", "Craftsman", "Prairie School", "Modernist"],
        },
        "building_materials": {
            "common": ["red brick", "wood siding", "concrete", "glass", "limestone", "stucco"],
            "roofing": ["asphalt shingle", "slate (historic)", "flat membrane"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["sugar maple", "red maple", "oak", "honey locust", "littleleaf linden", "Colorado blue spruce"],
            "common_plants": ["hosta", "daylily", "lilac", "black-eyed Susan", "Russian sage"],
            "climate_zone": "USDA 4a-4b",
        },
        "road_infrastructure": {
            "signage_style": "green Minnesota highway signs; white-on-green street signs",
            "utility_poles": "wooden, grey/brown; some buried in newer areas",
            "streetlights": "cobra-head LEDs on metal poles; decorative in downtown",
            "crosswalk_signs": "standard Minnesota pedestrian crossing signs",
        },
        "parks": [
            {"name": "Minnehaha Park", "lat": 44.9170, "lng": -93.2110, "area_ha": 77},
            {"name": "Bde Maka Ska", "lat": 44.9400, "lng": -93.3110, "area_ha": 142},
            {"name": "Lake Harriet", "lat": 44.9210, "lng": -93.3030, "area_ha": 108},
            {"name": "Loring Park", "lat": 44.9680, "lng": -93.2830, "area_ha": 13},
            {"name": "Lake of the Isles", "lat": 44.9550, "lng": -93.3040, "area_ha": 55},
            {"name": "Theodore Wirth Park", "lat": 44.9970, "lng": -93.3200, "area_ha": 265},
            {"name": "Powderhorn Park", "lat": 44.9430, "lng": -93.2550, "area_ha": 25},
            {"name": "Boom Island Park", "lat": 44.9890, "lng": -93.2590, "area_ha": 7},
            {"name": "Gold Medal Park", "lat": 44.9770, "lng": -93.2580, "area_ha": 3},
            {"name": "North Mississippi Regional Park", "lat": 45.0770, "lng": -93.2730, "area_ha": 120},
        ],
    },
    "Cleveland": {
        "province": "OH",
        "country": "USA",
        "lat": 41.50,
        "lng": -81.69,
        "timezone": "America/New_York",
        "climate_zone": "Humid continental (Dfa)",
        "architectural_styles": {
            "primary": ["Colonial", "brick", "Victorian", "mid-century"],
            "secondary": ["Craftsman", "Tudor Revival", "Art Deco", "Modernist"],
        },
        "building_materials": {
            "common": ["red brick", "limestone", "concrete", "glass", "wood siding"],
            "roofing": ["asphalt shingle", "slate (historic)", "flat membrane"],
            "sidewalk": ["concrete", "asphalt", "brick"],
        },
        "vegetation": {
            "tree_types": ["sugar maple", "red oak", "Norway maple", "American elm", "honey locust", "flowering crabapple"],
            "common_plants": ["hosta", "daylily", "lilac", "impatiens", "black-eyed Susan"],
            "climate_zone": "USDA 5b-6a",
        },
        "road_infrastructure": {
            "signage_style": "green Ohio highway signs; white-on-green street signs",
            "utility_poles": "wooden, grey/brown; some buried in downtown",
            "streetlights": "cobra-head LEDs on metal poles; decorative in historic districts",
            "crosswalk_signs": "standard Ohio pedestrian crossing signs",
        },
        "parks": [
            {"name": "Cleveland Metroparks Zoo", "lat": 41.4500, "lng": -81.7100, "area_ha": 73},
            {"name": "Edgewater Park", "lat": 41.4930, "lng": -81.7550, "area_ha": 40},
            {"name": "Gordon Park", "lat": 41.5100, "lng": -81.7000, "area_ha": 30},
            {"name": "Wade Oval (University Circle)", "lat": 41.5080, "lng": -81.6080, "area_ha": 5},
            {"name": "Rockefeller Park", "lat": 41.5200, "lng": -81.6220, "area_ha": 10},
            {"name": "Tremont Park", "lat": 41.4860, "lng": -81.6960, "area_ha": 3},
            {"name": "Detroit-Shoreway Park", "lat": 41.4880, "lng": -81.7370, "area_ha": 4},
            {"name": "Ohio City Park", "lat": 41.4830, "lng": -81.7100, "area_ha": 2},
            {"name": "Lake View Park", "lat": 41.5150, "lng": -81.6720, "area_ha": 8},
            {"name": "Cleveland Lakefront State Park", "lat": 41.5100, "lng": -81.6800, "area_ha": 100},
        ],
    },
    "Denver": {
        "province": "CO",
        "country": "USA",
        "lat": 39.74,
        "lng": -104.99,
        "timezone": "America/Denver",
        "climate_zone": "Semi-arid (BSk)",
        "architectural_styles": {
            "primary": ["mid-century modern", "brick ranch", "stucco", "contemporary"],
            "secondary": ["Victorian", "Queen Anne", "Craftsman", "Tudor Revival"],
        },
        "building_materials": {
            "common": ["red brick", "stucco", "concrete", "glass", "stone veneer", "wood"],
            "roofing": ["asphalt shingle", "clay tile", "flat membrane", "metal"],
            "sidewalk": ["concrete", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["honey locust", "green ash", "cottonwood", "blue spruce", "ponderosa pine", "crabapple"],
            "common_plants": ["yarrow", "Russian sage", "lavender", "xeriscape grasses", "coneflower", "sedum"],
            "climate_zone": "USDA 5b-6a",
        },
        "road_infrastructure": {
            "signage_style": "green Colorado highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/grey; some buried in newer suburban areas",
            "streetlights": "cobra-head LEDs on metal poles; decorative in downtown",
            "crosswalk_signs": "standard Colorado pedestrian crossing signs",
        },
        "parks": [
            {"name": "City Park", "lat": 39.7470, "lng": -104.9500, "area_ha": 130},
            {"name": "Washington Park", "lat": 39.7000, "lng": -104.9690, "area_ha": 66},
            {"name": "Cheesman Park", "lat": 39.7330, "lng": -104.9720, "area_ha": 32},
            {"name": "Sloan's Lake Park", "lat": 39.7480, "lng": -105.0490, "area_ha": 71},
            {"name": "Confluence Park", "lat": 39.7550, "lng": -105.0080, "area_ha": 4},
            {"name": "Civic Center Park", "lat": 39.7390, "lng": -104.9880, "area_ha": 5},
            {"name": "Ruby Hill Park", "lat": 39.6900, "lng": -105.0100, "area_ha": 30},
            {"name": "Berkeley Park", "lat": 39.7800, "lng": -105.0480, "area_ha": 12},
            {"name": "Observatory Park", "lat": 39.6760, "lng": -104.9550, "area_ha": 5},
            {"name": "Bear Creek Park", "lat": 39.6600, "lng": -105.0700, "area_ha": 150},
        ],
    },
    "Portland": {
        "province": "OR",
        "country": "USA",
        "lat": 45.52,
        "lng": -122.68,
        "timezone": "America/Los_Angeles",
        "climate_zone": "Oceanic (Cfb)",
        "architectural_styles": {
            "primary": ["Craftsman", "NW modern", "bungalow", "mid-century"],
            "secondary": ["Victorian", "Queen Anne", "Tudor Revival", "Contemporary"],
        },
        "building_materials": {
            "common": ["wood (cedar, fir)", "stucco", "brick", "concrete", "glass", "stone"],
            "roofing": ["asphalt shingle", "cedar shake", "flat membrane", "green roof"],
            "sidewalk": ["concrete", "pavers", "asphalt"],
        },
        "vegetation": {
            "tree_types": ["Douglas fir", "western red cedar", "bigleaf maple", "Japanese maple", "flowering cherry", "sequoia"],
            "common_plants": ["rhododendron", "azalea", "fern", "salal", "hosta", "hydrangea", "bamboo"],
            "climate_zone": "USDA 8a-8b",
        },
        "road_infrastructure": {
            "signage_style": "green Oregon highway signs; white-on-green street signs",
            "utility_poles": "wooden, brown/dark grey; some buried utilities in newer areas",
            "streetlights": "cobra-head LEDs on metal poles; decorative in historic districts",
            "crosswalk_signs": "standard Oregon pedestrian crossing signs; distinctive Portland zebra crosswalks",
        },
        "parks": [
            {"name": "Forest Park", "lat": 45.5600, "lng": -122.7600, "area_ha": 2100},
            {"name": "Washington Park", "lat": 45.5200, "lng": -122.7100, "area_ha": 100},
            {"name": "Laurelhurst Park", "lat": 45.5250, "lng": -122.6260, "area_ha": 12},
            {"name": "Mount Tabor Park", "lat": 45.5130, "lng": -122.5930, "area_ha": 80},
            {"name": "Tom McCall Waterfront Park", "lat": 45.5200, "lng": -122.6700, "area_ha": 15},
            {"name": "Cathedral Park", "lat": 45.5890, "lng": -122.7570, "area_ha": 3},
            {"name": "Powell Butte Nature Park", "lat": 45.4880, "lng": -122.5030, "area_ha": 240},
            {"name": "Peninsula Park", "lat": 45.5660, "lng": -122.6730, "area_ha": 5},
            {"name": "Alberta Park", "lat": 45.5590, "lng": -122.6500, "area_ha": 4},
            {"name": "Sellwood Park", "lat": 45.4660, "lng": -122.6520, "area_ha": 6},
        ],
    },
}


# ── helper data structures ───────────────────────────────────────────────────

# Inverted index: style -> list of city names
_STYLE_TO_CITIES: Dict[str, List[str]] = {}
for city_name, info in CITY_DATA.items():
    for style_list in info["architectural_styles"].values():
        for style in style_list:
            _STYLE_TO_CITIES.setdefault(style.lower(), []).append(city_name)

# Inverted index: material -> list of city names
_MATERIAL_TO_CITIES: Dict[str, List[str]] = {}
for city_name, info in CITY_DATA.items():
    for mat_list in info["building_materials"].values():
        for mat in mat_list:
            _MATERIAL_TO_CITIES.setdefault(mat.lower(), []).append(city_name)

# Inverted index: tree type -> list of city names
_TREE_TO_CITIES: Dict[str, List[str]] = {}
for city_name, info in CITY_DATA.items():
    for tree in info["vegetation"]["tree_types"]:
        _TREE_TO_CITIES.setdefault(tree.lower(), []).append(city_name)


# ── main class ──────────────────────────────────────────────────────────────

class GeolocationDatabase:
    """A comprehensive database of North American cities for visual geolocation."""

    # ── query methods ────────────────────────────────────────────────────────

    @staticmethod
    def city_names() -> List[str]:
        """Return sorted list of all city names in the database."""
        return sorted(CITY_DATA.keys())

    @staticmethod
    def get_city_data(city_name: str) -> Optional[dict]:
        """Return the full data dict for *city_name*, or None if not found."""
        return CITY_DATA.get(city_name)

    # ── city style profile ───────────────────────────────────────────────────

    @staticmethod
    def get_city_style_profile(city_name: str) -> Optional[dict]:
        """Return the architectural / visual profile for a city.

        Returns a dictionary with coordinates, architecture, materials,
        vegetation, and road infrastructure, or *None* if the city is not
        in the database.
        """
        city = CITY_DATA.get(city_name)
        if city is None:
            return None
        return {
            "city": city_name,
            "province": city["province"],
            "country": city["country"],
            "coordinates": (city["lat"], city["lng"]),
            "timezone": city["timezone"],
            "climate_zone": city["climate_zone"],
            "architectural_styles": city["architectural_styles"],
            "building_materials": city["building_materials"],
            "vegetation": city["vegetation"],
            "road_infrastructure": city["road_infrastructure"],
        }

    # ── find matching cities ─────────────────────────────────────────────────

    @staticmethod
    def find_matching_cities(features: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """Rank cities by how well they match a set of observed visual features.

        Parameters
        ----------
        features : dict
            Keys can include ``'styles'``, ``'materials'``, ``'trees'``.
            Each value is a list of strings (matched case-insensitively).

        Returns
        -------
        list of (city_name, score)
            Descending by score (0 to 1).  Score is the fraction of all provided
            feature strings that have at least one match in that city's data.
        """
        all_feature_terms: List[str] = []
        for key in ("styles", "materials", "trees"):
            if key in features:
                all_feature_terms.extend(features[key])

        if not all_feature_terms:
            return []

        def _score(city_name: str) -> float:
            info = CITY_DATA[city_name]
            match_count = 0
            for term in all_feature_terms:
                term_lower = term.lower()
                found = False
                # check styles
                for style_list in info["architectural_styles"].values():
                    for s in style_list:
                        if term_lower == s.lower() or term_lower in s.lower():
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # check materials
                    for mat_list in info["building_materials"].values():
                        for m in mat_list:
                            if term_lower == m.lower() or term_lower in m.lower():
                                found = True
                                break
                        if found:
                            break
                if not found:
                    # check trees
                    for t in info["vegetation"]["tree_types"]:
                        if term_lower == t.lower() or term_lower in t.lower():
                            found = True
                            break
                if found:
                    match_count += 1
            return match_count / len(all_feature_terms) if all_feature_terms else 0.0

        scores = [(name, _score(name)) for name in CITY_DATA]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ── find nearby parks radius ─────────────────────────────────────────────

    @staticmethod
    def find_nearby_parks(lat: float, lng: float, radius_km: float = 1.0) -> List[dict]:
        """Find all parks within *radius_km* of a given coordinate.

        Returns a list of dicts with keys ``name``, ``city``, ``lat``, ``lng``,
        ``area_ha``, ``distance_km``, sorted by distance ascending.
        """
        results: List[dict] = []
        for city_name, info in CITY_DATA.items():
            for park in info["parks"]:
                d = _haversine_km(lat, lng, park["lat"], park["lng"])
                if d <= radius_km:
                    results.append({
                        "name": park["name"],
                        "city": city_name,
                        "lat": park["lat"],
                        "lng": park["lng"],
                        "area_ha": park["area_ha"],
                        "distance_km": round(d, 3),
                    })
        results.sort(key=lambda x: x["distance_km"])
        return results

    # ── find all parks within walk time ──────────────────────────────────────

    @staticmethod
    def get_all_parks_in_radius(lat: float, lng: float, max_walk_minutes: float = 15.0) -> List[dict]:
        """Find parks reachable within *max_walk_minutes* minutes of walking.

        Uses a default walking pace of 5 km/h (about 3.1 mph).

        Returns a list sorted by distance (closest first), each entry includes
        the park details plus an estimated ``walk_minutes`` field.
        """
        pace_kmh = 5.0
        radius_km = (max_walk_minutes / 60.0) * pace_kmh
        results = GeolocationDatabase.find_nearby_parks(lat, lng, radius_km)
        for r in results:
            r["walk_minutes"] = round(_walk_time_minutes(r["distance_km"], pace_kmh), 1)
        results.sort(key=lambda x: x["walk_minutes"])
        return results

    # ── query by style / material / vegetation ───────────────────────────────

    @staticmethod
    def find_cities_by_style(style: str) -> List[str]:
        """Return list of city names that have a given architectural style."""
        return sorted(set(_STYLE_TO_CITIES.get(style.lower(), [])))

    @staticmethod
    def find_cities_by_material(material: str) -> List[str]:
        """Return list of city names that use a given building material."""
        return sorted(set(_MATERIAL_TO_CITIES.get(material.lower(), [])))

    @staticmethod
    def find_cities_by_tree(tree_type: str) -> List[str]:
        """Return list of city names that have a given tree type."""
        return sorted(set(_TREE_TO_CITIES.get(tree_type.lower(), [])))

    @staticmethod
    def find_cities_by_climate(climate_zone: str) -> List[str]:
        """Return list of city names matching a climate zone (case-insensitive partial match)."""
        cz = climate_zone.lower()
        return sorted(
            name for name, info in CITY_DATA.items()
            if cz in info["climate_zone"].lower()
        )

    # ── regional queries ─────────────────────────────────────────────────────

    @staticmethod
    def find_cities_in_region(lat_min: float, lat_max: float,
                              lng_min: float, lng_max: float) -> List[dict]:
        """Return all cities whose centre falls within the bounding box."""
        results = []
        for name, info in CITY_DATA.items():
            if lat_min <= info["lat"] <= lat_max and lng_min <= info["lng"] <= lng_max:
                results.append({"name": name, **info})
        return results

    @staticmethod
    def find_cities_nearby(lat: float, lng: float, radius_km: float = 50.0) -> List[dict]:
        """Return cities whose centre is within *radius_km* of the given point."""
        results = []
        for name, info in CITY_DATA.items():
            d = _haversine_km(lat, lng, info["lat"], info["lng"])
            if d <= radius_km:
                results.append({"name": name, **info, "distance_km": round(d, 2)})
        results.sort(key=lambda x: x["distance_km"])
        return results

    # ── summary statistics ───────────────────────────────────────────────────

    @staticmethod
    def summary() -> dict:
        """Return a high-level summary of the database contents."""
        total_cities = len(CITY_DATA)
        total_parks = sum(len(info["parks"]) for info in CITY_DATA.values())
        all_styles = set()
        all_materials = set()
        all_trees = set()
        for info in CITY_DATA.values():
            for sl in info["architectural_styles"].values():
                all_styles.update(sl)
            for ml in info["building_materials"].values():
                all_materials.update(ml)
            all_trees.update(info["vegetation"]["tree_types"])
        return {
            "total_cities": total_cities,
            "total_parks": total_parks,
            "unique_architectural_styles": len(all_styles),
            "unique_building_materials": len(all_materials),
            "unique_tree_types": len(all_trees),
            "cities": sorted(CITY_DATA.keys()),
        }


# ── convenience aliases ──────────────────────────────────────────────────────

GeolocationDB = GeolocationDatabase  # short alias


# ── simple CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== GeoVision Geolocation Database ===\n")
    print(f"Cities: {len(CITY_DATA)}")
    print(f"Parks total: {sum(len(v['parks']) for v in CITY_DATA.values())}\n")

    # summary
    s = GeolocationDatabase.summary()
    print("Summary:")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print()

    # style profile for Toronto
    profile = GeolocationDatabase.get_city_style_profile("Toronto")
    print("Toronto style profile:")
    print(json.dumps(profile, indent=2)[:500])
    print()

    # find matching cities
    features = {
        "styles": ["Victorian", "brick"],
        "materials": ["red brick"],
        "trees": ["maple"],
    }
    matches = GeolocationDatabase.find_matching_cities(features)
    print("Top 5 cities matching 'Victorian, brick, red brick, maple':")
    for name, score in matches[:5]:
        print(f"  {name}: {score:.2f}")
    print()

    # nearby parks test (Toronto city centre)
    parks = GeolocationDatabase.find_nearby_parks(43.65, -79.38, radius_km=2.0)
    print(f"Parks within 2 km of Toronto centre ({len(parks)} found):")
    for p in parks[:5]:
        print(f"  {p['name']} ({p['city']}) - {p['distance_km']:.3f} km")
    print()

    # walk time parks
    walk_parks = GeolocationDatabase.get_all_parks_in_radius(43.65, -79.38, max_walk_minutes=20)
    print(f"Parks within 20 min walk of Toronto centre ({len(walk_parks)} found):")
    for p in walk_parks[:5]:
        print(f"  {p['name']} - {p['walk_minutes']:.1f} min walk")
    print()

    # query by style
    print("Cities with 'Craftsman' style:")
    print("  ", ", ".join(GeolocationDatabase.find_cities_by_style("Craftsman")))
    print()

    print("Cities with 'stucco' material:")
    print("  ", ", ".join(GeolocationDatabase.find_cities_by_material("stucco")))
    print()

    print("Cities with 'eucalyptus' trees:")
    print("  ", ", ".join(GeolocationDatabase.find_cities_by_tree("eucalyptus")))
