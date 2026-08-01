import re
import json
import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

class TelecomOSINT:
    """
    Advanced Telecom OSINT module.
    Extracts phone numbers from raw OCR text and cross-references Area Codes and Country Codes
    against a telecom database to instantly pinpoint geographic regions (cities/states).
    """
    def __init__(self):
        # A mapping of common North American Area Codes to their regions.
        # This is a highly truncated list for demonstration, but covers major metropolitan areas.
        self.nanp_area_codes = {
            "212": {"city": "New York City", "state": "NY", "country": "US", "lat": 40.7128, "lon": -74.0060},
            "310": {"city": "Los Angeles", "state": "CA", "country": "US", "lat": 34.0522, "lon": -118.2437},
            "312": {"city": "Chicago", "state": "IL", "country": "US", "lat": 41.8781, "lon": -87.6298},
            "415": {"city": "San Francisco", "state": "CA", "country": "US", "lat": 37.7749, "lon": -122.4194},
            "206": {"city": "Seattle", "state": "WA", "country": "US", "lat": 47.6062, "lon": -122.3321},
            "305": {"city": "Miami", "state": "FL", "country": "US", "lat": 25.7617, "lon": -80.1918},
            "404": {"city": "Atlanta", "state": "GA", "country": "US", "lat": 33.7490, "lon": -84.3880},
            "617": {"city": "Boston", "state": "MA", "country": "US", "lat": 42.3601, "lon": -71.0589},
            "713": {"city": "Houston", "state": "TX", "country": "US", "lat": 29.7604, "lon": -95.3698},
            "416": {"city": "Toronto", "state": "ON", "country": "CA", "lat": 43.6510, "lon": -79.3470},
            "514": {"city": "Montreal", "state": "QC", "country": "CA", "lat": 45.5017, "lon": -73.5673},
            "604": {"city": "Vancouver", "state": "BC", "country": "CA", "lat": 49.2827, "lon": -123.1207},
            "403": {"city": "Calgary", "state": "AB", "country": "CA", "lat": 51.0447, "lon": -114.0719},
            "902": {"city": "Halifax", "state": "NS", "country": "CA", "lat": 44.6488, "lon": -63.5752},
            "804": {"city": "Richmond", "state": "VA", "country": "US", "lat": 37.5407, "lon": -77.4360},
            "434": {"city": "Lynchburg/Charlottesville", "state": "VA", "country": "US", "lat": 37.4138, "lon": -79.1422},
            "276": {"city": "Southwest Virginia", "state": "VA", "country": "US", "lat": 36.6000, "lon": -81.0000},
            "703": {"city": "Northern Virginia", "state": "VA", "country": "US", "lat": 38.8400, "lon": -77.1000},
            "540": {"city": "Roanoke/Harrisonburg", "state": "VA", "country": "US", "lat": 37.2700, "lon": -79.9400},
            "757": {"city": "Hampton Roads", "state": "VA", "country": "US", "lat": 36.8500, "lon": -76.2800}
        }
        
        self.intl_codes = {
            "+44": {"country": "United Kingdom", "lat": 55.3781, "lon": -3.4360},
            "+61": {"country": "Australia", "lat": -25.2744, "lon": 133.7751},
            "+33": {"country": "France", "lat": 46.2276, "lon": 2.2137},
            "+49": {"country": "Germany", "lat": 51.1657, "lon": 10.4515},
            "+81": {"country": "Japan", "lat": 36.2048, "lon": 138.2529}
        }

    def analyze_ocr_for_telecom_data(self, ocr_text: str) -> List[Dict[str, Any]]:
        """
        Parses OCR text for phone number patterns and maps them to geographic regions.
        """
        if not ocr_text:
            return []

        findings = []
        
        # Look for international codes (+XX)
        for code, data in self.intl_codes.items():
            if code in ocr_text:
                logger.info(f"  [+] Found International Code {code} in OCR text -> {data['country']}")
                findings.append({
                    "type": "international_code",
                    "code": code,
                    "region": data["country"],
                    "latitude": data["lat"],
                    "longitude": data["lon"],
                    "confidence": 0.85
                })

        # Look for NANP (North American) phone numbers
        # Pattern matches: (123) 456-7890, 123-456-7890, 123.456.7890
        nanp_pattern = r'\(?([2-9][0-9]{2})\)?[-.\s]?([2-9][0-9]{2})[-.\s]?([0-9]{4})'
        matches = re.finditer(nanp_pattern, ocr_text)
        
        seen_area_codes = set()
        for match in matches:
            area_code = match.group(1)
            if area_code in seen_area_codes:
                continue
            
            seen_area_codes.add(area_code)
            if area_code in self.nanp_area_codes:
                data = self.nanp_area_codes[area_code]
                logger.info(f"  [+] Found Area Code {area_code} in OCR text -> {data['city']}, {data['state']}")
                findings.append({
                    "type": "area_code",
                    "code": area_code,
                    "full_number": match.group(0),
                    "region": f"{data['city']}, {data['state']}",
                    "latitude": data["lat"],
                    "longitude": data["lon"],
                    "confidence": 0.95  # Area codes on signs are EXTREMELY high confidence indicators
                })
            else:
                logger.info(f"  [?] Found unknown Area Code {area_code} in OCR text")

        return findings
