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
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "telecom_data.json")
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                self.nanp_area_codes = data.get("nanp_area_codes", {})
                self.intl_codes = data.get("intl_codes", {})
        except Exception as e:
            logger.error(f"Failed to load telecom data: {e}")
            self.nanp_area_codes = {}
            self.intl_codes = {}

    def analyze_ocr_for_telecom_data(self, ocr_text: str) -> List[Dict[str, Any]]:
        """
        Parses OCR text for phone number patterns and maps them to geographic regions.
        """
        if not ocr_text:
            return []

        findings = []
        
        # Look for international codes (+XX)
        for code, data in self.intl_codes.items():
            # Fix regex to use word boundaries so +33 doesn't match inside +330
            # Note: since code starts with '+', we need to escape it and use \b at the end
            pattern = re.escape(code) + r"\b"
            if re.search(pattern, ocr_text):
                logger.info(f"  [+] Found International Code {code} in OCR text -> {data['country']}")
                findings.append({
                    "type": "international_code",
                    "code": code,
                    "region": data["country"],
                    "latitude": data["lat"],
                    "longitude": data["lon"],
                    "confidence": 0.85,
                    "evidence": "international_code",
                    "sources": ["telecom_db"]
                })

        # Look for NANP (North American) phone numbers
        # Pattern matches: (123) 456-7890, 123-456-7890, 123.456.7890
        nanp_pattern = r'\b\(?([2-9][0-9]{2})\)?[-.\s]?([2-9][0-9]{2})[-.\s]?([0-9]{4})\b'
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
                    "confidence": 0.95,
                    "evidence": "area_code",
                    "sources": ["nanp_db"]
                })
            else:
                logger.info(f"  [?] Found unknown Area Code {area_code} in OCR text")

        return findings
