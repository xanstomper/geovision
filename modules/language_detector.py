import unicodedata
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Detects language/script, TLDs, and currency from OCR text to estimate geographic regions."""
    
    def __init__(self):
        self.tld_mapping = {
            r'\.co\.uk': {'region': 'United Kingdom', 'lat': 55.3781, 'lon': -3.4360},
            r'\.com\.au': {'region': 'Australia', 'lat': -25.2744, 'lon': 133.7751},
            r'\.ca': {'region': 'Canada', 'lat': 56.1304, 'lon': -106.3468},
            r'\.de': {'region': 'Germany', 'lat': 51.1657, 'lon': 10.4515},
            r'\.fr': {'region': 'France', 'lat': 46.2276, 'lon': 2.2137},
            r'\.jp': {'region': 'Japan', 'lat': 36.2048, 'lon': 138.2529},
            r'\.ru': {'region': 'Russia', 'lat': 61.5240, 'lon': 105.3188},
            r'\.br': {'region': 'Brazil', 'lat': -14.2350, 'lon': -51.9253},
            r'\.in': {'region': 'India', 'lat': 20.5937, 'lon': 78.9629}
        }
        
        self.currency_mapping = {
            '$': {'region': 'North America/Australia/etc', 'lat': 37.0902, 'lon': -95.7129},
            '€': {'region': 'Eurozone', 'lat': 51.1657, 'lon': 10.4515},
            '£': {'region': 'United Kingdom', 'lat': 55.3781, 'lon': -3.4360},
            '¥': {'region': 'Japan/China', 'lat': 36.2048, 'lon': 138.2529},
            '₹': {'region': 'India', 'lat': 20.5937, 'lon': 78.9629}
        }

        self.script_mapping = {
            'LATIN': {'region': 'Americas/Europe', 'lat': 48.0, 'lon': 14.0},
            'CYRILLIC': {'region': 'Russia/Eastern Europe', 'lat': 61.5240, 'lon': 105.3188},
            'ARABIC': {'region': 'Middle East/North Africa', 'lat': 23.8859, 'lon': 45.0792},
            'CJK': {'region': 'East Asia', 'lat': 35.8617, 'lon': 104.1954},
            'DEVANAGARI': {'region': 'India', 'lat': 20.5937, 'lon': 78.9629},
            'THAI': {'region': 'Thailand', 'lat': 15.8700, 'lon': 100.9925}
        }

    def analyze_text(self, ocr_text: str) -> List[Dict]:
        results = []
        if not ocr_text:
            return results
            
        # Detect TLDs
        for tld_pattern, data in self.tld_mapping.items():
            if re.search(tld_pattern + r'\b', ocr_text, re.IGNORECASE):
                results.append({
                    "type": "tld",
                    "detected": tld_pattern.replace('\\', ''),
                    "region": data['region'],
                    "latitude": data['lat'],
                    "longitude": data['lon'],
                    "confidence": 0.95
                })
                
        # Detect Currencies
        for curr, data in self.currency_mapping.items():
            if curr in ocr_text:
                results.append({
                    "type": "currency",
                    "detected": curr,
                    "region": data['region'],
                    "latitude": data['lat'],
                    "longitude": data['lon'],
                    "confidence": 0.70
                })
                
        # Detect Scripts
        script_counts = {}
        for char in ocr_text:
            if not char.strip() or unicodedata.category(char).startswith('P'):
                continue
            try:
                name = unicodedata.name(char)
                script = name.split()[0]
                if script in self.script_mapping:
                    script_counts[script] = script_counts.get(script, 0) + 1
                elif 'CJK' in name:
                    script_counts['CJK'] = script_counts.get('CJK', 0) + 1
            except ValueError:
                pass
                
        total_chars = sum(script_counts.values())
        if total_chars > 0:
            for script, count in script_counts.items():
                if count / total_chars > 0.1:  # At least 10% of text
                    results.append({
                        "type": "script",
                        "detected": script,
                        "region": self.script_mapping[script]['region'],
                        "latitude": self.script_mapping[script]['lat'],
                        "longitude": self.script_mapping[script]['lon'],
                        "confidence": 0.85 * (count / total_chars)
                    })
                    
        return results
