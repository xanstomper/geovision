import unicodedata
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Detects language/script, TLDs, and currency from OCR text to estimate geographic regions."""
    
    def __init__(self):
        self.tld_mapping = {
            r'\.co\.uk': {'region': 'United Kingdom', 'country_code': 'GB', 'lat': 55.3781, 'lon': -3.4360},
            r'\.uk': {'region': 'United Kingdom', 'country_code': 'GB', 'lat': 55.3781, 'lon': -3.4360},
            r'\.com\.au': {'region': 'Australia', 'country_code': 'AU', 'lat': -25.2744, 'lon': 133.7751},
            r'\.au': {'region': 'Australia', 'country_code': 'AU', 'lat': -25.2744, 'lon': 133.7751},
            r'\.ca': {'region': 'Canada', 'country_code': 'CA', 'lat': 56.1304, 'lon': -106.3468},
            r'\.de': {'region': 'Germany', 'country_code': 'DE', 'lat': 51.1657, 'lon': 10.4515},
            r'\.fr': {'region': 'France', 'country_code': 'FR', 'lat': 46.2276, 'lon': 2.2137},
            r'\.jp': {'region': 'Japan', 'country_code': 'JP', 'lat': 36.2048, 'lon': 138.2529},
            r'\.ru': {'region': 'Russia', 'country_code': 'RU', 'lat': 61.5240, 'lon': 105.3188},
            r'\.br': {'region': 'Brazil', 'country_code': 'BR', 'lat': -14.2350, 'lon': -51.9253},
            r'\.in': {'region': 'India', 'country_code': 'IN', 'lat': 20.5937, 'lon': 78.9629},
            r'\.pl': {'region': 'Poland', 'country_code': 'PL', 'lat': 51.9194, 'lon': 19.1451},
            r'\.nl': {'region': 'Netherlands', 'country_code': 'NL', 'lat': 52.1326, 'lon': 5.2913},
            r'\.es': {'region': 'Spain', 'country_code': 'ES', 'lat': 40.4637, 'lon': -3.7492},
            r'\.it': {'region': 'Italy', 'country_code': 'IT', 'lat': 41.8719, 'lon': 12.5674},
            r'\.se': {'region': 'Sweden', 'country_code': 'SE', 'lat': 60.1282, 'lon': 18.6435},
            r'\.no': {'region': 'Norway', 'country_code': 'NO', 'lat': 60.4720, 'lon': 8.4689},
            r'\.dk': {'region': 'Denmark', 'country_code': 'DK', 'lat': 56.2639, 'lon': 9.5018},
            r'\.fi': {'region': 'Finland', 'country_code': 'FI', 'lat': 61.9241, 'lon': 25.7482},
            r'\.ch': {'region': 'Switzerland', 'country_code': 'CH', 'lat': 46.8182, 'lon': 8.2275},
            r'\.at': {'region': 'Austria', 'country_code': 'AT', 'lat': 47.5162, 'lon': 14.5501},
            r'\.cz': {'region': 'Czech Republic', 'country_code': 'CZ', 'lat': 49.8175, 'lon': 15.4730},
            r'\.pt': {'region': 'Portugal', 'country_code': 'PT', 'lat': 39.3999, 'lon': -8.2245},
            r'\.ie': {'region': 'Ireland', 'country_code': 'IE', 'lat': 53.1424, 'lon': -7.6921},
            r'\.gr': {'region': 'Greece', 'country_code': 'GR', 'lat': 39.0742, 'lon': 21.8243},
            r'\.tr': {'region': 'Turkey', 'country_code': 'TR', 'lat': 38.9637, 'lon': 35.2433},
            r'\.za': {'region': 'South Africa', 'country_code': 'ZA', 'lat': -30.5595, 'lon': 22.9375},
            r'\.mx': {'region': 'Mexico', 'country_code': 'MX', 'lat': 23.6345, 'lon': -102.5528},
            r'\.nz': {'region': 'New Zealand', 'country_code': 'NZ', 'lat': -40.9006, 'lon': 174.8860},
            r'\.kr': {'region': 'South Korea', 'country_code': 'KR', 'lat': 35.9078, 'lon': 127.7669},
            r'\.tw': {'region': 'Taiwan', 'country_code': 'TW', 'lat': 23.6978, 'lon': 120.9605},
            r'\.th': {'region': 'Thailand', 'country_code': 'TH', 'lat': 15.8700, 'lon': 100.9925},
            r'\.vn': {'region': 'Vietnam', 'country_code': 'VN', 'lat': 14.0583, 'lon': 108.2772},
            r'\.id': {'region': 'Indonesia', 'country_code': 'ID', 'lat': -0.7893, 'lon': 113.9213},
            r'\.ua': {'region': 'Ukraine', 'country_code': 'UA', 'lat': 48.3794, 'lon': 31.1656},
            r'\.ro': {'region': 'Romania', 'country_code': 'RO', 'lat': 45.9432, 'lon': 24.9668},
            r'\.hu': {'region': 'Hungary', 'country_code': 'HU', 'lat': 47.1625, 'lon': 19.5033},
            r'\.be': {'region': 'Belgium', 'country_code': 'BE', 'lat': 50.5039, 'lon': 4.4699},
        }

        # International Phone Dialing Codes
        self.phone_mapping = {
            r'(?:\+44|0044)[\s\d\-]{8,}': {'region': 'United Kingdom', 'country_code': 'GB', 'lat': 55.3781, 'lon': -3.4360},
            r'(?:\+33|0033)[\s\d\-]{8,}': {'region': 'France', 'country_code': 'FR', 'lat': 46.2276, 'lon': 2.2137},
            r'(?:\+49|0049)[\s\d\-]{8,}': {'region': 'Germany', 'country_code': 'DE', 'lat': 51.1657, 'lon': 10.4515},
            r'(?:\+81|0081)[\s\d\-]{8,}': {'region': 'Japan', 'country_code': 'JP', 'lat': 36.2048, 'lon': 138.2529},
            r'(?:\+34|0034)[\s\d\-]{8,}': {'region': 'Spain', 'country_code': 'ES', 'lat': 40.4637, 'lon': -3.7492},
            r'(?:\+39|0039)[\s\d\-]{8,}': {'region': 'Italy', 'country_code': 'IT', 'lat': 41.8719, 'lon': 12.5674},
            r'(?:\+48|0048)[\s\d\-]{8,}': {'region': 'Poland', 'country_code': 'PL', 'lat': 51.9194, 'lon': 19.1451},
            r'(?:\+61|0061)[\s\d\-]{8,}': {'region': 'Australia', 'country_code': 'AU', 'lat': -25.2744, 'lon': 133.7751},
            r'(?:\+55|0055)[\s\d\-]{8,}': {'region': 'Brazil', 'country_code': 'BR', 'lat': -14.2350, 'lon': -51.9253},
            r'(?:\+82|0082)[\s\d\-]{8,}': {'region': 'South Korea', 'country_code': 'KR', 'lat': 35.9078, 'lon': 127.7669},
            r'(?:\+31|0031)[\s\d\-]{8,}': {'region': 'Netherlands', 'country_code': 'NL', 'lat': 52.1326, 'lon': 5.2913},
            r'(?:\+46|0046)[\s\d\-]{8,}': {'region': 'Sweden', 'country_code': 'SE', 'lat': 60.1282, 'lon': 18.6435},
            r'(?:\+47|0047)[\s\d\-]{8,}': {'region': 'Norway', 'country_code': 'NO', 'lat': 60.4720, 'lon': 8.4689},
            r'(?:\+27|0027)[\s\d\-]{8,}': {'region': 'South Africa', 'country_code': 'ZA', 'lat': -30.5595, 'lon': 22.9375},
            r'(?:\+52|0052)[\s\d\-]{8,}': {'region': 'Mexico', 'country_code': 'MX', 'lat': 23.6345, 'lon': -102.5528},
            r'(?:\+90|0090)[\s\d\-]{8,}': {'region': 'Turkey', 'country_code': 'TR', 'lat': 38.9637, 'lon': 35.2433},
            r'(?:\+30|0030)[\s\d\-]{8,}': {'region': 'Greece', 'country_code': 'GR', 'lat': 39.0742, 'lon': 21.8243},
            r'(?:\+351|00351)[\s\d\-]{8,}': {'region': 'Portugal', 'country_code': 'PT', 'lat': 39.3999, 'lon': -8.2245},
            r'(?:\+41|0041)[\s\d\-]{8,}': {'region': 'Switzerland', 'country_code': 'CH', 'lat': 46.8182, 'lon': 8.2275},
            r'(?:\+43|0043)[\s\d\-]{8,}': {'region': 'Austria', 'country_code': 'AT', 'lat': 47.5162, 'lon': 14.5501},
            r'(?:\+420|00420)[\s\d\-]{8,}': {'region': 'Czech Republic', 'country_code': 'CZ', 'lat': 49.8175, 'lon': 15.4730},
            r'(?:\+64|0064)[\s\d\-]{8,}': {'region': 'New Zealand', 'country_code': 'NZ', 'lat': -40.9006, 'lon': 174.8860},
            r'(?:\+353|00353)[\s\d\-]{8,}': {'region': 'Ireland', 'country_code': 'IE', 'lat': 53.1424, 'lon': -7.6921},
        }
        
        self.currency_mapping = {
            '$': [
                {'region': 'United States', 'lat': 37.0902, 'lon': -95.7129},
                {'region': 'Canada', 'lat': 56.1304, 'lon': -106.3468},
                {'region': 'Australia', 'lat': -25.2744, 'lon': 133.7751}
            ],
            '€': [
                {'region': 'France', 'lat': 46.2276, 'lon': 2.2137},
                {'region': 'Germany', 'lat': 51.1657, 'lon': 10.4515},
                {'region': 'Spain', 'lat': 40.4637, 'lon': -3.7492},
                {'region': 'Italy', 'lat': 41.8719, 'lon': 12.5674}
            ],
            '£': [
                {'region': 'United Kingdom', 'lat': 55.3781, 'lon': -3.4360}
            ],
            '¥': [
                {'region': 'Japan', 'lat': 36.2048, 'lon': 138.2529},
                {'region': 'China', 'lat': 35.8617, 'lon': 104.1954}
            ],
            '₹': [
                {'region': 'India', 'lat': 20.5937, 'lon': 78.9629}
            ]
        }

        self.script_mapping = {
            'LATIN': [
                {'region': 'United States', 'lat': 37.0902, 'lon': -95.7129},
                {'region': 'United Kingdom', 'lat': 55.3781, 'lon': -3.4360},
                {'region': 'France', 'lat': 46.2276, 'lon': 2.2137},
                {'region': 'Germany', 'lat': 51.1657, 'lon': 10.4515},
                {'region': 'Spain', 'lat': 40.4637, 'lon': -3.7492},
                {'region': 'Canada', 'lat': 56.1304, 'lon': -106.3468},
                {'region': 'Australia', 'lat': -25.2744, 'lon': 133.7751}
            ],
            'CYRILLIC': [
                {'region': 'Russia', 'lat': 61.5240, 'lon': 105.3188},
                {'region': 'Ukraine', 'lat': 48.3794, 'lon': 31.1656},
                {'region': 'Belarus', 'lat': 53.7098, 'lon': 27.9534}
            ],
            'ARABIC': [
                {'region': 'Saudi Arabia', 'lat': 23.8859, 'lon': 45.0792},
                {'region': 'Egypt', 'lat': 26.8206, 'lon': 30.8025},
                {'region': 'UAE', 'lat': 23.4241, 'lon': 53.8478}
            ],
            'CJK': [
                {'region': 'China', 'lat': 35.8617, 'lon': 104.1954},
                {'region': 'Japan', 'lat': 36.2048, 'lon': 138.2529},
                {'region': 'South Korea', 'lat': 35.9078, 'lon': 127.7669}
            ],
            'DEVANAGARI': [
                {'region': 'India', 'lat': 20.5937, 'lon': 78.9629},
                {'region': 'Nepal', 'lat': 28.3949, 'lon': 84.1240}
            ],
            'THAI': [
                {'region': 'Thailand', 'lat': 15.8700, 'lon': 100.9925}
            ]
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
                    "country_code": data.get('country_code'),
                    "latitude": data['lat'],
                    "longitude": data['lon'],
                    "confidence": 0.78  # High confidence ccTLD
                })

        # Detect Phone Calling Codes
        for phone_pat, data in self.phone_mapping.items():
            match = re.search(phone_pat, ocr_text)
            if match:
                results.append({
                    "type": "phone_code",
                    "detected": match.group(0),
                    "region": data['region'],
                    "country_code": data.get('country_code'),
                    "latitude": data['lat'],
                    "longitude": data['lon'],
                    "confidence": 0.85  # International phone prefix is hard proof
                })
                
        # Detect Currencies
        for curr, candidates in self.currency_mapping.items():
            if curr in ocr_text:
                for data in candidates:
                    results.append({
                        "type": "currency",
                        "detected": curr,
                        "region": data['region'],
                        "latitude": data['lat'],
                        "longitude": data['lon'],
                        "confidence": 0.05  # Lower confidence so generic country centers don't win
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
                    candidates = self.script_mapping[script]
                    # Calculate a base confidence between 0.01 and 0.05 based on percentage
                    base_conf = 0.01 + (0.04 * (count / total_chars))
                    for data in candidates:
                        results.append({
                            "type": "script",
                            "detected": script,
                            "region": data['region'],
                            "latitude": data['lat'],
                            "longitude": data['lon'],
                            "confidence": base_conf
                        })
                    
        return results
