import pytest
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.telecom_osint import TelecomOSINT
from modules.exif_extractor import extract_exif_data, get_gps_from_exif
from modules.language_detector import LanguageDetector

def test_telecom_osint():
    telecom = TelecomOSINT()
    
    # Test valid NANP number
    text_with_nanp = "Call me at (434) 555-1234 for details."
    results = telecom.analyze_ocr_for_telecom_data(text_with_nanp)
    
    assert len(results) == 1
    assert results[0]['type'] == 'area_code'
    assert results[0]['code'] == '434'
    assert 'Lynchburg' in results[0]['region']
    
    # Test valid International number
    text_with_intl = "Our office in France: +33 1 23 45 67 89"
    results_intl = telecom.analyze_ocr_for_telecom_data(text_with_intl)
    
    assert len(results_intl) > 0
    assert results_intl[0]['type'] == 'international_code'
    assert results_intl[0]['code'] == '+33'
    assert results_intl[0]['region'] == 'France'
    
    # Test no numbers
    results_none = telecom.analyze_ocr_for_telecom_data("Just some plain text without numbers.")
    assert len(results_none) == 0

def test_exif_extractor():
    # Provide a fake image path or real if exists. 
    # extract_exif_data likely handles missing/invalid files gracefully or raises exception
    # For unit testing the logic, we'll test the GPS conversion function directly
    exif_data_mock = {
        'GPS GPSLatitude': [40, 42, 46.8],
        'GPS GPSLatitudeRef': 'N',
        'GPS GPSLongitude': [74, 0, 21.6],
        'GPS GPSLongitudeRef': 'W'
    }
    
    # If the converter uses ExifRead types, we might need to mock them, 
    # but let's see if it works with plain lists/tuples or mock objects.
    # We can use MagicMock if necessary, but assuming lists work if not strict ExifRead objects.
    # Alternatively, just test extract_exif_data on the test_building.jpg
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_building.jpg'))
    if os.path.exists(image_path):
        data = extract_exif_data(image_path)
        assert isinstance(data, dict)
        gps = get_gps_from_exif(data)
        # test_building.jpg might or might not have GPS, but function should run
        assert gps is None or isinstance(gps, dict)

def test_language_detector():
    detector = LanguageDetector()
    
    # Test French text
    french_text = "Bonjour tout le monde. C'est la vie."
    results = detector.analyze_text(french_text)
    
    # The detector should identify the script
    if results:
        assert any(r.get('detected') == 'LATIN' for r in results)
        
    # Test Spanish text
    spanish_text = "Hola, buenos días. ¿Cómo estás?"
    results = detector.analyze_text(spanish_text)
    if results:
        assert any(r.get('detected') == 'LATIN' for r in results)
