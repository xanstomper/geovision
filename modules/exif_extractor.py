from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def extract_exif_data(image_path: str) -> Dict[str, Any]:
    """
    GeoSpy Method: EXIF Metadata Extraction.
    Extracts all metadata including hidden GPS coordinates, timestamps, and camera models.
    """
    exif_data = {}
    try:
        image = Image.open(image_path)
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
                    exif_data[decoded] = gps_data
                else:
                    if isinstance(value, bytes):
                        try:
                            exif_data[decoded] = value.decode('utf-8', errors='ignore')
                        except:
                            pass
                    else:
                        exif_data[decoded] = str(value)
            logger.info(f"  ✓ Extracted EXIF data: {len(exif_data)} tags found")
        else:
            logger.info("  ⚠ No EXIF data found in image")
    except Exception as e:
        logger.warning(f"  Failed to extract EXIF data: {e}")
    return exif_data

def get_gps_from_exif(exif_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Converts EXIF GPSInfo into decimal latitude and longitude."""
    def _convert_to_degrees(value):
        d, m, s = value
        return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)

    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None

    try:
        gps_lat = gps_info.get("GPSLatitude")
        gps_lat_ref = gps_info.get("GPSLatitudeRef")
        gps_lon = gps_info.get("GPSLongitude")
        gps_lon_ref = gps_info.get("GPSLongitudeRef")

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = _convert_to_degrees(gps_lat)
            if gps_lat_ref != "N":
                lat = -lat

            lon = _convert_to_degrees(gps_lon)
            if gps_lon_ref != "E":
                lon = -lon

            return {"latitude": lat, "longitude": lon}
    except Exception as e:
        logger.warning(f"  Failed to parse GPS EXIF data: {e}")
    return None
