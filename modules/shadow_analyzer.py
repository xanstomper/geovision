"""
Shadow and Solar Analyzer Module
Estimates latitude bands based on sun elevation and shadow angles if time is known.
"""
import math
from datetime import datetime
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ShadowAnalyzer:
    def __init__(self):
        pass

    def estimate_latitude_from_sun(self, exif_data: Dict[str, Any], shadow_angle_deg: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        If EXIF has datetime, we can estimate possible latitudes based on sun position.
        Without knowing exact longitude, we give a probable hemisphere or latitude band.
        """
        if not exif_data:
            return None

        # Look for DateTimeOriginal (supports both PIL TAGS and ExifRead key formats)
        date_str = exif_data.get("DateTimeOriginal") or exif_data.get("EXIF DateTimeOriginal") or exif_data.get("DateTime")
        if not date_str:
            return None

        try:
            # EXIF format: YYYY:MM:DD HH:MM:SS
            date_str = str(date_str).strip()
            dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            day_of_year = dt.timetuple().tm_yday
            
            # Solar declination angle approximation
            declination = -23.44 * math.cos(math.radians((360.0 / 365.24) * (day_of_year + 10)))
            
            # Without exact longitude, we can just infer Northern vs Southern hemisphere likelihood 
            # if the VLM tells us if it's winter/summer looking. But purely mathematically:
            # We just return the declination for the VLM to use as a strong prior.
            
            logger.info(f"  Sun declination calculated as {declination:.2f} degrees for {dt.date()}")
            result = {
                "solar_declination": declination,
                "day_of_year": day_of_year,
                "estimated_time": f"{dt.hour}:{dt.minute}"
            }
            if shadow_angle_deg is not None:
                result["estimated_latitude"] = shadow_angle_deg + declination
            return result
        except Exception as e:
            logger.warning(f"Shadow Analyzer failed to parse EXIF date: {e}")
            return None

