#!/usr/bin/env python3
"""Continuous weather feed script for real-time data updates."""
import requests, time, json
from pathlib import Path

def fetch_continuous_weather(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": "2024-01-01", "end_date": "2024-12-31", "daily": "temperature_2m_max,precipitation_sum", "timezone": "auto"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

if __name__ == "__main__":
    # Example continuous feed for major cities
    cities = [(40.7128, -74.0060), (34.0522, -118.2437), (51.5074, -0.1278)]
    results = {}
    for lat, lon in cities:
        data = fetch_continuous_weather(lat, lon)
        results[f"{lat},{lon}"] = data.get("daily", {}).get("temperature_2m_max", [None])[0] if data else None
    out = Path(__file__).parent.parent / "data" / "continuous_weather.json"
    out.write_text(json.dumps({"last_update": time.time(), "feeds": results}, indent=2))
    print("Continuous weather feed updated:", out)
