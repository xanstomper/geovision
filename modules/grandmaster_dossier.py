"""
GeoVision Grandmaster Intelligence Dossier Generator
======================================================
Synthesizes zero-storage geolocation evidence into a comprehensive,
forensic-grade investigative dossier matching and exceeding GeoSpy Raven.

Generates:
  1. Executive Geolocation Verdict (Lat, Lon, Address, Confidence, Precision Tier).
  2. Physical Visual Forensics Table (driving side, road marks, poles, plates, soil, sun).
  3. PlonkIt Grandmaster Infrastructure Deductions (85-country matching & falsification).
  4. CarID Vehicle Fleet Demographics (pickup ratio, Kei cars, mopeds, regional affinity).
  5. Topological Micro-GIS Street Targeter Intersections (exact street names & cross streets).
  6. High-density Markdown and self-contained interactive HTML dossiers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GrandmasterDossierGenerator:
    """Produces multi-format intelligence dossiers from zero-storage geolocation runs."""

    def generate_dossier(
        self,
        image_path: str,
        investigation_result: Dict[str, Any],
        plonkit_result: Optional[Dict[str, Any]] = None,
        car_fleet_result: Optional[Dict[str, Any]] = None,
        street_target_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes all modules into a unified intelligence dossier structure."""
        best = investigation_result.get("best_estimate", {})
        lat = best.get("latitude")
        lon = best.get("longitude")
        city = best.get("city", "Unknown City")
        country = best.get("country", "")
        conf = best.get("confidence", 0.0)
        precision = best.get("precision_tier", "regional")

        # Top street intersection if available
        top_intersection = None
        if street_target_result and street_target_result.get("top_match"):
            top_intersection = street_target_result["top_match"]

        # Top country from Plonkit if available
        top_plonkit = None
        if plonkit_result and plonkit_result.get("top_candidate"):
            top_plonkit = plonkit_result["top_candidate"]

        forensics = investigation_result.get("forensic_breakdown", {})
        elimination_matrix = investigation_result.get("elimination_matrix", [])

        dossier = {
            "title": "GeoVision Grandmaster Intelligence Dossier",
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "image_path": os.path.abspath(image_path),
            "verdict": {
                "latitude": lat,
                "longitude": lon,
                "city": city,
                "country": country,
                "confidence": conf,
                "precision_tier": precision,
                "top_street_intersection": top_intersection.get("intersection") if top_intersection else None,
                "google_maps_url": best.get("google_maps_url"),
                "osm_url": best.get("osm_url"),
            },
            "forensic_breakdown": forensics,
            "plonkit_infrastructure": plonkit_result,
            "car_fleet_demographics": car_fleet_result,
            "street_targeting": street_target_result,
            "elimination_matrix": elimination_matrix,
            "reasoning_chain": investigation_result.get("reasoning_chain", []),
        }

        return dossier

    def render_markdown(self, dossier: Dict[str, Any]) -> str:
        """Renders the dossier into GitHub-flavored Markdown."""
        v = dossier.get("verdict", {})
        forensics = dossier.get("forensic_breakdown", {})
        plonkit = dossier.get("plonkit_infrastructure", {})
        fleet = dossier.get("car_fleet_demographics", {})
        street = dossier.get("street_targeting", {})
        elim = dossier.get("elimination_matrix", [])

        lines = [
            f"# {dossier.get('title', 'GeoVision Dossier')}",
            f"**Generated:** {dossier.get('generation_time')} | **Image:** `{os.path.basename(dossier.get('image_path', ''))}`",
            "",
            "## 1. Executive Geolocation Verdict",
            f"- **Target Coordinates:** `{v.get('latitude')}, {v.get('longitude')}`",
            f"- **Settlement / Region:** **{v.get('city')}, {v.get('country')}**",
            f"- **Confidence:** **{v.get('confidence', 0.0) * 100:.1f}%** (Tier: `{v.get('precision_tier')}`)",
        ]

        if v.get("top_street_intersection"):
            lines.append(f"- **Pinpoint Street Intersection:** **{v.get('top_street_intersection')}**")

        lines.extend([
            f"- [View on Google Maps]({v.get('google_maps_url')}) | [View on OpenStreetMap]({v.get('osm_url')})",
            "",
            "## 2. Physical Visual Forensics",
            "| Forensic Feature | Observed Value | Confidence | Regional Implication |",
            "| :--- | :--- | :--- | :--- |",
        ])

        ds = forensics.get("driving_side", {})
        lines.append(f"| **Driving Side** | `{ds.get('driving_side')}` | {ds.get('confidence', 0)*100:.0f}% | Traffic Flow |")

        rm = forensics.get("road_markings", {})
        lines.append(f"| **Road Markings** | `{rm.get('line_color')}` lines | {rm.get('confidence', 0)*100:.0f}% | {', '.join(rm.get('suggested_regions', [])[:2])} |")

        up = forensics.get("utility_pole", {})
        lines.append(f"| **Utility Pole** | `{up.get('dominant_type') or 'none'}` | {up.get('confidence', 0)*100:.0f}% | {', '.join(up.get('regions', [])[:3])} |")

        lp = forensics.get("license_plate", {})
        lines.append(f"| **License Plate** | `{lp.get('format')}` (Euroband: {lp.get('has_euroband')}) | {lp.get('confidence', 0)*100:.0f}% | {', '.join(lp.get('candidate_regions', [])[:3])} |")

        sb = forensics.get("soil_and_biome", {})
        lines.append(f"| **Soil & Biome** | `{sb.get('soil_type')}` | {sb.get('soil_confidence', 0)*100:.0f}% | Vegetation: `{sb.get('vegetation_biome')}` |")

        ss = forensics.get("solar_shadow", {})
        lines.append(f"| **Solar Shadow** | Inferred `{ss.get('inferred_hemisphere')}` | {ss.get('confidence', 0)*100:.0f}% | Solar Azimuth |")

        if plonkit and plonkit.get("candidates"):
            lines.extend([
                "",
                "## 3. PlonkIt Grandmaster Infrastructure Signatures",
                "| Country (ISO) | Relative Score | Probability | Elimination Status |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for c in plonkit.get("candidates", [])[:5]:
                status = "❌ Falsified" if c.get("falsified") else "✅ Compatible"
                lines.append(f"| **{c.get('country')} ({c.get('iso')})** | {c.get('relative_score')} | {c.get('probability', 0)*100:.1f}% | {status} |")

        if fleet and fleet.get("demographics"):
            demo = fleet.get("demographics", {})
            lines.extend([
                "",
                "## 4. CarID Vehicle Fleet Demographics",
                f"- **Detected Vehicles:** {fleet.get('vehicle_count', 0)}",
                f"- **Pickup / SUV Ratio:** {demo.get('pickup_suv_ratio', 0)*100:.0f}%",
                f"- **Kei Car Ratio:** {demo.get('kei_car_ratio', 0)*100:.0f}% (Japan Discriminator)",
                f"- **Sedan / Hatchback Ratio:** {demo.get('sedan_hatchback_ratio', 0)*100:.0f}%",
                f"- **Top Regional Match:** {fleet.get('top_regional_match', {}).get('region') if fleet.get('top_regional_match') else 'Global Normal'}",
            ])

        if street and street.get("candidates"):
            lines.extend([
                "",
                "## 5. Topological Micro-GIS Street Targeter",
                "| Intersection | Confidence | Nearby POIs / Storefronts | Distance | Action |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for sc in street.get("candidates", [])[:5]:
                pois = ", ".join(sc.get("sample_pois", [])[:2]) or "Residential / Highway"
                lines.append(f"| **{sc.get('intersection')}** | {sc.get('confidence', 0)*100:.0f}% | {pois} | {sc.get('distance_from_center_m')}m | [Street View]({sc.get('google_street_view_url')}) |")

        if elim:
            lines.extend([
                "",
                "## 6. Negative-Evidence Falsification Trail",
            ])
            for e in elim[:8]:
                lines.append(f"- 🛑 **{e.get('city', 'Unknown')} ({e.get('country', '')})**: {'; '.join(e.get('reasons', []))}")

        lines.extend([
            "",
            "## 7. Deduction Chain",
        ])
        for step in dossier.get("reasoning_chain", []):
            lines.append(f"- {step}")

        return "\n".join(lines)

    def render_html(self, dossier: Dict[str, Any]) -> str:
        """Generates a self-contained, responsive HTML dossier with dark terminal styling."""
        v = dossier.get("verdict", {})
        lat = v.get("latitude", 0.0)
        lon = v.get("longitude", 0.0)
        md_text = self.render_markdown(dossier)

        # Simple HTML conversion
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{dossier.get('title', 'GeoVision Dossier')}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  .container {{ max-width: 1000px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5); }}
  h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 12px; }}
  h2 {{ color: #93c5fd; margin-top: 32px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
  #map {{ height: 350px; border-radius: 8px; margin: 20px 0; border: 1px solid #475569; }}
  .badge {{ background: #0284c7; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
  a {{ color: #38bdf8; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
  <h1>{dossier.get('title')}</h1>
  <p><strong>Generated:</strong> {dossier.get('generation_time')} | <strong>Coordinates:</strong> <span class="badge">{lat}, {lon}</span></p>
  <div id="map"></div>
  <pre style="white-space: pre-wrap; font-family: inherit; line-height: 1.6;">{md_text}</pre>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  var map = L.map('map').setView([{lat}, {lon}], 15);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);
  L.marker([{lat}, {lon}]).addTo(map).bindPopup('<b>{v.get("city")}</b><br>{v.get("top_street_intersection") or ""}').openPopup();
</script>
</body>
</html>
"""
        return html
