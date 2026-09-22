#!/usr/bin/env python3
"""
GeoVision CLI - Next Generation Geolocation AI Tool
Provides a rich, interactive, and beautiful terminal interface for the GeoVision pipeline.
"""

import sys
import os
import json
import warnings

# Suppress non-error warnings and HF noise from corrupting JSON stdout
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
if "--json-only" in sys.argv or "-j" in sys.argv:
    os.environ["TQDM_DISABLE"] = "1"

import typer
from typing import Optional, List, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint
from rich.markdown import Markdown

# Ensure we can import geovision_deep_scan
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import logging
import contextlib

@contextlib.contextmanager
def clean_json_context(json_only: bool):
    if not json_only:
        yield
        return
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout

if "--json-only" in sys.argv or "-j" in sys.argv:
    logging.disable(logging.CRITICAL)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.resolve() / "reports"

app = typer.Typer(
    name="geovision",
    help="GeoVision AI: Advanced OSINT & Geolocation CLI",
    add_completion=False,
    no_args_is_help=True
)
console = Console()

@app.command()
def scan(
    image: Path = typer.Argument(
        ..., 
        help="Path to the input image file for geolocation analysis",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir", "-o",
        help="Directory to save the reports (HTML and JSON)"
    ),
    near_park: bool = typer.Option(
        False, 
        "--near-park", "-p", 
        help="Enable advanced park proximity filtering"
    ),
    region: Optional[str] = typer.Option(
        None, 
        "--region", "-r", 
        help="Narrow the search to a specific region or city"
    ),
    date: Optional[str] = typer.Option(
        None,
        "--date", "-d",
        help="Date the image was taken (YYYY-MM-DD) for weather/season analysis. Overrides EXIF."
    ),
    no_vlm: bool = typer.Option(
        False, 
        "--no-vlm", 
        help="Skip the Vision-Language Model (AI reasoning) phase"
    ),
    interactive: bool = typer.Option(
        False, 
        "--interactive", "-i", 
        help="Open browser automatically for interactive verification"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable verbose debug logging"
    ),
    json_only: bool = typer.Option(
        False,
        "--json-only", "-j",
        help="Output raw JSON to stdout instead of rich formatting"
    )
):
    """
    Run a full GeoVision deep scan on an image.
    
    Extracts visual features, OCR text, deep features, performs database and satellite matching,
    runs cross-view verification, and synthesizes an AI-driven location estimate.
    """
    options = {
        "output_dir": output_dir,
        "near_park": near_park,
        "region": region,
        "date": date,
        "no_vlm": no_vlm,
        "interactive": interactive,
        "verbose": verbose,
    }

    try:
        from geovision_deep_scan import run_pipeline
    except ImportError as e:
        rprint(f"[red]Error importing GeoVision deep scan pipeline: {e}[/red]")
        raise typer.Exit(1)

    if json_only:
        with clean_json_context(True):
            result = run_pipeline(str(image), options)
        print(result.to_json())
        raise typer.Exit(0 if result.success else 1)

    # Rich beautiful interface
    console.print(Panel.fit(
        "[bold cyan]🛰️ GeoVision AI Deep Scan[/bold cyan]\n"
        "[dim]GeoSpy-class OSINT Geolocation Pipeline[/dim]",
        border_style="cyan"
    ))
    
    console.print(f"[{'cyan'}]Target Image:[/] {image.name}")
    if region:
        console.print(f"[{'cyan'}]Region Filter:[/] {region}")

    console.print("\n[bold]Initializing Deep Scan Pipeline...[/bold]\n")
    
    # We run the pipeline. Since the pipeline itself has logging, 
    # we can rely on the RichHandler we set up in geovision_deep_scan for inner logs.
    result = run_pipeline(str(image), options)

    console.print("\n[bold]Pipeline Execution Complete.[/bold]")
    
    # Render Synthesis Results Beautifully
    if not result.success:
        console.print(Panel(
            "[bold red]Pipeline failed to complete successfully.[/bold red]\n"
            f"Check logs or rerun with [cyan]--verbose[/cyan].\n\n"
            f"Errors: {', '.join(result.errors) if result.errors else 'Unknown'}",
            title="Scan Failed",
            border_style="red"
        ))
        raise typer.Exit(1)
        
    be = result.best_estimate
    if be:
        lat = be.get('latitude', '?')
        lon = be.get('longitude', '?')
        conf = be.get('confidence', 0)
        
        # Determine confidence color
        if conf > 0.8:
            color = "green"
        elif conf > 0.5:
            color = "yellow"
        else:
            color = "red"
            
        console.print(Panel(
            f"Latitude:  [bold {color}]{lat:.6f}[/bold {color}]\n"
            f"Longitude: [bold {color}]{lon:.6f}[/bold {color}]\n"
            f"Confidence: [{color}]{conf:.2%}[/{color}]\n\n"
            f"Sources: [dim]{', '.join(be.get('sources', []))}[/dim]",
            title="📍 Top Location Estimate",
            border_style="green",
            expand=False
        ))


    # Estimates Table
    if result.location_estimates:
        table = Table(title="Top Location Candidates", border_style="blue")
        table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
        table.add_column("Latitude", justify="right", style="magenta")
        table.add_column("Longitude", justify="right", style="magenta")
        table.add_column("Confidence", justify="right", style="green")
        table.add_column("Phase", justify="left", style="yellow")

        for i, est in enumerate(result.location_estimates[:5]):
            table.add_row(
                str(i + 1),
                f"{est.get('latitude', 0):.6f}",
                f"{est.get('longitude', 0):.6f}",
                f"{est.get('confidence', 0):.1%}",
                est.get('phase', 'N/A')
            )
        
        console.print(table)
        
    # Reports
    console.print(f"\n[bold]Reports saved to:[/bold] [cyan]{output_dir}[/cyan]")
    console.print("Run with [cyan]--interactive[/cyan] to open the map automatically.")


@app.command()
def quick(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Fast model-only geolocation (GeoCLIP + StreetCLIP + city snap + heuristics)."""
    with clean_json_context(json_only):
        from mcp_geovision_server import tool_geolocate_quick
        result = tool_geolocate_quick({"image_path": str(image)})
    if json_only:
        print(json.dumps(result, indent=2, default=str))
        return

    console.print(Panel.fit("[bold cyan]⚡ GeoVision Fast Triage[/bold cyan]", border_style="cyan"))
    estimates = result.get("estimates", [])
    if estimates:
        top = estimates[0]
        console.print(f"[bold green]Top Prediction:[/] {top['latitude']:.4f}, {top['longitude']:.4f}")
        if "nearest_city" in top:
            console.print(f"[cyan]Nearest City:[/] {top['nearest_city']} ({top.get('city_country', '')}) — {top.get('city_distance_km', 0):.1f}km away")
    
    sc = result.get("streetclip", {}).get("countries") or result.get("streetclip", {}).get("predictions", [])
    if sc:
        console.print("\n[bold]StreetCLIP Country Predictions:[/bold]")
        for c in sc[:3]:
            console.print(f"  • {c.get('country', '?')}: {c.get('confidence', 0):.1%}")

    clues = result.get("forensic_clues", [])
    if clues:
        console.print("\n[bold]GeoGuessr CV Clues:[/bold]")
        for clue in clues:
            console.print(f"  🔍 {clue}")


@app.command()
def heuristics(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Run fast computer vision classifiers (<100ms) for driving side, road marks, poles, plates, and biome."""
    from modules.geoguessr_heuristics import GeoGuessrAnalyzer
    analyzer = GeoGuessrAnalyzer()
    res = analyzer.analyze(str(image))
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return

    console.print(Panel.fit("[bold yellow]🚗 GeoGuessr CV Heuristics[/bold yellow]", border_style="yellow"))
    for clue in res.get("forensic_clues", []):
        console.print(f"  • {clue}")
    top_votes = res.get("summary", {}).get("top_regional_votes", [])
    if top_votes:
        console.print("\n[bold]Regional Votes:[/bold]")
        for r in top_votes:
            console.print(f"  🌐 {r['region']}: {r['votes']} vote(s)")


@app.command()
def ocr(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Extract visible text lines, signs, and numbers from an image."""
    from geovision_deep_scan import phase2_ocr
    res = phase2_ocr(str(image))
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(Panel.fit("[bold magenta]📝 OCR Text Extraction[/bold magenta]", border_style="magenta"))
    texts = res.get("all_texts", [])
    if texts:
        for t in texts:
            console.print(f"  • [cyan]{t}[/cyan]")
    else:
        console.print("  [dim]No text detected.[/dim]")


@app.command(name="reverse-search")
def reverse_search(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Un-gated reverse image search and landmark matching (Wikimedia Commons/Wikipedia)."""
    from modules.reverse_image_search import ReverseImageSearcher
    res = ReverseImageSearcher().search_similar_images(str(image))
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(Panel.fit("[bold green]🌐 Reverse Image Search (Open OSINT)[/bold green]", border_style="green"))
    matches = res.get("similar_images", [])
    if matches:
        for m in matches[:5]:
            console.print(f"  • [bold]{m.get('title', 'Unknown')}[/] (Score: {m.get('similarity_score', 0):.2f})")
            if m.get("latitude") and m.get("longitude"):
                console.print(f"    Coords: {m['latitude']:.4f}, {m['longitude']:.4f}")
            if m.get("url"):
                console.print(f"    URL: [dim]{m['url']}[/dim]")
    else:
        console.print("  [dim]No reverse matches found.[/dim]")


@app.command()
def resolve(
    clues_json: Optional[str] = typer.Option(None, "--clues", help="Raw JSON dictionary of visual clues"),
    city: Optional[str] = typer.Option(None, "--city", help="City hint"),
    country: Optional[str] = typer.Option(None, "--country", help="Country hint"),
    state: Optional[str] = typer.Option(None, "--state", help="State/province hint"),
    street: Optional[str] = typer.Option(None, "--street", help="Street or intersection name"),
    text: Optional[str] = typer.Option(None, "--text", help="Detected text or sign"),
    store: Optional[str] = typer.Option(None, "--store", help="Chain store / brand name"),
    amenity: Optional[str] = typer.Option(None, "--amenity", help="Amenity type"),
    near_park: bool = typer.Option(False, "--near-park", help="Image shows proximity to a park"),
    driving_side: Optional[str] = typer.Option(None, "--driving-side", help="Driving side (left/right)"),
    road_heading: Optional[float] = typer.Option(None, "--road-heading", help="Observed road compass heading in degrees (0-360)"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Resolve external visual observations (from any AI vision model) to exact GPS coordinates."""
    from modules.vision_clue_resolver import VisionClueResolver
    clues_dict = {}
    if clues_json:
        try:
            clues_dict = json.loads(clues_json)
        except Exception as e:
            rprint(f"[red]Error parsing --clues JSON: {e}[/red]")
            raise typer.Exit(1)
    if city:
        clues_dict["city_hint"] = city
    if country:
        clues_dict["country_hint"] = country
    if state:
        clues_dict["state_or_province"] = state
    if street:
        clues_dict.setdefault("street_names", []).append(street)
    if text:
        clues_dict.setdefault("detected_text", []).append(text)
    if store:
        clues_dict.setdefault("chain_stores", []).append(store)
    if amenity:
        clues_dict.setdefault("amenities", []).append(amenity)
    if near_park:
        clues_dict["near_park"] = True
    if driving_side:
        clues_dict["driving_side"] = driving_side
    if road_heading is not None:
        clues_dict["road_heading_deg"] = road_heading

    res = VisionClueResolver().resolve(clues_dict)
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return

    console.print(Panel.fit("[bold cyan]🧭 Vision Clue Resolver (GIS Grounding)[/bold cyan]", border_style="cyan"))
    be = res.get("best_estimate")
    if be:
        console.print(f"[bold green]Best Estimate:[/] {be['latitude']:.6f}, {be['longitude']:.6f} ([bold yellow]{be['confidence']:.1%}[/])")
        console.print(f"[bold]Address:[/] {be.get('display_name', 'N/A')}")
        if be.get("nearest_city"):
            nc = be["nearest_city"]
            console.print(f"[cyan]GeoNames Snap:[/] {nc['name']} ({nc['country_code']}) — {nc['distance_km']}km")
        if be.get("satellite"):
            console.print(f"[blue]Satellite Landcover:[/] {be['satellite'].get('classification')} (green: {be['satellite'].get('green_ratio', 0):.1%})")
        console.print(f"[magenta]Google Maps:[/] {be.get('google_maps_url')}")
        console.print("\n[bold]Evidence Chain:[/bold]")
        for ev in be.get("evidence", []):
            console.print(f"  • {ev}")
    else:
        console.print("[yellow]No grounded candidate could be resolved from the provided clues.[/yellow]")


@app.command()
def snap(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    max_km: float = typer.Option(75.0, "--max-km", help="Maximum snap radius in km"),
    k: int = typer.Option(3, "-k", help="Number of nearest cities to return"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Instant offline nearest-city lookup via the 70k GeoNames dataset."""
    from modules.geonames_city_snap import get_city_index
    idx = get_city_index()
    snapped = idx.snap(lat, lon, max_km=max_km)
    nearest = idx.nearest(lat, lon, k=k)
    data = {"snapped_city": snapped, "nearest_cities": nearest}
    if json_only:
        print(json.dumps(data, indent=2))
        return

    console.print(Panel.fit("[bold blue]📍 GeoNames City Snap[/bold blue]", border_style="blue"))
    if snapped:
        console.print(f"[bold green]Snapped City:[/] {snapped['name']} ({snapped['country_code']}) — {snapped['distance_km']:.1f}km (pop: {snapped.get('population', 0):,})")
    else:
        console.print(f"[yellow]No city within {max_km}km[/yellow]")
    console.print("\n[bold]Nearest Cities:[/bold]")
    for c in nearest:
        console.print(f"  • {c['name']} ({c['country_code']}): {c['distance_km']:.1f}km, pop {c.get('population', 0):,}")


@app.command()
def geocode(
    query: str = typer.Argument(..., help="Address, landmark, or city name to geocode"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Forward geocode a place name, street, or address to lat/lon (OSM Nominatim)."""
    from modules.nominatim_geocoder import NominatimGeocoder
    hit = NominatimGeocoder().forward_geocode(query)
    if json_only:
        print(json.dumps(hit or {}, indent=2))
        return
    if hit:
        console.print(f"[bold green]Coordinates:[/] {hit['latitude']:.6f}, {hit['longitude']:.6f}")
        console.print(f"[bold]Display Name:[/] {hit.get('display_name')}")
    else:
        console.print(f"[red]No geocoding match for '{query}'[/red]")


@app.command(name="reverse-geocode")
def reverse_geocode(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Reverse geocode lat/lon into human-readable address (OSM Nominatim)."""
    from modules.nominatim_geocoder import NominatimGeocoder
    hit = NominatimGeocoder().reverse_geocode(lat, lon)
    if json_only:
        print(json.dumps(hit or {}, indent=2))
        return
    if hit and "display_name" in hit:
        console.print(f"[bold green]Address:[/] {hit['display_name']}")
    else:
        console.print(f"[red]Could not reverse geocode {lat}, {lon}[/red]")


@app.command()
def osm(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    radius: int = typer.Option(500, "--radius", "-r", help="Radius in meters"),
    type: str = typer.Option("pois", "--type", "-t", help="Query type: pois, parks, roads, density, amenity_search"),
    name: Optional[str] = typer.Option(None, "--name", help="Amenity name for amenity_search"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Query OpenStreetMap infrastructure near coordinates."""
    from mcp_geovision_server import tool_osm_query
    res = tool_osm_query({"lat": lat, "lon": lon, "radius_m": radius, "query_type": type, "name_query": name})
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit(f"[bold cyan]🗺️ OSM Query ({type})[/bold cyan]", border_style="cyan"))
    console.print(json.dumps(res, indent=2))


@app.command()
def satellite(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    zoom: int = typer.Option(18, "--zoom", "-z", help="Zoom level (15-19)"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Inspect high-resolution satellite imagery (ESRI World Imagery) at coordinates."""
    from mcp_geovision_server import tool_satellite_landcover
    res = tool_satellite_landcover({"lat": lat, "lon": lon, "zoom": zoom})
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit("[bold blue]🛰️ Satellite Landcover[/bold blue]", border_style="blue"))
    console.print(f"[bold]Classification:[/] {res.get('classification')}")
    console.print(f"[bold]Green Ratio:[/] {res.get('green_ratio', 0):.1%}")
    if res.get("tile_path"):
        console.print(f"[dim]Tile saved:[/] {res['tile_path']}")


@app.command()
def weather(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    date: str = typer.Option(..., "--date", "-d", help="Date in YYYY-MM-DD format"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Retrieve historical daily weather from Open-Meteo archive for coordinates and date."""
    from mcp_geovision_server import tool_weather_corroborate
    res = tool_weather_corroborate({"lat": lat, "lon": lon, "date": date})
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit(f"[bold yellow]⛅ Historical Weather ({date})[/bold yellow]", border_style="yellow"))
    console.print(f"[bold]Condition:[/] {res.get('weather_description')}")
    console.print(f"[bold]Max Temp:[/] {res.get('max_temp_c')} °C")
    console.print(f"[bold]Precipitation:[/] {res.get('precipitation_mm')} mm")
    console.print(f"[bold]Snowfall:[/] {res.get('snowfall_cm')} cm")


@app.command()
def verify(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True),
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Visually verify candidate coordinates against image using reference photos and StreetCLIP."""
    from modules.candidate_visual_verification import verify_candidates
    res = verify_candidates(str(image), [{"latitude": lat, "longitude": lon}])
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit("[bold green]🔍 Visual Candidate Verification[/bold green]", border_style="green"))
    console.print(json.dumps(res, indent=2))


@app.command(name="road-heading")
def road_heading(
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="Image path to extract road perspective heading"),
    lat: Optional[float] = typer.Option(None, "--lat", help="Candidate latitude"),
    lon: Optional[float] = typer.Option(None, "--lon", help="Candidate longitude"),
    expected_heading: Optional[float] = typer.Option(None, "--heading", help="Expected compass azimuth in degrees (0-360)"),
    radius: int = typer.Option(300, "--radius", "-r", help="Radius in meters for OSM roads"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Extract road vanishing perspective or verify candidate coordinates against OSM road azimuths."""
    from modules.road_orientation_matcher import RoadOrientationMatcher
    out = {}
    with clean_json_context(json_only):
        matcher = RoadOrientationMatcher()
        if image:
            out["image_road_perspective"] = matcher.extract_image_road_heading(str(image))
        if lat is not None and lon is not None:
            if expected_heading is not None:
                out["candidate_alignment"] = matcher.match_candidate_road_alignment(
                    lat, lon, expected_azimuth_deg=expected_heading, radius_m=radius
                )
            else:
                out["osm_roads"] = matcher.fetch_osm_road_azimuths(lat, lon, radius_m=radius)

    if json_only:
        print(json.dumps(out, indent=2))
        return
    console.print(Panel.fit("[bold blue]🧭 Road Orientation & Azimuth Alignment[/bold blue]", border_style="blue"))
    console.print(json.dumps(out, indent=2))


@app.command()
def benchmark(
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to evaluation manifest JSON"),
    no_vlm: bool = typer.Option(True, "--no-vlm/--vlm", help="Run without external VLM"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Run GeoVision benchmark suite and report accuracy metrics (@1km, @25km, @200km, country)."""
    from scripts.run_eval import run_eval, DEFAULT_MANIFEST
    from modules.geo_eval_metrics import compute_metrics, format_report
    manifest = dataset or DEFAULT_MANIFEST
    with clean_json_context(json_only):
        results = run_eval(manifest, no_vlm=no_vlm)
        metrics = compute_metrics(results)
    if json_only:
        print(json.dumps(metrics, indent=2))
        return
    console.print(Panel.fit("[bold green]🏆 GeoVision Benchmark Evaluation[/bold green]", border_style="green"))
    console.print(format_report(metrics))


@app.command()
def environment(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Classify environmental scene biome (urban, suburban, rural, coastal, forest, mountain, desert, park, highway)."""
    from modules.environment_classifier import EnvironmentClassifier
    with clean_json_context(json_only):
        res = EnvironmentClassifier().classify(str(image))
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit(f"[bold cyan]🌲 Environment Classification[/bold cyan]", border_style="cyan"))
    console.print(f"[bold]Primary Type:[/] {res.get('primary_type')} (conf: {res.get('confidence')})")
    if res.get("secondary_types"):
        console.print(f"[bold]Secondary Types:[/] {', '.join(res.get('secondary_types'))}")
    console.print(json.dumps(res.get("features", {}), indent=2))


@app.command(name="ground-photos")
def ground_photos(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    radius: int = typer.Option(1000, "--radius", "-r", help="Radius in meters"),
    limit: int = typer.Option(6, "--limit", "-l", help="Number of photos to return"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Retrieve real ground-level photographs and street-level imagery near GPS coordinates."""
    from modules.ground_imagery_client import GroundImageryClient
    with clean_json_context(json_only):
        res = GroundImageryClient().get_nearby_ground_photos(lat, lon, radius_m=radius, limit=limit)
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit(f"[bold green]📸 Ground-Truth Reference Photos[/bold green]", border_style="green"))
    console.print(f"Total found within {radius}m: {res.get('total_found')}")
    for p in res.get("ground_photos", []):
        console.print(f"  • [bold]{p.get('title')}[/] ({p.get('distance_m')}m away) -> {p.get('thumbnail_url')}")


@app.command()
def uncertainty(
    lat: float = typer.Option(..., "--lat", help="Latitude"),
    lon: float = typer.Option(..., "--lon", help="Longitude"),
    conf: float = typer.Option(0.80, "--conf", help="Confidence score (0-1)"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Calculate 95% spatial uncertainty radius, bounding box, and geographic granularity."""
    from modules.uncertainty_estimator import UncertaintyEstimator
    with clean_json_context(json_only):
        res = UncertaintyEstimator().estimate([{"latitude": lat, "longitude": lon, "confidence": conf}])
    if json_only:
        print(json.dumps(res, indent=2))
        return
    console.print(Panel.fit(f"[bold yellow]🎯 Spatial Uncertainty & Granularity[/bold yellow]", border_style="yellow"))
    console.print(f"[bold]Uncertainty Radius:[/] ±{res.get('uncertainty_radius_km')} km")
    console.print(f"[bold]Scale Granularity:[/] {res.get('granularity')}")
    console.print(f"[bold]Confidence Level:[/] {res.get('confidence_level')}")
    console.print(json.dumps(res.get("bounding_box", {}), indent=2))


@app.command(name="investigate")
def investigate(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                 help="Path to the image to investigate"),
    case_name: str = typer.Option(None, "--case-name", help="Case name for case-file persistence"),
    save_case: bool = typer.Option(False, "--save-case", help="Persist investigation to the SQLite CaseManager"),
    hint: str = typer.Option(None, "--hint", help="Optional location hint (region/city)"),
    top_k: int = typer.Option(3, "--top-k", help="Number of top candidates to VLM-verify"),
    radius: float = typer.Option(1500.0, "--radius", help="Regional-retrieval radius in km around the coarse GPS prior"),
    no_regional: bool = typer.Option(False, "--no-regional", help="Disable region-constrained retrieval (global CLIP search)"),
    listings: bool = typer.Option(False, "--listings", help="Snapshot nearby property/business listings into the case"),
    report: bool = typer.Option(False, "--report", help="Auto-write a professional Markdown case report"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Full multi-stage investigation harness: VLM coarse reason → REGIONAL
    constrained VPR → VLM verification → case record. Any vision model can be the
    VLM via GEOVISION_VLM_* env vars; deterministic signal runs regardless.

    Regional retrieval (better than global GeoSpy-style search) restricts the
    CLIP reference-DB lookup to real geotagged photos within RADIUS km of the
    coarse GeoCLIP prior, pruning cross-hemisphere false matches before ranking.
    """
    from modules.geo_harness import GeoVisionHarness
    with clean_json_context(json_only):
        res = GeoVisionHarness().investigate(
            str(image), location_hint=hint or None,
            save_case=save_case, case_name=case_name,
            case_tags=["harness"], top_k_verify=top_k,
            radius_km=radius, use_regional=not no_regional,
            with_listings=listings,
            auto_report=report)
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    best = res.get("best_estimate") or {}
    console.print(Panel.fit("[bold magenta]🕵️  GeoVision Investigation Harness[/bold magenta]", border_style="magenta"))
    console.print(f"[bold]Image:[/] {res.get('image_path')}")
    if best:
        console.print(f"[bold green]Best Estimate:[/] {best.get('latitude')}, {best.get('longitude')} "
                      f"(conf {best.get('confidence')})")
    console.print("[bold]Stage statuses:[/] "
                  + ", ".join(f"{k}={v.get('status','?') if isinstance(v,dict) else '?'}"
                              for k, v in res.get("stages", {}).items()))
    constr = res.get("constraints_applied") or []
    if constr:
        console.print("[bold]Negative-evidence constraints:[/]")
        for c in constr:
            console.print(f"  • {c}")
    if res.get("reasoning_chain"):
        console.print("[bold]Reasoning chain:[/]")
        for r in res["reasoning_chain"]:
            console.print(f"  → {r}")
    console.print(f"[dim]Duration: {res.get('duration_s')}s | "
                  f"Case ID: {res.get('case_id','(not saved)')}[/dim]")


@app.command(name="grow-db")
def grow_db(
    lat: float = typer.Option(..., "--lat", help="Coarse GPS prior latitude (region center)"),
    lon: float = typer.Option(..., "--lon", help="Coarse GPS prior longitude"),
    radius: int = typer.Option(5000, "--radius", "--radius-m", help="Search radius in meters around the prior"),
    per_rate: int = typer.Option(40, "--per-rate", help="Max geosearch hits to request"),
    max_new: int = typer.Option(200, "--max-new", help="Max new real references to append"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Grow the visual_geo_db retrieval index with REAL geotagged Wikimedia photos
    near a GPS prior. Each appended reference is a real photograph with real GPS.
    Run iteratively per region to densify retrieval coverage (the GeoSpy-style
    'more real data' path)."""
    from modules.geo_harness import GeoVisionHarness
    with clean_json_context(json_only):
        res = GeoVisionHarness().grow_reference_db(lat, lon, radius_m=radius,
                                                   per_rate=per_rate, max_new=max_new)
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(Panel.fit("[bold cyan]📈 Grow Reference DB[/bold cyan]", border_style="cyan"))
    console.print(f"[bold]Prior:[/] ({lat}, {lon}) ±{radius}m")
    console.print(f"[bold]status:[/] {res.get('status')} | added: {res.get('added')} | "
                  f"db_size_after: {res.get('db_size_after')}")
    console.print(f"[dim]{res.get('note')}[/dim]")


@app.command(name="find-listing")
def find_listing(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                 help="Path to an indoor/listing screenshot"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Find the exact house/listing from an indoor (or any) photo via reverse-image
    search. Uses Google Vision (needs GOOGLE_VISION_API_KEY) when set, else a key-free
    Google-image scrape. Returns matched pages, listing URLs, and geocoded addresses.
    Honest: only works if the photo (or a near-duplicate) is indexed online."""
    from modules.listing_finder import find_listing as _fl
    with clean_json_context(json_only):
        res = _fl(str(image))
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(Panel.fit("[bold yellow]🏠 Listing Finder[/bold yellow]", border_style="yellow"))
    console.print(f"[bold]Image:[/] {image}")
    console.print(f"[bold]Engine:[/] {res.get('engine')} | status: {res.get('status')}")
    if res.get("geolocated"):
        g = res["geolocated"]
        console.print(f"[bold green]Listing:[/] {g.get('address')} -> {g.get('latitude')}, {g.get('longitude')}")
        console.print(f"[dim]{g.get('display_name')}[/dim]")
    else:
        console.print(f"[dim]{res.get('note')}[/dim]")
    for u in res.get("listing_pages", [])[:5]:
        console.print(f"  • listing: {u[:90]}")
    for u in res.get("visual_matches", [])[:5]:
        console.print(f"  · match: {u[:90]}")


@app.command(name="track-properties")
def track_properties(
    lat: float = typer.Option(None, "--lat", help="Latitude of the point of interest"),
    lon: float = typer.Option(None, "--lon", help="Longitude of the point of interest"),
    address: str = typer.Option(None, "--address", help="Instead of lat/lon, geocode this address"),
    radius: int = typer.Option(2000, "--radius", "--radius-m", help="Search radius in meters"),
    cats: str = typer.Option("lodging,commercial,dining,office",
                             "--cats", help="Comma-separated categories"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """List and track nearby hospitality/rental/commercial/property listings (hotels,
    guest houses, apartments, offices, shops, dining) from real OSM/Overpass at a
    coordinate or address. Key-free. Save the JSON `listings` and pass them back to
    detect new/gone listings between runs (change monitoring)."""
    from modules.property_locator import PropertyLocator
    pl = PropertyLocator()
    cats_l = [c.strip() for c in cats.split(",") if c.strip()]
    with clean_json_context(json_only):
        if address:
            res = pl.from_address(address, radius_m=radius, categories=cats_l)
        elif lat is not None and lon is not None:
            res = pl.list_nearby(lat, lon, radius_m=radius, categories=cats_l)
        else:
            print("provide --lat/--lon or --address", file=sys.stderr)
            raise typer.Exit(code=1)
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(Panel.fit("[bold blue]🏨 Property & Business Listings[/bold blue]", border_style="blue"))
    console.print(f"[bold]Target:[/] {address or f'({lat},{lon})'} ±{radius}m")
    console.print(f"[bold]Found:[/] {res.get('count')} listings (status {res.get('status')})")
    for l in res.get("listings", []):
        tag = l.get("tags", {})
        extra = ", ".join(str(tag.get(k)) for k in ("phone", "website", "stars") if tag.get(k))
        console.print(f"  • [{l.get('type')}] {l.get('name')} @ {l.get('latitude'):.5f},{l.get('longitude'):.5f}"
                      + (f" — {extra}" if extra else ""))
    console.print("[dim]To track changes: save the JSON `listings` and re-run with the same"
                  " coords, passing prior listings via the track() API (change diff).[/dim]")


@app.command(name="render-report")
def render_report(
    case_file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                     help="Path to a saved case JSON (e.g. reports/case_*.json)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output .md or .html file (default: alongside input)"),
    title: str = typer.Option("GeoVision Case Report", "--title", "-t", help="Report title")
):
    """Render a saved investigation JSON into a professional, shareable Markdown or
    HTML case report (honest confidence bands, full evidence/reasoning chain,
    candidate table, reference imagery, sources). Output format from the extension:
    .md or .html."""
    import json as _json
    from modules.case_report import render_case
    rec = _json.loads(Path(case_file).read_text(encoding="utf-8"))
    out = output or Path(case_file).with_suffix(".md")
    path = render_case(rec, str(out), title=title)
    console.print(Panel.fit("[bold green]📄 Case Report[/bold green]", border_style="green"))
    console.print(f"[bold]Case:[/] {case_file}")
    console.print(f"[bold]Status:[/] {rec.get('status')} | Best: "
                  f"{rec.get('best_estimate') or 'none'}")
    console.print(f"[bold]Rendered:[/] [cyan]{path}[/cyan] ({path.stat().st_size} bytes)")
    return path


@app.command(name="signals")
def signals(
    lat: float = typer.Option(None, "--lat", help="Latitude anchor"),
    lon: float = typer.Option(None, "--lon", help="Longitude anchor"),
    query: str = typer.Option(None, "--query", "-q", help="Text query (geocoded if no lat/lon)"),
    address: str = typer.Option(None, "--address", help="Address anchor"),
    image: Path = typer.Option(None, "--image", help="Optional image for reverse-image signal"),
    radius: int = typer.Option(2500, "--radius", help="OSINT/imagery radius in meters"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout")
):
    """Pull many independent LIVE signal sources for an anchor (coords, address, or
    text query) and bundle them for cross-examination: web search, ground imagery,
    OSM/Overpass business OSINT, weather, GeoNames city snap, reverse-geocode, and
    optional reverse-image. Key-free. A CV/model agent reasons over the bundle.

    Example:
      geovision signals --query "Colosseum Rome"
      geovision signals --lat 48.8584 --lon 2.2945 --image photo.jpg"""
    from modules.live_signal_orchestrator import LiveSignalOrchestrator
    with clean_json_context(json_only):
        res = LiveSignalOrchestrator().investigate(
            lat=lat, lon=lon, query=query, address=address,
            image_path=str(image) if image else None, radius_m=radius)
    if json_only:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(Panel.fit("[bold cyan]🛰️  Live Signal Orchestrator[/bold cyan]", border_style="cyan"))
    console.print(res.get("note", ""))
    for name, sig in res.get("signals", {}).items():
        st = sig.get("status", "?") if isinstance(sig, dict) else "?"
        cnt = ""
        if isinstance(sig, dict):
            for k in ("count", "weather", "city", "place", "coords"):
                if sig.get(k):
                    cnt = f" | {k}={sig[k]}" if not isinstance(sig[k], (dict, list)) else ""
                    break
        console.print(f"  [bold]{name}[/]: {st}{cnt}")
    img = res.get("signals", {}).get("ground_imagery", {})
    for p in (img.get("photos") or [])[:3]:
        console.print(f"    • {p.get('source')}: {str(p.get('thumbnail_url',''))[:60]}")


@app.command(name="canvas")
def canvas_cmd(
    image: Path = typer.Argument(None, file_okay=True, dir_okay=False, readable=True,
                                 help="Image to investigate on the canvas"),
    session: str = typer.Option(None, "--session", "-s", help="Canvas session name (default: timestamped)"),
    title: str = typer.Option("GeoVision Investigation", "--title", help="Board title"),
    listings: bool = typer.Option(False, "--listings", help="Also snapshot nearby property/business listings onto the canvas"),
    radius: float = typer.Option(1500.0, "--radius", help="Regional-retrieval radius km"),
    no_regional: bool = typer.Option(False, "--no-regional", help="Disable regional retrieval"),
):
    """Open a LIVE detective canvas and (if an image is given) run the full
    investigation streaming onto it: every step, pulled reference images,
    side-by-side visual comparisons with scores, pinned candidates, and the
    verdict. Prints the viewer URL — open it in a browser; it auto-refreshes.
    Without an image, just opens an empty board (drive it via canvas_* MCP tools)."""
    from modules.canvas import DetectiveCanvas
    c = DetectiveCanvas(session=session, title=title)
    console.print(Panel.fit("[bold yellow]🎨 Detective Canvas[/bold yellow]", border_style="yellow"))
    console.print(f"[bold]Viewer:[/] [cyan]{c.viewer_url}[/cyan]  (auto-refreshes — keep it open)")
    if image is None:
        c.add("note", text="Empty board. Drive me via the canvas_* MCP tools.")
        return
    from modules.geo_harness import GeoVisionHarness
    h = GeoVisionHarness()
    if listings:
        c.add("step", title="Listings snapshot", text="will pull nearby hotels/offices/shops after geolocation")
    rec = h.investigate(str(image), canvas=c, radius_km=radius,
                        use_regional=not no_regional, with_listings=listings)
    if listings and (rec.get("best_estimate") or {}).get("latitude") is not None:
        b = rec["best_estimate"]
        for l in (rec.get("nearby_listings", {}).get("listings") or [])[:8]:
            c.add("evidence", title=f"[{l.get('type')}] {l.get('name')}",
                  text=f"{l.get('address','').strip()} @ {l.get('latitude'):.5f},{l.get('longitude'):.5f}",
                  lat=l.get("latitude"), lon=l.get("longitude"))
        c.add("summary", title="Nearby listings",
              text=f"{rec.get('nearby_listings',{}).get('count',0)} real listings around the estimate (OSM/Overpass)")
    best = rec.get("best_estimate") or {}
    c.finish(f"Investigation complete — status={rec.get('status')} "
             f"best={best.get('latitude')},{best.get('longitude')} "
             f"(canvas holds {len(c.events)} events)")
    console.print(f"[bold]Result:[/] {rec.get('status')} — "
                  f"{best.get('latitude')}, {best.get('longitude')} "
                  f"conf={best.get('confidence')}")
    console.print(f"[bold]Board:[/] {len(c.events)} events — [cyan]{c.viewer_url}[/cyan]")


@app.command(name="grandmaster")
def grandmaster_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                 help="Path to image for zero-storage Grandmaster geolocation"),
    evidence: Optional[str] = typer.Option(None, "--evidence", "-e", help="Optional observed evidence summary"),
    hint: Optional[str] = typer.Option(None, "--hint", "-h", help="Optional location hint"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of candidate coordinates to return"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Zero-storage Grandmaster Geolocation (GeoSpy Raven-class).
    Combines multi-scale patch spatial consensus, negative-evidence falsification lattice
    (driving side, road lines, poles, plates, soil, sun), and JIT OpenStreetMap micro-GIS
    grounding without requiring a large reference photo database."""
    from modules.grandmaster_forensics import GrandmasterForensicsEngine
    engine = GrandmasterForensicsEngine()
    with clean_json_context(json_only):
        if not json_only:
            console.print(Panel.fit("[bold magenta]🎯 GeoVision Grandmaster Forensics Engine[/bold magenta]\n"
                                    "[dim]Zero-Storage Raven-Class Spatial Consensus & Falsification Lattice[/dim]",
                                    border_style="magenta"))
        dossier = engine.investigate(str(image), evidence_summary=evidence, location_hint=hint, top_k=top_k)

    if json_only:
        sys.stdout.write(json.dumps(dossier, indent=2) + "\n")
        return

    best = dossier.get("best_estimate", {})
    console.print(f"\n[bold green]📍 Pinned Location:[/] [bold]{best.get('city')}, {best.get('country')}[/bold]")
    console.print(f"   [cyan]GPS:[/] {best.get('latitude'):.6f}, {best.get('longitude'):.6f} "
                  f"(±{best.get('uncertainty_radius_km')} km radius, tier: [bold]{best.get('precision_tier')}[/bold])")
    console.print(f"   [yellow]Confidence:[/] {best.get('confidence', 0.0):.1%}")
    console.print(f"   [blue]Maps:[/] {best.get('google_maps_url')}")

    # Forensic Breakdown Table
    fb = dossier.get("forensic_breakdown", {})
    t = Table(title="🔬 Physical Visual Forensics", show_header=True, header_style="bold cyan")
    t.add_column("Signal Category")
    t.add_column("Detected Signature")
    t.add_column("Confidence / Value")

    ds = fb.get("driving_side", {})
    t.add_row("Driving Side", str(ds.get("driving_side", "unknown")).upper(), f"{ds.get('confidence',0):.1%}")

    rm = fb.get("road_markings", {})
    t.add_row("Road Markings", str(rm.get("line_color", "none")).title(), f"{rm.get('confidence',0):.1%}")

    lp = fb.get("license_plate", {})
    euro = " [blue](Euroband)[/blue]" if lp.get("has_euroband") else ""
    t.add_row("License Plate", f"{lp.get('format', 'none')}{euro}", f"{lp.get('confidence',0):.1%}")

    up = fb.get("utility_pole", {})
    t.add_row("Utility Pole", str(up.get("dominant_type", "none")), f"{up.get('confidence',0):.1%}")

    sb = fb.get("soil_and_biome", {})
    t.add_row("Soil / Biome", f"{sb.get('soil_type','temperate')} / {sb.get('vegetation_biome','temperate')}", f"{sb.get('soil_confidence',0):.1%}")

    ss = fb.get("solar_shadow", {})
    t.add_row("Solar Hemisphere", str(ss.get("inferred_hemisphere", "unknown")).title(), f"{ss.get('confidence',0):.1%}")
    console.print(t)

    # Elimination Matrix
    elim = dossier.get("elimination_matrix", [])
    if elim:
        et = Table(title=f"🚫 Falsification Lattice ({len(elim)} Regions Disproven)", show_header=True, header_style="bold red")
        et.add_column("Falsified Candidate")
        et.add_column("Physical Contradiction")
        for e in elim[:5]:
            et.add_row(f"{e.get('city')}, {e.get('country')}", "; ".join(e.get("reasons", [])[:2]))
        console.print(et)

    # Reasoning Chain
    console.print("\n[bold]🧠 Grandmaster Deduction Chain:[/]")
    for step in dossier.get("reasoning_chain", []):
        console.print(f"  [dim]•[/dim] {step}")


@app.command(name="forensics")
def forensics_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                 help="Path to image for visual forensics analysis"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Run local OpenCV visual forensics (<100ms, offline):
    Driving side, road line colors, utility pole architecture, license plate morphology,
    bollards, soil ecology, and solar shadow vector."""
    from modules.grandmaster_forensics import GrandmasterForensicsEngine
    engine = GrandmasterForensicsEngine()
    fb = engine.extract_visual_forensics(str(image))

    if json_only:
        sys.stdout.write(json.dumps(fb, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold cyan]🔬 OpenCV Visual Forensics Engine[/bold cyan]", border_style="cyan"))
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Signal")
    t.add_column("Classification")
    t.add_column("Confidence")
    t.add_column("Geographic Implication")

    ds = fb.get("driving_side", {})
    t.add_row("Driving Side", str(ds.get("driving_side", "unknown")), f"{ds.get('confidence',0):.1%}",
              ", ".join(ds.get("candidate_countries", [])[:4]) or "—")

    rm = fb.get("road_markings", {})
    t.add_row("Road Markings", str(rm.get("line_color", "none")), f"{rm.get('confidence',0):.1%}",
              ", ".join(rm.get("suggested_regions", [])[:2]) or "—")

    up = fb.get("utility_pole", {})
    t.add_row("Utility Pole", str(up.get("dominant_type", "none")), f"{up.get('confidence',0):.1%}",
              ", ".join(up.get("regions", [])[:3]) or "—")

    lp = fb.get("license_plate", {})
    t.add_row("License Plate", str(lp.get("format", "none")), f"{lp.get('confidence',0):.1%}",
              ", ".join(lp.get("candidate_regions", [])[:3]) or "—")

    sb = fb.get("soil_and_biome", {})
    t.add_row("Soil", str(sb.get("soil_type", "temperate")), f"{sb.get('soil_confidence',0):.1%}",
              ", ".join(sb.get("soil_candidate_regions", [])[:3]) or "—")

    ss = fb.get("solar_shadow", {})
    t.add_row("Solar Hemisphere", str(ss.get("inferred_hemisphere", "unknown")), f"{ss.get('confidence',0):.1%}",
              f"Shadow Angle: {ss.get('shadow_angle_deg', 0)}°")

    console.print(t)


@app.command(name="hermes")
def hermes_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                 help="Path to image for Hermes autonomous investigation"),
    goal: Optional[str] = typer.Option(None, "--goal", "-g", help="Specific investigation goal"),
):
    """Run an autonomous geolocation investigation with Hermes Agent.
    Drives Hermes using the integrated geovision MCP tools and the geovision-geolocate skill."""
    import subprocess
    hermes_bin = Path.home() / "hermes-env" / "bin" / "hermes"
    if not hermes_bin.exists():
        console.print("[bold red]Hermes binary not found at ~/hermes-env/bin/hermes[/bold red]")
        raise typer.Exit(1)

    abs_image = str(image.resolve())
    prompt = (
        f"Perform an exhaustive geolocation investigation of the image at '{abs_image}' using "
        f"the geovision MCP tools (geovision_grandmaster_locate, geovision_forensic_breakdown, "
        f"or investigate_image). Falsify impossible regions, cite physical evidence (driving side, "
        f"road marks, poles, plates, soil), and report the final pinned coordinates with confidence."
    )
    if goal:
        prompt += f" Goal: {goal}"

    console.print(Panel.fit(f"[bold cyan]🤖 Invoking Hermes Autonomous Geolocation Agent...[/bold cyan]\n"
                            f"[dim]Target: {abs_image}[/dim]", border_style="cyan"))
    cmd = [str(hermes_bin), "-z", prompt]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Hermes execution exited with code {e.returncode}[/bold red]")
        raise typer.Exit(e.returncode)


@app.command(name="street-target")
def street_target_cmd(
    lat: float = typer.Option(..., "--lat", help="Candidate latitude"),
    lon: float = typer.Option(..., "--lon", help="Candidate longitude"),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="Optional image to extract heading and OCR"),
    radius: int = typer.Option(1500, "--radius", "-r", help="Search radius in meters"),
    heading: Optional[float] = typer.Option(None, "--heading", help="Road heading / azimuth angle (0-180)"),
    ocr: Optional[List[str]] = typer.Option(None, "--ocr", help="Storefront / sign OCR text clues"),
    amenity: Optional[List[str]] = typer.Option(None, "--amenity", help="Amenity categories (e.g. cafe, pharmacy)"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Topological micro-GIS intersection targeter using OpenStreetMap."""
    from modules.street_targeter import StreetTargeter
    targeter = StreetTargeter()
    res = targeter.target_street(
        lat=lat,
        lon=lon,
        radius_m=radius,
        expected_heading_deg=heading,
        ocr_clues=ocr,
        amenity_clues=amenity,
        image_path=str(image) if image else None,
    )

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit(f"[bold cyan]🎯 Street Targeter (OSM Topology at {lat}, {lon})[/bold cyan]", border_style="cyan"))
    candidates = res.get("candidates", [])
    if not candidates:
        console.print("[yellow]No road intersections identified within search radius.[/yellow]")
        return

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Intersection")
    t.add_column("Conf")
    t.add_column("Distance")
    t.add_column("Nearby Storefronts / POIs")
    t.add_column("Maps Link")

    for c in candidates[:6]:
        pois = ", ".join(c.get("sample_pois", [])[:2]) or "—"
        t.add_row(
            c.get("intersection", ""),
            f"{c.get('confidence', 0):.1%}",
            f"{c.get('distance_from_center_m', 0):.0f}m",
            pois,
            f"[link={c.get('osm_url')}]OSM[/link]",
        )
    console.print(t)


@app.command(name="plonkit")
def plonkit_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Image path to evaluate"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """PlonkIt Grandmaster Infrastructure Evaluation (85-country rules)."""
    from modules.plonkit_meta_engine import PlonkitMetaEngine
    engine = PlonkitMetaEngine()
    res = engine.scan_image(str(image))

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold green]🌍 PlonkIt Grandmaster Infrastructure Meta[/bold green]", border_style="green"))
    t = Table(show_header=True, header_style="bold green")
    t.add_column("Country (ISO)")
    t.add_column("Relative Score")
    t.add_column("Probability")
    t.add_column("Status")

    for c in res.get("candidates", [])[:8]:
        status = "[red]❌ Falsified[/red]" if c.get("falsified") else "[green]✅ Compatible[/green]"
        t.add_row(
            f"{c.get('country')} ({c.get('iso')})",
            str(c.get("relative_score")),
            f"{c.get('probability', 0):.1%}",
            status,
        )
    console.print(t)


@app.command(name="car-id")
def car_id_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Image path to profile"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """CarID vehicle silhouette and fleet demographic profiler."""
    from modules.car_fleet_identifier import CarFleetIdentifier
    identifier = CarFleetIdentifier()
    res = identifier.analyze(str(image))

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold magenta]🚗 CarID Vehicle Fleet Profiler[/bold magenta]", border_style="magenta"))
    demo = res.get("demographics", {})
    console.print(f"[bold]Detected Vehicles:[/] {res.get('vehicle_count', 0)}")
    console.print(f"[bold]Pickup/SUV Ratio:[/] {demo.get('pickup_suv_ratio', 0):.1%}")
    console.print(f"[bold]Kei Car Ratio:[/] {demo.get('kei_car_ratio', 0):.1%}")
    console.print(f"[bold]Sedan/Hatchback Ratio:[/] {demo.get('sedan_hatchback_ratio', 0):.1%}")
    console.print(f"[bold]Top Regional Match:[/] {res.get('top_regional_match', {}).get('region', 'Global Normal')}")
    console.print(f"[bold]Inferred Traffic Side:[/] {res.get('inferred_traffic_side', 'unknown')}")


@app.command(name="dossier")
def dossier_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Image path to investigate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (e.g. report.html or report.md)"),
    format_type: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown, html, json"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Generate complete Raven-class forensic intelligence dossier."""
    from mcp_geovision_server import tool_geovision_dossier
    res = tool_geovision_dossier({"image_path": str(image), "format": format_type})

    if json_only or format_type == "json":
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    content = res.get("dossier_markdown") or res.get("dossier_html") or json.dumps(res, indent=2)
    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Dossier saved to {output}[/green]")
    else:
        console.print(content)


@app.command(name="skyeye")
def skyeye_cmd(
    lat: float = typer.Option(..., "--lat", help="Candidate latitude"),
    lon: float = typer.Option(..., "--lon", help="Candidate longitude"),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="Optional query image to extract facade features"),
    radius: int = typer.Option(400, "--radius", "-r", help="Search radius in meters"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """SkyEye cross-view building facade and overhead satellite footprint verifier."""
    from modules.skyeye_footprint_verifier import SkyEyeFootprintVerifier
    verifier = SkyEyeFootprintVerifier()
    res = verifier.verify_candidate_footprints(
        lat=lat,
        lon=lon,
        image_path=str(image) if image else None,
        radius_m=radius,
    )

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit(f"[bold blue]🛰️ SkyEye Overhead Satellite Footprint Verifier ({lat}, {lon})[/bold blue]", border_style="blue"))
    gf = res.get("ground_features", {})
    if gf:
        console.print(f"[bold]Detected Roof Archetype:[/] {gf.get('roof_archetype')} (conf {gf.get('confidence', 0):.0%})")
        console.print(f"[bold]Perspective Facade Yaw:[/] {gf.get('facade_yaw_deg')}°")

    footprints = res.get("candidates", [])
    if not footprints:
        console.print("[yellow]No building footprints found within search radius.[/yellow]")
        return

    t = Table(show_header=True, header_style="bold blue")
    t.add_column("Building")
    t.add_column("Type / Levels")
    t.add_column("Roof Shape")
    t.add_column("Orientation")
    t.add_column("Cross-View Match")

    for fp in footprints[:6]:
        t.add_row(
            fp.get("name", "Building"),
            f"{fp.get('building_type')} (L{fp.get('levels') or '?'})",
            fp.get("roof_shape", "unspecified"),
            f"{fp.get('principal_orientation_deg')}°",
            f"{fp.get('alignment_confidence', 0):.1%}",
        )
    console.print(t)


@app.command(name="solar-lock")
def solar_lock_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to image with visible shadows"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Optional date in YYYY-MM-DD format"),
    season: Optional[str] = typer.Option(None, "--season", "-s", help="Optional season: summer, winter, spring_fall"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Solar Lock: NOAA astronomical ephemeris shadow and latitude solver."""
    from modules.solar_lock_solver import SolarLockSolver
    solver = SolarLockSolver()
    res = solver.analyze_image_shadows(str(image), date_str=date, approx_season=season)

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold yellow]☀️ Solar Lock: Astronomical Latitude Envelope[/bold yellow]", border_style="yellow"))
    bounds = res.get("latitude_bounds", [0, 0])
    console.print(f"[bold]Inferred Hemisphere:[/] [green]{res.get('inferred_hemisphere', 'unknown').upper()}[/green]")
    console.print(f"[bold]Constrained Latitude Band:[/] [cyan]{bounds[0]}° to {bounds[1]}°[/cyan] (Center: {res.get('center_latitude_deg')}°)")
    console.print(f"[bold]Solar Elevation:[/] {res.get('solar_elevation_deg')}° | [bold]Zenith Angle:[/] {res.get('zenith_angle_deg')}°")
    console.print(f"[bold]Shadow Azimuth:[/] {res.get('shadow_azimuth_deg')}° (detected {res.get('detected_shadow_lines', 0)} shadow lines)")


@app.command(name="mountain-ridge")
def mountain_ridge_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to mountain / wilderness landscape image"),
    hint: Optional[str] = typer.Option(None, "--hint", "-H", help="Optional mountain range or region hint"),
    lat: Optional[float] = typer.Option(None, "--lat", help="Optional candidate latitude"),
    lon: Optional[float] = typer.Option(None, "--lon", help="Optional candidate longitude"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Mountain Ridge & DEM Skyline Solver: matches horizon contour against global topography without reference photos."""
    from modules.mountain_ridge_matcher import MountainRidgeMatcher
    matcher = MountainRidgeMatcher()
    res = matcher.analyze_image_horizon(str(image), hint_region=hint, candidate_lat=lat, candidate_lon=lon)

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold green]🏔️ Mountain Ridge & DEM Skyline Solver[/bold green]", border_style="green"))
    console.print(f"[bold]Skyline Detected:[/] {res.get('skyline_detected')} (Roughness: {res.get('topographic_roughness')})")
    console.print(f"[bold]Top Mountain Range Match:[/] [cyan]{res.get('top_mountain_range')}[/cyan] ({res.get('top_region')})")
    console.print(f"[bold]Topographic Confidence:[/] [green]{res.get('confidence', 0):.1%}[/green] | [bold]Peaks Detected:[/] {res.get('peaks_detected')}")
    for cand in res.get("candidates", [])[:3]:
        console.print(f"  • [bold]{cand.get('range')}[/] ({cand.get('region')}): {cand.get('confidence', 0):.0%} confidence")


@app.command(name="ecoregion")
def ecoregion_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to landscape/vegetation image"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Global Ecoregion & Soil Bio-Classifier: classifies WWF biome and soil taxonomy without reference photos."""
    from modules.ecoregion_classifier import EcoregionClassifier
    classifier = EcoregionClassifier()
    res = classifier.extract_bio_spectral_features(str(image))

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold magenta]🌿 Global Ecoregion & Soil Bio-Classifier[/bold magenta]", border_style="magenta"))
    console.print(f"[bold]Classified Biome:[/] [cyan]{res.get('top_biome')}[/cyan] (Archetype: {res.get('foliage_archetype')})")
    console.print(f"[bold]Identified Soil Order:[/] [yellow]{res.get('soil_type')}[/yellow] (conf {res.get('soil_confidence', 0):.0%})")
    console.print(f"[bold]Vegetation Coverage Ratio:[/] {res.get('vegetation_coverage_ratio', 0):.1%}")
    console.print(f"[bold]Inferred Latitude Limits:[/] {res.get('inferred_lat_range')}")
    console.print(f"[bold]Dominant Global Jurisdictions:[/] {', '.join(res.get('top_countries', []))}")


@app.command(name="hermes-solve")
def hermes_solve_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to query image"),
    hint: Optional[str] = typer.Option(None, "--hint", "-H", help="Optional location hint"),
    season: Optional[str] = typer.Option(None, "--season", "-s", help="Optional season: summer, winter, spring_fall"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Autonomous Hermes & Antigravity Multi-Agent Collaborative Solver."""
    from modules.hermes_collaborative_solver import HermesCollaborativeSolver
    solver = HermesCollaborativeSolver()
    res = solver.run_collaborative_investigation(str(image), location_hint=hint, season_hint=season)

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    verdict = res.get("verdict", {})
    console.print(Panel.fit("[bold gold1]🤖 Hermes & Antigravity Autonomous Multi-Agent Consensus[/bold gold1]", border_style="gold1"))
    console.print(f"[bold]Pinpoint Coordinates:[/] [cyan]{verdict.get('latitude')}, {verdict.get('longitude')}[/cyan]")
    console.print(f"[bold]Target Location:[/] {verdict.get('street')} ({verdict.get('city')}, {verdict.get('country')})")
    console.print(f"[bold]Consensus Confidence:[/] [green]{verdict.get('confidence', 0):.1%}[/green] ({verdict.get('precision_tier')})")
    console.print("\n[bold]Multi-Agent Cross-Interrogation Debate Log:[/]")
    for line in res.get("multi_agent_debate_log", []):
        console.print(f"  {line}")


@app.command(name="panorama")
def panorama_cmd(
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to video (MP4/WebM) or image"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output path to save stitched panorama"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Video & Dashcam Multi-Frame Panorama Spatial Stitcher."""
    from modules.panorama_spatial_stitcher import VideoPanoramaStitcher
    stitcher = VideoPanoramaStitcher()
    res = stitcher.process_video_or_images(str(input_path), output_panorama_path=str(output) if output else None)

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold blue]🎥 Video & Multi-Frame Panorama Spatial Stitcher[/bold blue]", border_style="blue"))
    st = res.get("stitching", {})
    br = res.get("spatial_bearings", {})
    console.print(f"[bold]Input Type:[/] {'Video Clip' if res.get('is_video') else 'Image'}")
    console.print(f"[bold]Stitching Status:[/] {st.get('status')} (Estimated FOV: {st.get('estimated_fov_deg')}°)")
    console.print(f"[bold]Dominant Road Angle:[/] {br.get('dominant_road_angle_deg')}° | [bold]Sun Azimuth Offset:[/] {br.get('sun_bearing_offset_deg')}°")


@app.command(name="utility-grid")
def utility_grid_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to image with visible utility poles / wires"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Global Power Grid & Utility Transformer Hardware Classifier."""
    from modules.utility_grid_signature import UtilityGridClassifier
    classifier = UtilityGridClassifier()
    res = classifier.detect_grid_features(str(image))

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold yellow]⚡ Global Power Grid & Utility Transformer Hardware[/bold yellow]", border_style="yellow"))
    console.print(f"[bold]Top Grid Standard:[/] [cyan]{res.get('top_grid_region')}[/cyan] (conf {res.get('confidence', 0):.0%})")
    console.print(f"[bold]Transformer Archetype:[/] {res.get('transformer_archetype')}")
    console.print(f"[bold]Line Architecture:[/] {res.get('line_architecture')}")
    console.print(f"[bold]Identified Countries:[/] {', '.join(res.get('top_countries', []))}")


@app.command(name="camera-gen")
def camera_gen_cmd(
    image: Path = typer.Argument(..., exists=True, file_okay=True, readable=True, help="Path to Street View or query image"),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output raw JSON to stdout"),
):
    """Camera Generation & Street View Optical Artifact Classifier."""
    from modules.camera_generation_classifier import CameraGenerationClassifier
    classifier = CameraGenerationClassifier()
    res = classifier.analyze_optical_artifacts(str(image))

    if json_only:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(Panel.fit("[bold magenta]📷 Camera Generation & Optical Artifact Profiler[/bold magenta]", border_style="magenta"))
    console.print(f"[bold]Identified Camera Generation:[/] [cyan]{res.get('camera_generation')}[/cyan] (conf {res.get('confidence', 0):.0%})")
    console.print(f"[bold]Resolution & Sharpness:[/] {res.get('resolution')} (Laplacian variance: {res.get('sharpness_laplacian')})")
    console.print(f"[bold]Chromatic Aberration Score:[/] {res.get('chromatic_aberration_score')}")
    if res.get("hardware_notes"):
        console.print(f"[bold]Hardware Notes:[/] {' '.join(res.get('hardware_notes'))}")
    console.print(f"[bold]Geographic Priors:[/] {', '.join(res.get('jurisdiction_priors', []))}")


if __name__ == "__main__":
    app()

