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
from typing import Optional
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


if __name__ == "__main__":
    app()

