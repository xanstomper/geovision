#!/usr/bin/env python3
"""
GeoVision CLI - Next Generation Geolocation AI Tool
Provides a rich, interactive, and beautiful terminal interface for the GeoVision pipeline.
"""

import sys
import os
import json
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

try:
    from geovision_deep_scan import run_pipeline, DEFAULT_OUTPUT_DIR
except ImportError as e:
    rprint(f"[red]Error importing GeoVision core modules: {e}[/red]")
    sys.exit(1)

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

    if json_only:
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

if __name__ == "__main__":
    app()
