# GeoVision Example Prompts

## Geo-Spy Sessions

> "Use geovision to analyze this image. Write a full geoguess report: image clues, OCR text, building style, vegetation, map search results, nearby parks within 8-10 min walk, and walking distances. Finally open Google Maps to the top candidate and take a screenshot."

> "Image 1: fire escape brick building. Image 2: Google Maps showing Toronto region 43.62,-79.75. Use geovision with nearPark=true and region=Ontario. Search for exact address, then verify walking distance to nearest park."

## Computer Control

> "Take a screenshot of current screen. Open Google Maps at these coordinates. Move mouse to (x,y) and click. Search for 'Starbucks' in the Google Maps search bar, press enter, take another screenshot, and tell me what you see."

## MCP Integration Examples (Claude Desktop / Cursor / Windsurf config)

```json
{
  "mcpServers": {
    "geovision": {
      "command": "node",
      "args": ["/home/jewboy420/geovision/src/mcp-server.js"],
      "env": {
        "GOOGLE_MAPS_API_KEY": "sk-your-key-here"
      }
    }
  }
}
```

```json
{
  "mcpServers": {
    "geovision-python": {
      "command": "python3",
      "args": ["/home/jewboy420/geovision/python-mcp-server.py"],
      "env": {
        "GOOGLE_MAPS_API_KEY": "sk-your-key-here"
      }
    }
  }
}
```

## Standalone Usage

```bash
# Analyze an image from CLI
python3 /home/jewboy420/geovision/geovision.py /path/to/image.jpg

# Search a location
python3 /home/jewboy420/geovision/geo_search.py search '{"query": "brick apartment Mississauga"}'

# Find parks near coordinates
python3 /home/jewboy420/geovision/geo_search.py parks '{"lat":43.6228857,"lng":-79.7555258,"walk_minutes":10}'

# Check walking distance
python3 /home/jewboy420/geovision/geo_search.py walk '{"slat":43.62,"slng":-79.75,"elat":43.63,"elng":-79.76}'
```
