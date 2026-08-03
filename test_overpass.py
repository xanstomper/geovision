import requests

query = """
[out:json][timeout:60];
(
  node["name"="Walmart"](around:200000,37.1248,-78.4946);
  way["name"="Walmart"](around:200000,37.1248,-78.4946);
  node["brand"="Walmart"](around:200000,37.1248,-78.4946);
  way["brand"="Walmart"](around:200000,37.1248,-78.4946);
);
out center;
"""
headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}
r = requests.post("https://overpass-api.de/api/interpreter", data={'data': query}, headers=headers)
data = r.json()
print("Top 10 returned by Overpass:")
for i, el in enumerate(data.get('elements', [])[:15]):
    lat = el.get('lat') or el.get('center', {}).get('lat')
    lon = el.get('lon') or el.get('center', {}).get('lon')
    print(f"{i+1}: {lat}, {lon}")
