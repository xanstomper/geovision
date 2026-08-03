import requests

query = """
[out:json][timeout:60];
(
  node["brand"="Walmart"](around:200000,37.1248,-78.4946);
  way["brand"="Walmart"](around:200000,37.1248,-78.4946);
);
out center;
"""
headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}
r = requests.post("https://overpass-api.de/api/interpreter", data={'data': query}, headers=headers)
if r.status_code != 200:
    print(r.status_code, r.text)
else:
    data = r.json()
    print(f"Total found with brand=Walmart: {len(data.get('elements', []))}")
