import fs from 'fs/promises';

const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY || '';
if (!GOOGLE_MAPS_API_KEY) console.error('[geo-engine] GOOGLE_MAPS_API_KEY not set — Google API calls will fail (no demo fallback).');
const MAPBOX_API_KEY = process.env.MAPBOX_API_KEY || '';
const CACHE_DIR = './cache';

async function ensureCache() {
  try { await fs.mkdir(CACHE_DIR, { recursive: true }); } catch (e) {}
}

async function cachedFetch(url) {
  await ensureCache();
  const key = Buffer.from(url).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 64);
  const path = `${CACHE_DIR}/${key}.json`;
  try {
    const data = await fs.readFile(path, 'utf-8');
    return JSON.parse(data);
  } catch (e) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    const json = await res.json();
    await fs.writeFile(path, JSON.stringify(json));
    return json;
  }
}

export async function reverseGeocode(lat, lng) {
  const url = `https://maps.googleapis.com/maps/api/geocode/json?latlng=${lat},${lng}&key=${GOOGLE_MAPS_API_KEY}`;
  const data = await cachedFetch(url);
  if (data.status === 'OK') {
    const result = data.results[0];
    return {
      formatted_address: result.formatted_address,
      components: result.address_components,
      types: result.types,
    };
  }
  return { error: data.status, raw: data };
}

export async function searchLocation(query, coords, radius = 1000) {
  const url = `https://maps.googleapis.com/maps/api/place/textsearch/json?query=${encodeURIComponent(query)}${coords ? `&location=${coords}` : ''}&radius=${radius}&key=${GOOGLE_MAPS_API_KEY}`;
  const data = await cachedFetch(url);
  if (data.status === 'OK' || data.results) {
    return {
      results: data.results.map(r => ({
        name: r.name,
        place_id: r.place_id,
        address: r.formatted_address,
        location: r.geometry?.location,
        types: r.types,
        rating: r.rating,
      })),
      rawStatus: data.status,
    };
  }
  return { results: [], rawStatus: data.status, raw: data };
}

export async function findNearbyParks(lat, lng, walkMinutes = 10) {
  const radius = walkMinutes * 100; // approx 100m per minute walk
  const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${lat},${lng}&radius=${radius}&type=park&key=${GOOGLE_MAPS_API_KEY}`;
  const data = await cachedFetch(url);
  const parks = (data.results || []).map(r => ({
    name: r.name,
    place_id: r.place_id,
    distance_m: roughDistance(lat, lng, r.geometry?.location?.lat, r.geometry?.location?.lng),
    rating: r.rating,
  }));
  return {
    parks: parks.sort((a,b) => a.distance_m - b.distance_m).slice(0, 10),
    withinWalkMinutes: walkMinutes,
  };
}

export async function calculateWalkingDistance(startLat, startLng, endLat, endLng) {
  const url = `https://maps.googleapis.com/maps/api/directions/json?origin=${startLat},${startLng}&destination=${endLat},${endLng}&mode=walking&key=${GOOGLE_MAPS_API_KEY}`;
  const data = await cachedFetch(url);
  if (data.routes && data.routes.length > 0) {
    const route = data.routes[0];
    const leg = route.legs[0];
    return {
      distance_m: leg.distance?.value,
      distance_text: leg.distance?.text,
      duration_s: leg.duration?.value,
      duration_text: leg.duration?.text,
      steps: leg.steps.map(s => ({ instruction: s.html_instructions.replace(/<[^>]+>/g, ''), distance: s.distance.text, duration: s.duration.text })),
    };
  }
  return { error: data.status, raw: data };
}

function roughDistance(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const toRad = (v) => v * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng/2)**2;
  return Math.round(R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)));
}
