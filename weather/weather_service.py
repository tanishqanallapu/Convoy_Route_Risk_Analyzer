# file: weather_service.py

import openrouteservice
import math, statistics, asyncio, aiohttp
from collections import Counter

# ---------------------------
# API KEYS (use .env later if needed)
# ---------------------------
ORS_API_KEY = os.getenv("ORS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------------------------
# Tuned parameters (more reliable)
# ---------------------------
SPACING_KM = 80               # more sampling points
GRID_DEG = 0.15               # reduce coordinate collapsing
MAX_WEATHER_POINTS = 24
MIN_VALID_POINTS = 3          # allow small number of valid points
REQ_TIMEOUT = 4.0             # OpenWeather can take 2–3s
CONN_LIMIT = 80


# ---------------------------
# Geo helpers
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat/2)**2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon/2)**2
    )
    return 2 * R * math.asin(math.sqrt(a))


def polyline_length_km(coords_lonlat):
    if len(coords_lonlat) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords_lonlat)):
        lon1, lat1 = coords_lonlat[i-1]
        lon2, lat2 = coords_lonlat[i]
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total


def sample_evenly_by_distance(coords_lonlat, target_points):
    n = len(coords_lonlat)
    if n == 0:
        return []
    if target_points >= n:
        return coords_lonlat

    # distance accumulation
    cum = [0.0]
    for i in range(1, n):
        lon1, lat1 = coords_lonlat[i-1]
        lon2, lat2 = coords_lonlat[i]
        cum.append(cum[-1] + haversine_km(lat1, lon1, lat2, lon2))

    total = cum[-1]
    if total == 0:
        return [coords_lonlat[0], coords_lonlat[-1]]

    targets = [total * i / (target_points - 1) for i in range(target_points)]

    out = []
    j = 0
    for td in targets:
        while j < len(cum) - 1 and cum[j] < td:
            j += 1
        out.append(coords_lonlat[j])

    # remove duplicates
    dedup = []
    seen = set()
    for lon, lat in out:
        key = (round(lon, 5), round(lat, 5))
        if key not in seen:
            seen.add(key)
            dedup.append([lon, lat])

    return dedup


def unique_by_grid(coords_lonlat, grid_deg=GRID_DEG):
    seen = set()
    out = []
    for lon, lat in coords_lonlat:
        key = (round(lon / grid_deg, 2), round(lat / grid_deg, 2))
        if key not in seen:
            seen.add(key)
            out.append([lon, lat])
    return out


# ---------------------------
# Async weather fetching
# ---------------------------
async def fetch_one_weather(session, lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        async with session.get(url, timeout=REQ_TIMEOUT) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "lat": lat,
                    "lon": lon,
                    "temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"].capitalize(),
                }
    except:
        return None
    return None


async def fetch_weather_fast(points_latlon, min_valid=MIN_VALID_POINTS):
    connector = aiohttp.TCPConnector(limit=CONN_LIMIT, ttl_dns_cache=300)
    results = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(fetch_one_weather(session, lat, lon))
            for (lat, lon) in points_latlon
        ]

        try:
            for coro in asyncio.as_completed(tasks):
                res = await coro
                if res:
                    results.append(res)
                    if len(results) >= min_valid:
                        break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return results


# ---------------------------
# Stats summary
# ---------------------------
def summarize_weather(weather_points):
    valid = [w for w in weather_points if w]
    if not valid:
        return None

    return {
        "avg_temp": statistics.mean(w["temp"] for w in valid),
        "avg_humidity": statistics.mean(w["humidity"] for w in valid),
        "dominant_condition": Counter(w["description"] for w in valid).most_common(1)[0][0],
    }


# ---------------------------
# Main function
# ---------------------------
def analyze_route_weather(start: str, end: str):
    client = openrouteservice.Client(key=ORS_API_KEY)

    # ---- VALIDATION ----
    if start.strip().lower() == end.strip().lower():
        raise ValueError("INVALID_SAME_PLACE")

    start_geo = client.pelias_search(text=start)
    end_geo = client.pelias_search(text=end)

    # ✅ identify exactly which place is invalid
    if not start_geo.get("features") and not end_geo.get("features"):
        raise ValueError("BOTH_NOT_FOUND")
    if not start_geo.get("features"):
        raise ValueError("START_NOT_FOUND")
    if not end_geo.get("features"):
        raise ValueError("END_NOT_FOUND")

    # ✅ FIX: extract coordinates (THIS WAS MISSING)
    start_coords = start_geo["features"][0]["geometry"]["coordinates"]  # [lon, lat]
    end_coords   = end_geo["features"][0]["geometry"]["coordinates"]    # [lon, lat]

    # ---- ROUTE ----
    route = client.directions(
        coordinates=[start_coords, end_coords],
        profile="driving-car",
        format="geojson",
    )

    coords = route["features"][0]["geometry"]["coordinates"]

    total_km = polyline_length_km(coords)
    target_points = max(8, min(MAX_WEATHER_POINTS, int(total_km / SPACING_KM) + 1))

    even_coords = sample_evenly_by_distance(coords, target_points)
    even_coords = unique_by_grid(even_coords, GRID_DEG)

    # ORS → Weather expects (lat, lon)
    latlon_points = [(lat, lon) for (lon, lat) in even_coords]
    latlon_points = [(lat, lon) for lat, lon in latlon_points if lat and lon]

    weather_points = asyncio.run(
        fetch_weather_fast(latlon_points, min_valid=MIN_VALID_POINTS)
    )

    summary = summarize_weather(weather_points)
    if not summary:
        raise RuntimeError("No weather data collected along route")

    return {
        "start": start,
        "end": end,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "route_geojson": route,
        "route_length_km": total_km,
        "sampled_points": even_coords,
        "weather_points": weather_points,
        "weather_points_used": len(weather_points),
        "summary": summary,
    }

def get_weather(city):
    # keep your existing logic
    return {"city": city, "temperature": "28°C", "status": "Sunny"}   