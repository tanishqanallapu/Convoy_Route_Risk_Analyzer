import openrouteservice
import requests
import folium
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import math

# ---------------------------------------------------------------------
# CONFIGURATION (tune these)
# ---------------------------------------------------------------------
ORS_API_KEY = os.getenv("ORS_API_KEY") # replace with your key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # replace with your key

SAMPLE_INTERVAL = 130                 # take every Nth route coordinate (higher = fewer calls)
MAX_THREADS = 10                      # parallel requests to OpenWeather
REQUEST_TIMEOUT = 5                   # seconds per HTTP request
MIN_KM_BETWEEN_SAMPLES_KM = 5.0       # skip points that are too close (<2 km). Set to 0 to disable.

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def get_weather_by_coords(lat, lon):
    """Fetch live weather data for given coordinates from OpenWeatherMap API."""
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return {
                "lat": lat,
                "lon": lon,
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"].capitalize(),
            }
    except Exception:
        # Swallow and treat as a failed point; we’ll still summarize with what we have
        return None
    return None


def sample_route_coords(route_coords, interval=30):
    """Down-sample coordinates to avoid excessive API calls."""
    return route_coords[::interval]


def get_route(client, start_coords, end_coords):
    """Fetch route between source and destination using OpenRouteService."""
    try:
        route = client.directions(
            coordinates=[start_coords, end_coords],
            profile="driving-car",
            format="geojson"
        )
        coords = route["features"][0]["geometry"]["coordinates"]
        return coords, route
    except Exception as e:
        print(f"❌ Error fetching route: {e}")
        return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def thin_by_distance(sampled_coords, min_km=2.0):
    """Keep only coordinates that are at least min_km apart."""
    if min_km <= 0 or not sampled_coords:
        return sampled_coords

    filtered = []
    last_lat, last_lon = None, None
    for lon, lat in sampled_coords:
        if last_lat is None:
            filtered.append((lon, lat))
            last_lat, last_lon = lat, lon
            continue
        if haversine_km(lat, lon, last_lat, last_lon) >= min_km:
            filtered.append((lon, lat))
            last_lat, last_lon = lat, lon
    return filtered


def summarize_weather(weather_points):
    """Compute average temperature, humidity, and dominant condition."""
    valid = [w for w in weather_points if w is not None]
    if not valid:
        return None

    avg_temp = statistics.mean([w["temp"] for w in valid])
    avg_humidity = statistics.mean([w["humidity"] for w in valid])
    dominant_condition = Counter([w["description"] for w in valid]).most_common(1)[0][0]

    return {
        "avg_temp": avg_temp,
        "avg_humidity": avg_humidity,
        "dominant_condition": dominant_condition
    }


def create_weather_map(route, summary, start, end, midpoint):
    """Create and display Folium map with route and overall weather summary."""
    m = folium.Map(location=midpoint, zoom_start=10)

    # Draw route line
    folium.GeoJson(route, name="Route", style_function=lambda x: {
        "color": "blue",
        "weight": 4,
        "opacity": 0.8
    }).add_to(m)

    # Add start and end markers
    folium.Marker(
        location=list(reversed(route["features"][0]["geometry"]["coordinates"][0])),
        popup=f"🚩 Start: {start}",
        icon=folium.Icon(color="green")
    ).add_to(m)
    folium.Marker(
        location=list(reversed(route["features"][0]["geometry"]["coordinates"][-1])),
        popup=f"🏁 End: {end}",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # Weather summary popup
    popup_html = f"""
    <div style='font-size:14px; font-family:Arial;'>
        <h4>🌤 Route Weather Summary</h4>
        🌡 Average Temp: {summary['avg_temp']:.2f}°C<br>
        💧 Avg Humidity: {summary['avg_humidity']:.2f}%<br>
        🌦 Dominant: {summary['dominant_condition']}
    </div>
    """
    folium.Marker(
        location=midpoint,
        popup=popup_html,
        icon=folium.Icon(
            color="green" if "clear" in summary["dominant_condition"].lower() else
                  "orange" if "cloud" in summary["dominant_condition"].lower() else "red",
            icon="cloud"
        )
    ).add_to(m)

    m.save("route_live_weather.html")
    webbrowser.open("route_live_weather.html")
    print("✅ Map saved as route_live_weather.html and opened in browser!")


# ---------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start = input("Enter Start Location: ").strip()
    end = input("Enter End Location: ").strip()

    client = openrouteservice.Client(key=ORS_API_KEY)

    try:
        start_geo = client.pelias_search(text=start)
        end_geo = client.pelias_search(text=end)
        start_coords = start_geo["features"][0]["geometry"]["coordinates"]
        end_coords = end_geo["features"][0]["geometry"]["coordinates"]
    except Exception as e:
        print(f"❌ Could not find one or both locations: {e}")
        raise SystemExit(1)

    coords, route = get_route(client, start_coords, end_coords)
    if not coords:
        raise SystemExit(1)

    # 1) Down-sample by index
    sampled_coords = sample_route_coords(coords, SAMPLE_INTERVAL)
    # 2) Optionally thin by real distance (km)
    sampled_coords = thin_by_distance(sampled_coords, MIN_KM_BETWEEN_SAMPLES_KM)

    print(f"🌍 Weather requests to make: {len(sampled_coords)} (in parallel with {MAX_THREADS} threads)")

    # Fetch weather in parallel
    weather_points = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(get_weather_by_coords, lat, lon) for lon, lat in sampled_coords]
        for f in as_completed(futures):
            res = f.result()
            if res:
                weather_points.append(res)

    summary = summarize_weather(weather_points)
    if not summary:
        print("⚠️ No weather data collected along route.")
        raise SystemExit(1)

    print("\n🌤 OVERALL ROUTE WEATHER SUMMARY")
    print(f"🌡 Avg Temp: {summary['avg_temp']:.2f}°C")
    print(f"💧 Avg Humidity: {summary['avg_humidity']:.2f}%")
    print(f"🌦 Dominant Condition: {summary['dominant_condition']}")

    midpoint = [
        (start_coords[1] + end_coords[1]) / 2,
        (start_coords[0] + end_coords[0]) / 2
    ]

    create_weather_map(route, summary, start, end, midpoint)
