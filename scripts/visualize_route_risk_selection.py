import folium
import openrouteservice
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import webbrowser

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjAyNWUzZWFhYjQyMjQzOGViNTliZjNjNTAxN2Y3YTJjIiwiaCI6Im11cm11cjY0In0="  # Replace with your OpenRouteService API key
MODEL_PATH = "models/terrorism/"
LABELS = ["Low Risk", "Medium Risk", "High Risk"]

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ---------------------------------------------------------------------
# RISK PREDICTION
# ---------------------------------------------------------------------
def predict_risk(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred_label = torch.argmax(probs).item()
    return LABELS[pred_label], round(probs[pred_label].item() * 100, 2)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
start = input("Enter Start Location: ").strip()
end = input("Enter End Location: ").strip()

client = openrouteservice.Client(key=ORS_API_KEY)

# ✅ Geocode both locations
try:
    start_geo = client.pelias_search(text=start)
    end_geo = client.pelias_search(text=end)
    start_coords = start_geo["features"][0]["geometry"]["coordinates"]  # [lon, lat]
    end_coords = end_geo["features"][0]["geometry"]["coordinates"]      # [lon, lat]
except Exception as e:
    print(f"❌ Error geocoding locations: {e}")
    exit()

print(f"📍 Start: {start_coords}, End: {end_coords}")

# ✅ Get multiple route options
route_options = [
    {"name": "Normal Route", "params": {}},
    {"name": "Avoid Highways", "params": {"options": {"avoid_features": ["highways"]}}},
    {"name": "Avoid Tolls", "params": {"options": {"avoid_features": ["tollways"]}}},
]

routes = []
for option in route_options:
    try:
        route = client.directions(
            coordinates=[start_coords, end_coords],
            profile="driving-car",
            format="geojson",
            **option["params"]
        )
        routes.append((option["name"], route))
    except Exception as e:
        print(f"⚠️ Could not fetch {option['name']} route: {e}")

if not routes:
    print("❌ No valid routes found.")
    exit()

# ---------------------------------------------------------------------
# RISK PREDICTION FOR EACH ROUTE
# ---------------------------------------------------------------------
RISK_LEVEL = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3}
route_risks = []

for name, route in routes:
    route_text = f"Traveling from {start} to {end} via {name} in Hyderabad region."
    predicted_risk, confidence = predict_risk(route_text)
    route_risks.append((name, route, predicted_risk, confidence))
    print(f"🛣️ {name}: {predicted_risk} ({confidence}%)")

# ✅ Find safest route
best_route = min(route_risks, key=lambda x: (RISK_LEVEL[x[2]], -x[3]))
best_name, best_data, best_risk, best_conf = best_route
print(f"\n✅ Safest Route: {best_name} → {best_risk} ({best_conf}%)")

# ---------------------------------------------------------------------
# MAP CREATION
# ---------------------------------------------------------------------
midpoint = [
    (start_coords[1] + end_coords[1]) / 2,
    (start_coords[0] + end_coords[0]) / 2
]
m = folium.Map(location=midpoint, zoom_start=13)

# Add each route with color-coded risk
for name, route, risk, conf in route_risks:
    color = "green" if risk == "Low Risk" else "orange" if risk == "Medium Risk" else "red"
    weight = 6 if name == best_name else 3

    folium.GeoJson(
        route,
        name=f"{name} ({risk}, {conf}%)",
        style_function=lambda x, color=color, weight=weight: {
            "color": color,
            "weight": weight,
            "opacity": 0.8
        }
    ).add_to(m)

# Add start and end markers
folium.Marker(
    location=list(reversed(start_coords)),
    popup=f"Start: {start}",
    icon=folium.Icon(color="blue")
).add_to(m)

folium.Marker(
    location=list(reversed(end_coords)),
    popup=f"End: {end}",
    icon=folium.Icon(color="blue")
).add_to(m)

# ---------------------------------------------------------------------
# RISK INFO PANEL
# ---------------------------------------------------------------------
from folium import IFrame, Popup

html = f"""
<div style='font-size:16px; font-family:Arial; padding:10px;'>
    <h4>🚗 Route Risk Analysis</h4>
    <b>Start:</b> {start}<br>
    <b>End:</b> {end}<br>
    <hr>
    <b>Safest Route:</b> {best_name}<br>
    <b>Predicted Risk:</b> {best_risk}<br>
    <b>Confidence:</b> {best_conf}%<br>
</div>
"""
iframe = IFrame(html, width=250, height=160)
popup = Popup(iframe, max_width=2650)

folium.Marker(
    location=midpoint,
    popup=popup,
    icon=folium.Icon(color="darkgreen")
).add_to(m)

# ---------------------------------------------------------------------
# SAVE AND OPEN
# ---------------------------------------------------------------------
m.save("route_risk_alternate_map.html")
webbrowser.open("route_risk_alternate_map.html")

print("✅ Route map generated and opened in browser!")
