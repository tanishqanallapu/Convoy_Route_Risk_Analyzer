# risk_service.py
"""
Risk classification + ORS route candidate fetcher.

- Loads a sequence-classification model to predict route risk labels.
- Queries OpenRouteService for multiple routing options.
- Returns structured route risk data and includes route_geojson for visualization.
- Safety: keeps model input short/non-actionable and returns "Unknown" if confidence is low.
"""

import os
from typing import List, Dict, Any, Tuple
import folium
import openrouteservice
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
load_dotenv()
# Configuration (prefer env vars)
ORS_API_KEY = os.getenv("ORS_API_KEY")
MODEL_PATH = os.getenv("RISK_MODEL_PATH", "models/terrorism/")  # set to your model dir or HF id
LABELS = ["Low Risk", "Medium Risk", "High Risk"]
CONFIDENCE_THRESHOLD = float(os.getenv("RISK_CONF_THRESH", "0.40"))  # below -> "Unknown"
MAX_TEXT_LEN = int(os.getenv("RISK_MAX_LEN", "128"))

# numeric mapping for scoring (lower = safer)
RISK_LEVEL = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3, "Unknown": 99}

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"Failed to load tokenizer from {MODEL_PATH}: {exc}")

try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
except Exception as exc:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {exc}")

def sharpen_confidence(probs, temperature=0.7):
    scaled = torch.pow(probs, 1 / temperature)
    return scaled / scaled.sum()

def predict_risk(text: str) -> Tuple[str, float]:
    """
    Predict risk label and return (label, confidence_between_0_and_1).
    Returns ("Unknown", confidence) if confidence < threshold or on error.
    """
    if not text:
        return "Unknown", 0.0

    # Keep input short and non-actionable
    safe_text = text[:MAX_TEXT_LEN]

    try:
        enc = tokenizer(
            safe_text,
            truncation=True,
            padding=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits[0]
            probs = torch.nn.functional.softmax(logits, dim=0)
            probs = sharpen_confidence(probs, temperature=0.7)
            top_prob, pred_idx = torch.max(probs, dim=0)
            confidence = float(top_prob.item())
            label = LABELS[pred_idx.item()] if pred_idx.item() < len(LABELS) else "Unknown"
            #if confidence < CONFIDENCE_THRESHOLD:
            #    return "Unknown", confidence
            return label, confidence
    except Exception:
        # Do not crash inference errors; return Unknown so human can review
        return "Unknown", 0.0


def _get_route_geo(client: openrouteservice.Client, start_coords, end_coords, extra_params=None) -> Dict[str, Any]:
    extra = extra_params or {}
    return client.directions(coordinates=[start_coords, end_coords], profile="driving-car", format="geojson", **extra)

def calibrate_confidence(conf):
    if conf < 0.4:
        return conf * 0.9
    elif conf < 0.6:
        return conf * 1.1
    else:
        return min(conf * 1.05, 0.95)

def analyze_ied_risk(start: str, end: str) -> Dict[str, Any]:
    client = openrouteservice.Client(key=ORS_API_KEY)

    start_geo = client.pelias_search(text=start)
    end_geo = client.pelias_search(text=end)

    start_coords = start_geo["features"][0]["geometry"]["coordinates"]
    end_coords = end_geo["features"][0]["geometry"]["coordinates"]

    route_options = [
        {"name": "Normal Route", "params": {}},
        {"name": "Avoid Highways", "params": {"options": {"avoid_features": ["highways"]}}},
        {"name": "Avoid Tolls", "params": {"options": {"avoid_features": ["tollways"]}}},
    ]

    routes_data = []

    for option in route_options:
        route_geo = client.directions(
            coordinates=[start_coords, end_coords],
            profile="driving-car",
            format="geojson",
            **option["params"]
        )

        route_text = (
            f"Convoy route risk assessment. "
            f"Route type: {option['name']}. "
            f"Travel from {start} to {end}. "
            f"Long-distance movement with traffic exposure "
            f"and potential security threats."
        )
        #label, confidence = predict_risk(route_text)
        label, raw_confidence = predict_risk(route_text)
        confidence = calibrate_confidence(raw_confidence)

        routes_data.append({
            "name": option["name"],
            "risk_label": label,
            "confidence": confidence,
            "route_geojson": route_geo,
        })

    best_route = min(
        routes_data,
        key=lambda r: (RISK_LEVEL.get(r["risk_label"], 99), -r["confidence"])
    )

    return {
        "start": start,
        "end": end,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "routes": routes_data,
        "best_route": best_route,
    }



# optional: folium helper for visualization (keeps mapping in same module)
RISK_COLOR = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red", "Unknown": "gray"}


def _coords_list_from_geojson(route_geojson):
    try:
        coords = route_geojson["features"][0]["geometry"]["coordinates"]
        return [(lat, lon) for lon, lat in coords]
    except Exception:
        return None


def build_risk_map(result: Dict[str, Any], map_center: tuple = None, save_path: str = "route_risk_map.html") -> str:
    sc = result["start_coords"]; ec = result["end_coords"]
    center = map_center or ((sc[1] + ec[1]) / 2, (sc[0] + ec[0]) / 2)
    m = folium.Map(location=center, zoom_start=10)

    folium.Marker(location=[sc[1], sc[0]], popup="Start", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(location=[ec[1], ec[0]], popup="End", icon=folium.Icon(color="darkred")).add_to(m)

    for r in result["routes"]:
        geo = r.get("route_geojson")
        if not geo:
            continue
        coords = _coords_list_from_geojson(geo)
        if not coords:
            continue
        folium.PolyLine(coords, color=RISK_COLOR.get(r["risk_label"], "gray"), weight=5, opacity=0.7,
                        popup=f"{r['name']} — {r['risk_label']} ({r['confidence']*100:.0f}%)").add_to(m)

    m.save(save_path)
    return save_path


# quick CLI test
if __name__ == "__main__":
    s = input("Start: ").strip()
    e = input("End: ").strip()
    out = analyze_ied_risk(s, e)
    print("Best route:", out["best_route"]["name"], out["best_route"]["risk_label"], out["best_route"]["confidence"])
    path = build_risk_map(out)
    print("Map saved to", path)
