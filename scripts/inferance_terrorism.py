from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
MODEL_PATH = "models/terrorism/"  # path to your trained model
LABELS = ["Low Risk", "Medium Risk", "High Risk"]

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------
print("📦 Loading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ---------------------------------------------------------------------
# FUNCTION TO PREDICT RISK
# ---------------------------------------------------------------------
def predict_risk(text: str):
    """Predict terrorism/IED risk level for a given text description."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred_label = torch.argmax(probs).item()
    return {
        "input_text": text,
        "predicted_label": LABELS[pred_label],
        "confidence": round(probs[pred_label].item() * 100, 2),
        "all_probs": {LABELS[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    }

# ---------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚗 Convoy Risk Analyzer")
    print("Enter start and end locations to estimate route risk.\n")

    start = input("Enter Start Location: ").strip()
    end = input("Enter End Location: ").strip()

    # Create a simple natural-language route description
    route_text = f"Traveling from {start} to {end} through Hyderabad region."
    print(f"\n🗺️ Route Description: {route_text}")

    # Predict risk
    result = predict_risk(route_text)

    # Display results
    print("\n🚀 Route Risk Prediction")
    print(f"Predicted Risk Level: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Probabilities: {result['all_probs']}")
