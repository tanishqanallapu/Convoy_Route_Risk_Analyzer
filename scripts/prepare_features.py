import pandas as pd

print("📦 Loading Hyderabad terrorism dataset...")
df = pd.read_csv("data/terrorism_train.csv")

# Combine useful text columns
df["text"] = (
    df["summary"].fillna("") + " | " +
    df["attacktype1_txt"].fillna("") + " | " +
    df["weaptype1_txt"].fillna("") + " | " +
    df["city"].fillna("") + " | " +
    df["provstate"].fillna("")
)

# Define label: risk severity
# (You can adjust thresholds later)
def risk_label(row):
    score = (row["nkill"] * 2) + row["nwound"]
    if score == 0:
        return 0
    elif score < 5:
        return 1
    else:
        return 2

df["label"] = df.apply(risk_label, axis=1)

# Save preprocessed file
output_path = "data/terrorism_text_dataset.csv"
df[["text", "label"]].to_csv(output_path, index=False)

print(f"✅ Saved {len(df)} processed rows to {output_path}")
print(df[["text", "label"]].head())
