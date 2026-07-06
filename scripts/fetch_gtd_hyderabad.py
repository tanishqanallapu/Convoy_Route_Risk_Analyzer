import pandas as pd
import os

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_FILE = "data/globalterrorismdb_0718dist.csv"  # your Kaggle GTD CSV
OUTPUT_FILE = "data/terrorism_india_ied.csv"       # output file

# ---------------------------------------------------------------------
# LOAD THE GTD DATASET
# ---------------------------------------------------------------------
print("📄 Loading GTD dataset...")
if not os.path.exists(INPUT_FILE):
    raise SystemExit(f"❌ File not found: {INPUT_FILE}\nPlease download the GTD CSV first.")

df = pd.read_csv(INPUT_FILE, encoding="ISO-8859-1", low_memory=False)
print(f"✅ Loaded {len(df)} total rows.")

# ---------------------------------------------------------------------
# FILTER FOR EXPLOSIVES / IED ATTACKS
# ---------------------------------------------------------------------
ied_df = df[df["weaptype1_txt"].str.contains("Explosives", case=False, na=False)]
print(f"💣 Found {len(ied_df)} explosive-related events worldwide.")

# ---------------------------------------------------------------------
# FILTER FOR INDIA
# ---------------------------------------------------------------------
india_df = ied_df[ied_df["country_txt"].str.contains("India", case=False, na=False)]
print(f"🇮🇳 Found {len(india_df)} explosive-related incidents in India.")

if india_df.empty:
    raise SystemExit("⚠️ No India-related incidents found. Check dataset year range or country column.")

# ---------------------------------------------------------------------
# SELECT RELEVANT COLUMNS
# ---------------------------------------------------------------------
cols = [
    "iyear", "imonth", "iday", "country_txt", "region_txt",
    "provstate", "city", "latitude", "longitude",
    "attacktype1_txt", "weaptype1_txt", "nkill", "nwound", "summary"
]
india_df = india_df[cols]

# ---------------------------------------------------------------------
# SAVE CLEANED DATASET
# ---------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
india_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(india_df)} India IED incidents to {OUTPUT_FILE}")

print("\n🔹 Columns:", list(india_df.columns))
print("🔹 Sample:")
print(india_df.head(5))
