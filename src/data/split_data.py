import pandas as pd
from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DATA_PATH = Path("data/processed")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(RAW_DATA_PATH)

# Sort by time (VERY IMPORTANT)
df = df.sort_values("Time").reset_index(drop=True)

# 40% baseline, 60% stream
split_index = int(0.4 * len(df))

baseline_df = df.iloc[:split_index]
stream_df = df.iloc[split_index:]

# Save
baseline_df.to_csv(PROCESSED_DATA_PATH / "baseline.csv", index=False)
stream_df.to_csv(PROCESSED_DATA_PATH / "stream.csv", index=False)

print("✅ Step 3 completed:")
print(f"Baseline size: {baseline_df.shape}")
print(f"Stream size: {stream_df.shape}")
