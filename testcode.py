import pandas as pd

# 1. Load the dataset
df = pd.read_csv("DATASET2.CSV")

# Parse datetime columns
for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
    df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M", errors="coerce")

# Check conversion worked
print(df[["start_time", "rat_period_start", "rat_period_end", "sunset_time"]].head())