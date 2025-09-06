import pandas as pd

# 1. Load the dataset
df = pd.read_csv("dataset1_merged.CSV")

# 2. Peek at the first 5 rows
print("===== HEAD =====")
print(df.head())

# 3. Info about columns (types + non-null counts)
print("\n===== INFO =====")
print(df.info())

# 4. Basic statistics (numeric columns only)
print("\n===== DESCRIBE =====")
print(df.describe())

# 5. Missing values count per column
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 6. Count rat_arrival_number = 0
print("\n===== RAT ARRIVAL NUMBER = 0 COUNT =====")
print((df['rat_arrival_number'] == 0).sum())

# 7. Count rat_period_start > start_time
print("\n===== RAT PERIOD START > START TIME COUNT =====")
print((df['rat_period_start'] > df['start_time']).sum())

print("\n===== RAT PERIOD START < START TIME COUNT =====")
print((df['rat_period_start'] < df['start_time']).sum())

print("\n===== FIRST RECORD WHERE RAT PERIOD START < START TIME =====")
print(df[df['rat_period_start'] < df['start_time']].head(1))
print("\n===== FIRST RECORD WHERE RAT PERIOD START > START TIME =====")
print(df[df['rat_period_start'] < df['start_time']].head(1))

print("\n===== UNIQUE VALUES IN HABIT =====")
print(df['habit'].unique())

# Clean habit column - replace entries with number patterns with null
import re
df['habit'] = df['habit'].apply(lambda x: None if pd.notna(x) and re.search(r'\d+\.\d+', str(x)) else x)

print("\n===== UNIQUE VALUES IN HABIT (AFTER CLEANING) =====")
print(df['habit'].unique())
