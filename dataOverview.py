import pandas as pd

# 1. Load the dataset
df = pd.read_csv("DATASET2.CSV")

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
