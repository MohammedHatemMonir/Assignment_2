import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('dataset1_merged.csv')

print("Original dataset shape:", df.shape)
print("\nFirst few rows before adding rat_encounter column:")
print(df[['rat_arrival_number', 'bat_landing_number']].head())

# Add the rat_encounter column
# rat_encounter = rat_arrival_number * bat_landing_number
df['rat_encounter'] = df['rat_arrival_number'] * df['bat_landing_number']

print("\nFirst few rows after adding rat_encounter column:")
print(df[['rat_arrival_number', 'bat_landing_number', 'rat_encounter']].head(10))

# Check for any missing values in the calculation
print(f"\nMissing values in rat_arrival_number: {df['rat_arrival_number'].isna().sum()}")
print(f"Missing values in bat_landing_number: {df['bat_landing_number'].isna().sum()}")
print(f"Missing values in rat_encounter: {df['rat_encounter'].isna().sum()}")

# Basic statistics for the new column
print(f"\nRat Encounter Statistics:")
print(f"Mean: {df['rat_encounter'].mean():.2f}")
print(f"Median: {df['rat_encounter'].median():.2f}")
print(f"Min: {df['rat_encounter'].min():.2f}")
print(f"Max: {df['rat_encounter'].max():.2f}")
print(f"Standard Deviation: {df['rat_encounter'].std():.2f}")

# Save the updated dataset back to the original file
df.to_csv('dataset1_merged.csv', index=False)
print(f"\nUpdated dataset saved back to 'dataset1_merged.csv'")
print(f"Updated dataset shape: {df.shape}")

# Display sample of the new column values
print(f"\nSample of rat_encounter values:")
print(df['rat_encounter'].value_counts().head(10))