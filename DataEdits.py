#when connecting between 2 datasets(dataset1_sorted, dataset2), we'll check the start_time in table 1 and join it with  record it is in in table 2 (time)
#add record "rat_period_end" we'll check where it is in time in table 2, if it's the same record, we'll not do anything, if it is in a different record in table 2, we will also join it
#every column we take from table 2 we'll add it as a column in table 1 for each of these columns: bat_landing_number,food_availability,rat_minutes,rat_arrival_number as a new column in table 1. and a duplicate record for them in case we take 2 record:  bat_landing_number2,food_availability2,rat_minutes2,rat_arrival_number2
#example table 1 after joining the 2 datasets: start_time,bat_landing_to_food,habit,rat_period_start,rat_period_end,seconds_after_rat_arrival,risk,reward,month,sunset_time,hours_after_sunset,season,bat_landing_number,food_availability,rat_minutes,rat_arrival_number,bat_landing_number2,food_availability2,rat_minutes2,rat_arrival_number2
import pandas as pd
from datetime import datetime

def join_datasets():
    """
    Join dataset1_sorted.csv with dataset2.csv based on time matching
    """
    # Load both datasets
    df1 = pd.read_csv('dataset1_sorted.csv')
    df2 = pd.read_csv('dataset2.csv')
    
    # Convert time columns to datetime for proper comparison
    df1['start_time'] = pd.to_datetime(df1['start_time'], format='%d/%m/%Y %H:%M')
    df1['rat_period_end'] = pd.to_datetime(df1['rat_period_end'], format='%d/%m/%Y %H:%M')
    df2['time'] = pd.to_datetime(df2['time'], format='%d/%m/%Y %H:%M')
    
    # Initialize new columns for dataset2 data
    columns_to_add = ['bat_landing_number', 'food_availability', 'rat_minutes', 'rat_arrival_number']
    columns_to_add2 = ['bat_landing_number2', 'food_availability2', 'rat_minutes2', 'rat_arrival_number2']
    
    # Initialize all new columns with NaN
    for col in columns_to_add + columns_to_add2:
        df1[col] = None
    
    # Function to find closest time match in dataset2
    def find_closest_time_match(target_time, df2_sorted):
        """Find the record in df2 with time closest to target_time"""
        time_diffs = abs(df2_sorted['time'] - target_time)
        closest_idx = time_diffs.idxmin()
        return df2_sorted.loc[closest_idx]
    
    # Sort df2 by time for efficient searching
    df2_sorted = df2.sort_values('time').reset_index(drop=True)
    
    # Process each row in dataset1
    for idx, row in df1.iterrows():
        start_time = row['start_time']
        end_time = row['rat_period_end']
        
        # Find matching record for start_time
        start_match = find_closest_time_match(start_time, df2_sorted)
        
        # Add data from start_time match
        for col in columns_to_add:
            df1.loc[idx, col] = start_match[col]
        
        # Find matching record for rat_period_end
        end_match = find_closest_time_match(end_time, df2_sorted)
        
        # Check if end_time match is different from start_time match
        if start_match.name != end_match.name:
            # Add data from end_time match with "2" suffix
            for i, col in enumerate(columns_to_add):
                df1.loc[idx, columns_to_add2[i]] = end_match[col]
        
        # Progress indicator
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df1)} records")
    
    # Save the merged dataset
    df1.to_csv('dataset1_merged.csv', index=False)
    print(f"Merged dataset saved as 'dataset1_merged.csv' with {len(df1)} records")
    print(f"Columns in merged dataset: {list(df1.columns)}")
    
    return df1

def preview_merge_results():
    """
    Preview the first few rows of the merged dataset
    """
    try:
        df = pd.read_csv('dataset1_merged.csv')
        print("First 5 rows of merged dataset:")
        print(df.head())
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Check for records where secondary columns were filled
        secondary_filled = df[df['bat_landing_number2'].notna()]
        print(f"\nRecords with secondary data (different end time): {len(secondary_filled)}")
        
    except FileNotFoundError:
        print("Merged dataset not found. Run join_datasets() first.")

if __name__ == "__main__":
    # Run the merge
    merged_df = join_datasets()
    
    # Preview results
    preview_merge_results()