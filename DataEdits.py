#when connecting between 2 datasets(dataset1_sorted, dataset2), we'll check the start_time in table 1 and join it with  record it is in in table 2 (time)
#add record "rat_period_end" we'lif __name__ == "__main__":
    # Run the merge
    # merged_df = join_datasets()
    
    # Unify datetime formats
    # uniif __name__ == "__main__":
    # Run the merge
    # merged_df = join_datasets()
    
    # Unify datetime formats
    # unify_datetime_format()
    
    # Clean habit column
    # clean_habit_column()
    
    # Create aggregated columns
    # create_aggregated_columns()
    
    # Reorder columns
    # reorder_columns()
    
    # Preview results
    # preview_merge_results()mat()
    
    # Clean habit column
#
    
    # Preview results
    # preview_merge_results()here it is in time in table 2, if it's the same record, we'll not do anything, if it is in a different record in table 2, we will also join it
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
    
    # Add table_2_time columns to track the times from dataset2
    df1['table_2_time'] = None
    df1['table_2_time_2'] = None
    
    # Function to find closest time match in dataset2 (never exceeding target time)
    def find_closest_time_match(target_time, df2_sorted):
        """Find the record in df2 with time closest to but not exceeding target_time"""
        # Filter records where time <= target_time
        valid_records = df2_sorted[df2_sorted['time'] <= target_time]
        
        if len(valid_records) == 0:
            # If no records are <= target_time, return the earliest record
            return df2_sorted.iloc[0]
        
        # Find the closest time that doesn't exceed target_time
        time_diffs = target_time - valid_records['time']
        closest_idx = time_diffs.idxmin()
        return valid_records.loc[closest_idx]
    
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
        
        # Store the time from dataset2 that was used for the match
        df1.loc[idx, 'table_2_time'] = start_match['time']
        
        # Find matching record for rat_period_end
        end_match = find_closest_time_match(end_time, df2_sorted)
        
        # Check if end_time match is different from start_time match
        if start_match.name != end_match.name:
            # Add data from end_time match with "2" suffix
            for i, col in enumerate(columns_to_add):
                df1.loc[idx, columns_to_add2[i]] = end_match[col]
            
            # Store the time from dataset2 that was used for the secondary match
            df1.loc[idx, 'table_2_time_2'] = end_match['time']
        
        # Progress indicator
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df1)} records")
    
    # Save the merged dataset
    df1.to_csv('dataset1_merged.csv', index=False)
    print(f"Merged dataset saved as 'dataset1_merged.csv' with {len(df1)} records")
    print(f"Columns in merged dataset: {list(df1.columns)}")
    
    return df1

def unify_datetime_format():
    """
    Unify the date-time format across all date columns in the merged dataset
    """
    try:
        df = pd.read_csv('dataset1_merged.csv')
        print("Original dataset loaded successfully")
        
        # Identify date/time columns
        datetime_columns = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time', 'table_2_time', 'table_2_time_2']
        
        # Standard format we want to use (no seconds)
        target_format = '%Y-%m-%d %H:%M'
        
        for col in datetime_columns:
            if col in df.columns:
                print(f"Processing column: {col}")
                
                # Convert to datetime first (handles various input formats)
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Convert to standardized string format
                df[col] = df[col].dt.strftime(target_format)
                
                print(f"  - {col}: converted to {target_format}")
        
        # Save the updated dataset
        df.to_csv('dataset1_merged.csv', index=False)
        print("\nDatetime format unification completed!")
        print(f"Updated dataset saved as 'dataset1_merged.csv'")
        
        # Show sample of updated columns
        print("\nSample of updated datetime columns:")
        for col in datetime_columns:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 'NaN'}")
        
        return df
        
    except FileNotFoundError:
        print("Merged dataset not found. Run join_datasets() first.")
        return None

def clean_habit_column():
    """
    Clean the habit column by replacing entries with number patterns with null
    """
    try:
        df = pd.read_csv('dataset1_merged.csv')
        print("Dataset loaded successfully")
        
        import re
        
        # Count original non-null values
        original_count = df['habit'].notna().sum()
        print(f"Original non-null habit values: {original_count}")
        
        # Clean habit column - replace entries with number patterns with null
        df['habit'] = df['habit'].apply(lambda x: None if pd.notna(x) and re.search(r'\d+\.\d+', str(x)) else x)
        
        # Count remaining non-null values
        cleaned_count = df['habit'].notna().sum()
        removed_count = original_count - cleaned_count
        
        print(f"Cleaned habit values (removed {removed_count} entries with number patterns)")
        print(f"Remaining non-null habit values: {cleaned_count}")
        
        # Save the updated dataset
        df.to_csv('dataset1_merged.csv', index=False)
        print("Updated dataset saved as 'dataset1_merged.csv'")
        
        # Show unique values after cleaning
        print("\nUnique habit values after cleaning:")
        unique_habits = df['habit'].unique()
        print(unique_habits)
        
        return df
        
    except FileNotFoundError:
        print("Merged dataset not found. Run join_datasets() first.")
        return None

def create_aggregated_columns():
    """
    Create new avg/sum columns and remove table_2_time columns
    """
    try:
        df = pd.read_csv('dataset1_merged.csv')
        print("Dataset loaded successfully")
        
        # Create average food_availability
        df['avg_food_availability'] = df[['food_availability', 'food_availability2']].mean(axis=1, skipna=True)
        
        # Create average bat_landing_number
        df['avg_bat_landing_number'] = df[['bat_landing_number', 'bat_landing_number2']].mean(axis=1, skipna=True)
        
        # Create average rat_arrival_number
        df['avg_rat_arrival_number'] = df[['rat_arrival_number', 'rat_arrival_number2']].mean(axis=1, skipna=True)
        
        # Create sum of rat_minutes
        df['sum_rat_minutes'] = df[['rat_minutes', 'rat_minutes2']].sum(axis=1, skipna=True)
        
        # Remove table_2_time columns
        columns_to_remove = ['table_2_time', 'table_2_time_2']
        df = df.drop(columns=columns_to_remove, errors='ignore')
        
        # Save the updated dataset
        df.to_csv('dataset1_merged.csv', index=False)
        print("Aggregated columns created and table_2_time columns removed")
        print(f"Updated dataset saved as 'dataset1_merged.csv'")
        
        # Show info about new columns
        print("\nNew aggregated columns created:")
        print("- avg_food_availability")
        print("- avg_bat_landing_number") 
        print("- avg_rat_arrival_number")
        print("- sum_rat_minutes")
        print("\nRemoved columns:")
        print("- table_2_time")
        print("- table_2_time_2")
        
        print(f"\nFinal dataset shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print("Merged dataset not found. Run join_datasets() first.")
        return None

def reorder_columns():
    """
    Reorder columns based on logical groupings, keeping month and season at the end
    """
    try:
        df = pd.read_csv('dataset1_merged.csv')
        print("Dataset loaded successfully")
        
        # Define column order based on logical groupings
        column_order = [
            # Time-related columns (main event timing)
            'start_time',
            'rat_period_start', 
            'rat_period_end',
            'sum_rat_minutes',
            'avg_rat_arrival_number',
            'avg_bat_landing_number',
            'avg_food_availability',
            'risk',
            'reward',

            # Bat behavior columns
            'habit',
            
            
            # Rat-related columns
            'seconds_after_rat_arrival',
            # Original dataset2 columns (for reference)
            'bat_landing_to_food',
            'bat_landing_number',
            'food_availability', 
            'rat_arrival_number',
            'bat_landing_number2',
            'food_availability2',
            'rat_minutes',
            'rat_minutes2', 
            'rat_arrival_number2',
            
            # Temporal context (at the end)
            'month',
            'season',
            'sunset_time',
            'hours_after_sunset',
        ]
        
        # Only include columns that actually exist in the dataframe
        available_columns = [col for col in column_order if col in df.columns]
        
        # Add any remaining columns that weren't in our predefined order
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_column_order = available_columns + remaining_columns
        
        # Reorder the dataframe
        df = df[final_column_order]
        
        # Save the reordered dataset
        df.to_csv('dataset1_merged.csv', index=False)
        print("Columns reordered successfully")
        print(f"Updated dataset saved as 'dataset1_merged.csv'")
        
        print("\nFinal column order:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        return df
        
    except FileNotFoundError:
        print("Merged dataset not found. Run previous steps first.")
        return None

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
def get_unique_habit_values(csv_path):
    """
    Returns a list of all unique values in the 'habit' column.
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        list: List of unique habit values
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Get unique values from the habit column, excluding NaN values
    unique_habits = df['habit'].dropna().unique().tolist()
    
    return unique_habits

if __name__ == "__main__":
    # Run the merge
    # merged_df = join_datasets()
    
    # Unify datetime formats
    # unify_datetime_format()
    
    # Clean habit column
    # clean_habit_column()
    
    # Create aggregated columns
    # create_aggregated_columns()
    reorder_columns()
    # Preview results
    # preview_merge_results()