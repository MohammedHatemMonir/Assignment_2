import pandas as pd
from datetime import datetime

def sort_dataset1():
    """
    Load dataset1.csv, sort it by start_time column, and save as dataset1_sorted.csv
    """
    try:
        # Read the CSV file
        print("Loading dataset1.csv...")
        df = pd.read_csv('dataset1.csv')
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Convert start_time to datetime for proper sorting
        print("Converting start_time to datetime format...")
        df['start_time'] = pd.to_datetime(df['start_time'], format='%d/%m/%Y %H:%M')
        
        # Sort by start_time column
        print("Sorting dataset by start_time...")
        df_sorted = df.sort_values('start_time').reset_index(drop=True)
        
        # Convert start_time back to original string format for saving
        df_sorted['start_time'] = df_sorted['start_time'].dt.strftime('%d/%m/%Y %H:%M')
        
        # Save sorted dataset
        output_file = 'dataset1_sorted.csv'
        df_sorted.to_csv(output_file, index=False)
        
        print(f"Sorted dataset saved as: {output_file}")
        print(f"Sorted dataset shape: {df_sorted.shape}")
        
        # Display first few rows of sorted dataset
        print("\nFirst 5 rows of sorted dataset:")
        print(df_sorted.head())
        
        # Display date range
        first_date = df_sorted['start_time'].iloc[0]
        last_date = df_sorted['start_time'].iloc[-1]
        print(f"\nDate range: {first_date} to {last_date}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = sort_dataset1()
    if success:
        print("\nDataset sorting completed successfully!")
    else:
        print("\nDataset sorting failed!")
