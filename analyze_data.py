import pandas as pd
from datetime import datetime

def analyze_start_time_vs_rat_period():
    """
    Analyze how many times start_time was after rat_period_end in the sorted dataset
    """
    try:
        # Read the sorted CSV file
        print("Loading dataset1_sorted.csv...")
        df = pd.read_csv('dataset1_sorted.csv')
        
        print(f"Dataset shape: {df.shape}")
        
        # Convert datetime columns to datetime format for comparison
        print("Converting datetime columns...")
        df['start_time'] = pd.to_datetime(df['start_time'], format='%d/%m/%Y %H:%M')
        df['rat_period_end'] = pd.to_datetime(df['rat_period_end'], format='%d/%m/%Y %H:%M')
        
        # Count how many times start_time is after rat_period_end
        after_rat_period = df['start_time'] > df['rat_period_end']
        count_after = after_rat_period.sum()
        
        # Print results
        print(f"\nAnalysis Results:")
        print(f"Total records: {len(df)}")
        print(f"Times start_time was after rat_period_end: {count_after}")
        print(f"Percentage: {(count_after/len(df)*100):.2f}%")
        
        return count_after
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    result = analyze_start_time_vs_rat_period()
    if result is not None:
        print(f"\nFinal answer: {result} times start_time was after rat_period_end")