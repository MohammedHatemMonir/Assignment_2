import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_habit_risk_analysis():
    """
    Create a CSV file with habit analysis showing risk and no_risk counts for each habit type
    """
    # Load the merged dataset
    df = pd.read_csv('dataset1_merged.csv')
    
    print("Dataset loaded successfully")
    print(f"Total records: {len(df)}")
    
    # Remove rows where habit is null or NaN
    df_clean = df[df['habit'].notna()]
    print(f"Records with habit data: {len(df_clean)}")
    
    # Create the analysis - group by habit and count risk vs no_risk
    habit_analysis = df_clean.groupby('habit')['risk'].agg(['sum', 'count']).reset_index()
    
    # Rename columns appropriately
    habit_analysis.columns = ['habit', 'risk', 'total']
    
    # Calculate no_risk as total - risk
    habit_analysis['no_risk'] = habit_analysis['total'] - habit_analysis['risk']
    
    # Keep only the required columns
    result_df = habit_analysis[['habit', 'risk', 'no_risk']]
    
    # Sort by habit name for better readability
    result_df = result_df.sort_values('habit')
    
    # Save to CSV
    output_file = 'habit_risk_analysis.csv'
    result_df.to_csv(output_file, index=False)
    
    print(f"\nHabit risk analysis saved to: {output_file}")
    print(f"Number of unique habits analyzed: {len(result_df)}")
    
    # Display the results
    print("\nHabit Risk Analysis Results:")
    print("=" * 50)
    print(result_df.to_string(index=False))
    
    # Additional statistics
    print(f"\nSummary Statistics:")
    print(f"Total risk events: {result_df['risk'].sum()}")
    print(f"Total no_risk events: {result_df['no_risk'].sum()}")
    print(f"Overall risk rate: {result_df['risk'].sum() / (result_df['risk'].sum() + result_df['no_risk'].sum()) * 100:.2f}%")
    
    return result_df

def create_simple_scatter_plot():
    """
    Create a simple scatter plot to visualize habit risk analysis data
    """
    # Load the habit risk analysis data
    df = pd.read_csv('habit_risk_analysis.csv')
    
    print(f"\nLoaded data with {len(df)} unique habits for visualization")
    
    # Create a simple scatter plot
    plt.figure(figsize=(10, 8))
    
    # Color points based on whether they have risk or not
    colors = ['red' if risk > 0 else 'blue' for risk in df['risk']]
    
    # Create the scatter plot
    plt.scatter(df['no_risk'], df['risk'], 
               c=colors, 
               s=80,
               alpha=0.7,
               edgecolors='black',
               linewidth=0.5)
    
    plt.xlabel('No Risk Events', fontsize=12, fontweight='bold')
    plt.ylabel('Risk Events', fontsize=12, fontweight='bold')
    plt.title('Habit Analysis: Risk vs No-Risk Events', 
              fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.scatter([], [], c='red', s=80, alpha=0.7, label='Habits with Risk Events')
    plt.scatter([], [], c='blue', s=80, alpha=0.7, label='Habits with No Risk Events')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('habit_risk_scatter.png', dpi=300, bbox_inches='tight')
    
    print("Simple scatter plot saved as: habit_risk_scatter.png")
    plt.show()

if __name__ == "__main__":
    # Create the CSV analysis
    #result_df = create_habit_risk_analysis()
    
    # Create simple scatter plot
    print("\nCreating scatter plot...")
    create_simple_scatter_plot()
