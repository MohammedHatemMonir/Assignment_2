import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_seasons_vs_bat_landing():
    """
    Create a bar chart visualization of average bat landing numbers by season
    """
    # Read the dataset
    df = pd.read_csv('dataset1_merged.csv')

    # Create season labels based on Australian seasons
    season_labels = {
        0: 'Summer',
        1: 'Autumn', 
        2: 'Winter',
        3: 'Spring',
        4: 'Summer',
        5: 'Autumn'
    }

    # Map season numbers to labels
    df['season_name'] = df['season'].map(season_labels)

    # Calculate average bat landing numbers by season (only distinct seasons)
    seasonal_avg = df.groupby(['season', 'season_name'])['avg_bat_landing_number'].mean().reset_index()

    # Create single bar chart visualization
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = plt.bar(range(len(seasonal_avg)), seasonal_avg['avg_bat_landing_number'], 
                   color=colors[:len(seasonal_avg)])

    plt.title('Average Bat Landing Numbers by Season', fontsize=16, fontweight='bold')
    plt.xlabel('Season')
    plt.ylabel('Average Bat Landing Number')

    # Set x-axis labels showing season number and name
    plt.xticks(range(len(seasonal_avg)), 
               [f"Season {row['season']}\n{row['season_name']}" for _, row in seasonal_avg.iterrows()],
               rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('seasons_bat_landing_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('seasons_bat_landing_analysis.pdf', bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print("="*50)
    print("\nAverage Bat Landing Numbers by Season:")
    for _, row in seasonal_avg.iterrows():
        print(f"Season {row['season']} - {row['season_name']}: {row['avg_bat_landing_number']:.2f}")

    print("\nSeason Distribution:")
    season_counts = df['season_name'].value_counts()
    for season, count in season_counts.items():
        print(f"{season}: {count} observations")

    # Additional analysis - seasonal trends
    print("\n" + "="*50)
    print("Detailed Seasonal Analysis:")
    print("="*50)

    seasonal_stats = df.groupby(['season', 'season_name']).agg({
        'avg_bat_landing_number': ['mean', 'std', 'min', 'max', 'count'],
        'bat_landing_number': ['mean', 'std', 'min', 'max']
    }).round(2)

    print(seasonal_stats)
    
    return df

def compare_rat_encounter_vs_bat_landing():
    """
    Create a scatter plot comparing rat_encounter vs bat_landing_number
    """
    # Read the dataset
    df = pd.read_csv('dataset1_merged.csv')
    
    # Check if rat_encounter column exists
    if 'rat_encounter' not in df.columns:
        print("Error: 'rat_encounter' column not found in dataset.")
        print("Please run the add_rat_encounter_column() function first.")
        return None
    
    # Remove any rows with missing values in the columns we need
    plot_data = df[['rat_encounter', 'bat_landing_number']].dropna()
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with some transparency
    plt.scatter(plot_data['rat_encounter'], plot_data['bat_landing_number'], 
               alpha=0.6, s=50, color='#2E86AB', edgecolors='white', linewidth=0.5)
    
    # Add trend line
    if len(plot_data) > 1:
        z = np.polyfit(plot_data['rat_encounter'], plot_data['bat_landing_number'], 1)
        p = np.poly1d(z)
        plt.plot(plot_data['rat_encounter'].sort_values(), 
                p(plot_data['rat_encounter'].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope: {z[0]:.3f})')
    
    plt.title('Rat Encounter vs Bat Landing Number', fontsize=16, fontweight='bold')
    plt.xlabel('Rat Encounter (rat_arrival_number × bat_landing_number)')
    plt.ylabel('Bat Landing Number')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add correlation coefficient
    correlation = plot_data['rat_encounter'].corr(plot_data['bat_landing_number'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('rat_encounter_vs_bat_landing_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('rat_encounter_vs_bat_landing_scatter.pdf', bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("Rat Encounter vs Bat Landing Analysis:")
    print("="*50)
    print(f"Number of data points: {len(plot_data)}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"\nRat Encounter statistics:")
    print(f"  Mean: {plot_data['rat_encounter'].mean():.2f}")
    print(f"  Std:  {plot_data['rat_encounter'].std():.2f}")
    print(f"  Min:  {plot_data['rat_encounter'].min():.2f}")
    print(f"  Max:  {plot_data['rat_encounter'].max():.2f}")
    
    print(f"\nBat Landing Number statistics:")
    print(f"  Mean: {plot_data['bat_landing_number'].mean():.2f}")
    print(f"  Std:  {plot_data['bat_landing_number'].std():.2f}")
    print(f"  Min:  {plot_data['bat_landing_number'].min():.2f}")
    print(f"  Max:  {plot_data['bat_landing_number'].max():.2f}")
    
    return plot_data

def visualize_seasons_vs_rat_encounter():
    """
    Create a bar chart visualization of average rat encounter by season
    """
    # Read the dataset
    df = pd.read_csv('dataset1_merged.csv')
    
    # Check if rat_encounter column exists
    if 'rat_encounter' not in df.columns:
        print("Error: 'rat_encounter' column not found in dataset.")
        print("Please run the add_rat_encounter_column() function first.")
        return None

    # Create season labels based on Australian seasons
    season_labels = {
        0: 'Summer',
        1: 'Autumn', 
        2: 'Winter',
        3: 'Spring',
        4: 'Summer',
        5: 'Autumn'
    }

    # Map season numbers to labels
    df['season_name'] = df['season'].map(season_labels)

    # Calculate average rat encounter by season (only distinct seasons)
    seasonal_avg = df.groupby(['season', 'season_name'])['rat_encounter'].mean().reset_index()

    # Create single bar chart visualization
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = plt.bar(range(len(seasonal_avg)), seasonal_avg['rat_encounter'], 
                   color=colors[:len(seasonal_avg)])

    plt.title('Average Rat Encounter by Season', fontsize=16, fontweight='bold')
    plt.xlabel('Season')
    plt.ylabel('Average Rat Encounter')

    # Set x-axis labels showing season number and name
    plt.xticks(range(len(seasonal_avg)), 
               [f"Season {row['season']}\n{row['season_name']}" for _, row in seasonal_avg.iterrows()],
               rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('seasons_rat_encounter_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('seasons_rat_encounter_analysis.pdf', bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print("="*50)
    print("\nAverage Rat Encounter by Season:")
    for _, row in seasonal_avg.iterrows():
        print(f"Season {row['season']} - {row['season_name']}: {row['rat_encounter']:.2f}")

    print("\nSeason Distribution:")
    season_counts = df['season_name'].value_counts()
    for season, count in season_counts.items():
        print(f"{season}: {count} observations")

    # Additional analysis - seasonal trends
    print("\n" + "="*50)
    print("Detailed Seasonal Rat Encounter Analysis:")
    print("="*50)

    seasonal_stats = df.groupby(['season', 'season_name']).agg({
        'rat_encounter': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)

    print(seasonal_stats)
    
    return df

if __name__ == "__main__":
    # # Run the seasonal analysis
    # print("Running seasonal analysis...")
    # visualize_seasons_vs_bat_landing()
    
    # print("\n" + "="*70 + "\n")
    
    # # Run the scatter plot comparison
    # print("Running rat encounter vs bat landing comparison...")
    # compare_rat_encounter_vs_bat_landing()
    
    # Run the rat encounter seasonal analysis
    print("Running rat encounter seasonal analysis...")
    visualize_seasons_vs_rat_encounter()