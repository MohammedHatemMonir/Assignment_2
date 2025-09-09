import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_motivation_bat_landing_hours_2_5_8():
    """Create a bar chart showing average bat landings vs motivation level for hours_after_sunset 2.5-8 and risk=1 only"""
    
    # Load the dataset
    print("Loading dataset2_edited.csv for motivation analysis (hours 2.5-8, risk=1)...")
    df = pd.read_csv('dataset2_edited.csv')
    
    # Filter for hours_after_sunset between 2.5 and 8 AND risk = 1
    filtered_df = df[(df['hours_after_sunset'] >= 2.5) & (df['hours_after_sunset'] <= 8) & (df['risk'] == 1)]
    
    # Clean the data - keep only records with all required values
    clean_df = filtered_df[['hours_after_sunset', 'bat_landing_number', 'motivation', 'risk']].dropna()
    
    print(f"Original dataset size: {len(df)} records")
    print(f"Records with hours_after_sunset 2.5-8 AND risk=1: {len(clean_df)} records")
    print(f"Hours range in filtered data: {clean_df['hours_after_sunset'].min():.1f} to {clean_df['hours_after_sunset'].max():.1f}")
    print(f"Risk values in filtered data: {clean_df['risk'].unique()}")
    print(f"Motivation range: {clean_df['motivation'].min():.2f} to {clean_df['motivation'].max():.2f}")
    print(f"Bat landing range: {clean_df['bat_landing_number'].min():.0f} to {clean_df['bat_landing_number'].max():.0f}")
    print()
    
    # Round motivation to nearest 0.5 for grouping
    clean_df = clean_df.copy()
    clean_df['motivation_rounded'] = (clean_df['motivation'] * 2).round() / 2
    
    # Calculate average bat landing numbers for each motivation group
    motivation_analysis = clean_df.groupby('motivation_rounded')['bat_landing_number'].agg([
        'mean', 'count', 'std', 'min', 'max', 'median'
    ]).reset_index()
    motivation_analysis.columns = ['motivation_level', 'avg_bat_landing', 'count', 'std_dev', 'min_landing', 'max_landing', 'median_landing']
    
    # Filter out groups with very few observations (less than 3 for this specific analysis)
    motivation_analysis = motivation_analysis[motivation_analysis['count'] >= 3]
    
    # Sort by motivation level
    motivation_analysis = motivation_analysis.sort_values('motivation_level')
    
    print("BAT LANDINGS vs MOTIVATION ANALYSIS (Hours 2.5-8 after sunset, Risk=1 ONLY):")
    print(f"Motivation groups analyzed: {len(motivation_analysis)}")
    print(f"Total records in analysis: {motivation_analysis['count'].sum()}")
    print()
    
    # Print detailed statistics for each motivation level
    print("DETAILED RESULTS BY MOTIVATION LEVEL:")
    for _, row in motivation_analysis.iterrows():
        print(f"Motivation {row['motivation_level']:.1f}: "
              f"Avg={row['avg_bat_landing']:.1f}, "
              f"Median={row['median_landing']:.1f}, "
              f"Count={row['count']:.0f}, "
              f"Std={row['std_dev']:.1f}, "
              f"Range={row['min_landing']:.0f}-{row['max_landing']:.0f}")
    
    print("="*80)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(motivation_analysis['motivation_level'], 
                   motivation_analysis['avg_bat_landing'],
                   width=0.25,
                   color='forestgreen',
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=0.8)
    
    # Customize the chart
    plt.xlabel('Estimated Motivation Level', fontsize=14, fontweight='bold')
    plt.ylabel('Average Bat Landing Count', fontsize=14, fontweight='bold')
    plt.title('Average Bat Landing Count by Motivation Level\n(Hours 2.5-8 After Sunset, Risk=1 Only)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add count labels below bars
    for i, bar in enumerate(bars):
        count = motivation_analysis.iloc[i]['count']
        plt.text(bar.get_x() + bar.get_width()/2., -2,
                f'n={count:.0f}',
                ha='center', va='top', fontsize=9, style='italic')
    
    # Customize grid and layout
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(motivation_analysis['motivation_level'])
    
    # Set y-axis to start from 0 for better visual comparison
    plt.ylim(bottom=0)
    
    # Add statistics text box
    overall_mean = motivation_analysis['avg_bat_landing'].mean()
    max_activity = motivation_analysis['avg_bat_landing'].max()
    min_activity = motivation_analysis['avg_bat_landing'].min()
    total_records = motivation_analysis['count'].sum()
    
    stats_text = f'Filters: 2.5-8 hours after sunset, Risk=1\nTotal Records: {total_records}\nOverall Mean: {overall_mean:.1f}\nHighest Avg: {max_activity:.1f}\nLowest Avg: {min_activity:.1f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('motivation_bat_landing_hours_2_5_8_risk_1_only.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('motivation_bat_landing_hours_2_5_8_risk_1_only.png', dpi=300, bbox_inches='tight')
    
    print("VISUALIZATION CREATED:")
    print("Average bat landing count by motivation level (2.5-8 hours after sunset, risk=1 only) saved as:")
    print("- motivation_bat_landing_hours_2_5_8_risk_1_only.pdf")
    print("- motivation_bat_landing_hours_2_5_8_risk_1_only.png")
    print()
    
    # Show some insights
    print("KEY INSIGHTS:")
    best_motivation = motivation_analysis.loc[motivation_analysis['avg_bat_landing'].idxmax()]
    worst_motivation = motivation_analysis.loc[motivation_analysis['avg_bat_landing'].idxmin()]
    
    print(f"Highest bat activity: Motivation level {best_motivation['motivation_level']:.1f} with {best_motivation['avg_bat_landing']:.1f} average landings")
    print(f"Lowest bat activity: Motivation level {worst_motivation['motivation_level']:.1f} with {worst_motivation['avg_bat_landing']:.1f} average landings")
    print(f"Difference: {best_motivation['avg_bat_landing'] - worst_motivation['avg_bat_landing']:.1f} landings between highest and lowest")
    
    plt.show()
    
    return motivation_analysis

if __name__ == "__main__":
    print("MOTIVATION vs BAT LANDING ANALYSIS")
    print("Hours After Sunset: 2.5-8 Period, Risk=1 Only")
    print("="*80)
    
    analysis_results = create_motivation_bat_landing_hours_2_5_8()
    
    print("\nAnalysis completed successfully!")
    print("="*80)
