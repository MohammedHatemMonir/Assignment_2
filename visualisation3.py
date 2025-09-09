import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load the dataset and perform comprehensive analysis of hours_after_sunset vs food_availability"""
    
    # Load the dataset
    print("Loading dataset2_edited.csv...")
    df = pd.read_csv('dataset2_edited.csv')
    
    # Extract the variables of interest
    hours_after_sunset = df['hours_after_sunset'].dropna()
    food_availability = df['food_availability'].dropna()
    
    # Create a clean dataset with both variables
    clean_df = df[['hours_after_sunset', 'food_availability']].dropna()
    x = clean_df['hours_after_sunset']
    y = clean_df['food_availability']
    
    print(f"Dataset loaded successfully with {len(clean_df)} valid observations.")
    print("="*80)
    
    return x, y, clean_df

def calculate_descriptive_statistics(x, y):
    """Calculate comprehensive descriptive statistics for both variables"""
    
    print("DESCRIPTIVE STATISTICS FOR HOURS AFTER SUNSET:")
    print(f"Mean hours after sunset: {np.mean(x):.2f} hours")
    print(f"Median hours after sunset: {np.median(x):.2f} hours")
    
    # Calculate mode
    from scipy import stats
    mode_result = stats.mode(x, keepdims=True)
    print(f"Mode hours after sunset: {mode_result.mode[0]:.2f} hours")
    print(f"Standard deviation: {np.std(x, ddof=1):.2f} hours")
    print(f"Minimum hours: {np.min(x):.2f} hours")
    print(f"Maximum hours: {np.max(x):.2f} hours")
    print(f"Range: {np.max(x) - np.min(x):.2f} hours")
    print(f"Variance: {np.var(x, ddof=1):.2f}")
    print(f"25th percentile (Q1): {np.percentile(x, 25):.2f} hours")
    print(f"75th percentile (Q3): {np.percentile(x, 75):.2f} hours")
    print(f"Interquartile Range (IQR): {np.percentile(x, 75) - np.percentile(x, 25):.2f} hours")
    print(f"Skewness: {stats.skew(x):.4f}")
    print(f"Kurtosis: {stats.kurtosis(x):.4f}")
    
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS FOR FOOD AVAILABILITY:")
    print(f"Mean food availability: {np.mean(y):.2f} units")
    print(f"Median food availability: {np.median(y):.2f} units")
    
    mode_result_y = stats.mode(y, keepdims=True)
    print(f"Mode food availability: {mode_result_y.mode[0]:.2f} units")
    print(f"Standard deviation: {np.std(y, ddof=1):.2f} units")
    print(f"Minimum availability: {np.min(y):.2f} units")
    print(f"Maximum availability: {np.max(y):.2f} units")
    print(f"Range: {np.max(y) - np.min(y):.2f} units")
    print(f"Variance: {np.var(y, ddof=1):.2f}")
    print(f"25th percentile (Q1): {np.percentile(y, 25):.2f} units")
    print(f"75th percentile (Q3): {np.percentile(y, 75):.2f} units")
    print(f"Interquartile Range (IQR): {np.percentile(y, 75) - np.percentile(y, 25):.2f} units")
    print(f"Skewness: {stats.skew(y):.4f}")
    print(f"Kurtosis: {stats.kurtosis(y):.4f}")
    
    print("\n" + "="*80)

def perform_trend_analysis(x, y):
    """Perform comprehensive trend analysis between the two variables"""
    
    # Linear regression analysis
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    print("TREND ANALYSIS (Hours After Sunset vs Food Availability):")
    print(f"Trend slope: {slope:.6f} units per hour")
    print(f"Intercept: {intercept:.6f} units")
    print(f"Standard error of slope: {std_err:.6f}")
    print(f"Correlation coefficient (r): {r_value:.4f}")
    print(f"R-squared (coefficient of determination): {r_value**2:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    # Statistical significance interpretation
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
        confidence = "99.9%"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
        confidence = "99%"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
        confidence = "95%"
    else:
        significance = "not significant (p >= 0.05)"
        confidence = "< 95%"
    
    print(f"Statistical significance: {significance}")
    print(f"Confidence level: {confidence}")
    
    # Trend interpretation
    r_squared = r_value**2
    if abs(r_value) < 0.1:
        trend_strength = "Very weak"
    elif abs(r_value) < 0.3:
        trend_strength = "Weak"
    elif abs(r_value) < 0.5:
        trend_strength = "Weak-to-moderate"
    elif abs(r_value) < 0.7:
        trend_strength = "Moderate"
    elif abs(r_value) < 0.9:
        trend_strength = "Strong"
    else:
        trend_strength = "Very strong"
    
    direction = "increasing" if slope > 0 else "decreasing"
    print(f"Trend interpretation: {trend_strength} {direction} trend")
    print(f"Effect size (R²): {r_squared*100:.1f}% of variance explained")
    print(f"Practical meaning: Food availability {'increases' if slope > 0 else 'decreases'} by {abs(slope):.6f} units per hour after sunset")
    
    # Additional correlation tests
    spearman_corr, spearman_p = stats.spearmanr(x, y)
    kendall_corr, kendall_p = stats.kendalltau(x, y)
    
    print(f"\nAdditional Correlation Analysis:")
    print(f"Spearman rank correlation: {spearman_corr:.4f} (p = {spearman_p:.6f})")
    print(f"Kendall's tau correlation: {kendall_corr:.4f} (p = {kendall_p:.6f})")
    
    print("\n" + "="*80)
    
    return slope, intercept, r_value, p_value, std_err

def create_visualizations(x, y, clean_df, slope, intercept):
    """Create a single scatter plot visualization for the relationship"""
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create a single scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.4f}x + {intercept:.4f}')
    
    plt.xlabel('Hours After Sunset', fontsize=14)
    plt.ylabel('Food Availability', fontsize=14)
    plt.title('Hours After Sunset vs Food Availability\n(with Linear Regression)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient to the plot
    r_value = np.corrcoef(x, y)[0, 1]
    plt.text(0.05, 0.95, f'r = {r_value:.4f}\nR² = {r_value**2:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('hours_sunset_food_availability_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('hours_sunset_food_availability_scatter.png', dpi=300, bbox_inches='tight')
    
    print("VISUALIZATION:")
    print("Scatter plot saved as:")
    print("- hours_sunset_food_availability_scatter.pdf")
    print("- hours_sunset_food_availability_scatter.png")
    
    plt.show()

def analyze_bat_landing_by_hour():
    """Analyze average bat landing numbers based on hours after sunset"""
    
    # Load the dataset
    print("Loading dataset2_edited.csv for bat landing analysis...")
    df = pd.read_csv('dataset2_edited.csv')
    
    # Clean the data
    clean_df = df[['hours_after_sunset', 'bat_landing_number']].dropna()
    
    # Round hours after sunset to nearest 0.5 for grouping
    clean_df['hours_rounded'] = clean_df['hours_after_sunset'].round(1)
    
    # Calculate average bat landing numbers for each hour group
    bat_avg_by_hour = clean_df.groupby('hours_rounded')['bat_landing_number'].agg(['mean', 'count', 'std']).reset_index()
    bat_avg_by_hour.columns = ['hours_after_sunset', 'avg_bat_landing', 'count', 'std_dev']
    
    # Filter out groups with very few observations (less than 5)
    bat_avg_by_hour = bat_avg_by_hour[bat_avg_by_hour['count'] >= 5]
    
    print(f"Analysis includes {len(bat_avg_by_hour)} hour groups with sufficient data.")
    print("="*80)
    
    # Print descriptive statistics
    print("BAT LANDING ANALYSIS BY HOURS AFTER SUNSET:")
    print(f"Total observations: {len(clean_df)}")
    print(f"Hour groups analyzed: {len(bat_avg_by_hour)}")
    print(f"Hours range: {clean_df['hours_after_sunset'].min():.1f} to {clean_df['hours_after_sunset'].max():.1f}")
    print(f"Overall average bat landings: {clean_df['bat_landing_number'].mean():.2f}")
    print(f"Overall median bat landings: {clean_df['bat_landing_number'].median():.2f}")
    print(f"Overall standard deviation: {clean_df['bat_landing_number'].std():.2f}")
    print()
    
    # Show top and bottom hours for bat activity
    top_hours = bat_avg_by_hour.nlargest(3, 'avg_bat_landing')
    bottom_hours = bat_avg_by_hour.nsmallest(3, 'avg_bat_landing')
    
    print("TOP 3 HOURS FOR BAT ACTIVITY:")
    for _, row in top_hours.iterrows():
        print(f"{row['hours_after_sunset']:.1f} hours: {row['avg_bat_landing']:.2f} average landings (n={row['count']:.0f})")
    
    print("\nBOTTOM 3 HOURS FOR BAT ACTIVITY:")
    for _, row in bottom_hours.iterrows():
        print(f"{row['hours_after_sunset']:.1f} hours: {row['avg_bat_landing']:.2f} average landings (n={row['count']:.0f})")
    
    print("="*80)
    
    return bat_avg_by_hour

def extract_middle_50_percent_records():
    """Extract the middle 50% of records based on bat_landing_number distribution"""
    
    # Load the dataset
    print("Loading dataset for middle 50% extraction...")
    df = pd.read_csv('dataset2_edited.csv')
    
    # Clean the data
    clean_df = df[['hours_after_sunset', 'bat_landing_number', 'motivation']].dropna()
    
    # Calculate Q1 and Q3 for bat_landing_number
    Q1 = clean_df['bat_landing_number'].quantile(0.25)
    Q3 = clean_df['bat_landing_number'].quantile(0.75)
    
    # Extract middle 50% records (interquartile range)
    middle_50_records = clean_df[(clean_df['bat_landing_number'] >= Q1) & 
                                (clean_df['bat_landing_number'] <= Q3)]
    
    print(f"Original dataset size: {len(clean_df)} records")
    print(f"Middle 50% dataset size: {len(middle_50_records)} records")
    print(f"Bat landing number range in middle 50%: {Q1:.1f} to {Q3:.1f}")
    print(f"Hours after sunset range in middle 50%: {middle_50_records['hours_after_sunset'].min():.1f} to {middle_50_records['hours_after_sunset'].max():.1f}")
    print(f"Motivation range in middle 50%: {middle_50_records['motivation'].min():.2f} to {middle_50_records['motivation'].max():.2f}")
    print("="*80)
    
    return middle_50_records

def visualize_middle_50_percent_records(middle_50_records, original_df):
    """Create a visualization showing the middle 50% of records"""
    
    plt.figure(figsize=(14, 10))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot comparing all data vs middle 50%
    ax1.scatter(original_df['hours_after_sunset'], original_df['bat_landing_number'], 
                alpha=0.3, s=20, color='lightgray', label='All Records')
    ax1.scatter(middle_50_records['hours_after_sunset'], middle_50_records['bat_landing_number'], 
                alpha=0.7, s=30, color='red', label='Middle 50% Records')
    ax1.set_xlabel('Hours After Sunset', fontsize=12)
    ax1.set_ylabel('Bat Landing Number', fontsize=12)
    ax1.set_title('Middle 50% Records vs All Records\n(Bat Landing vs Hours After Sunset)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of bat landing numbers
    ax2.hist(original_df['bat_landing_number'], bins=50, alpha=0.5, color='lightgray', 
             label='All Records', density=True)
    ax2.hist(middle_50_records['bat_landing_number'], bins=30, alpha=0.7, color='red', 
             label='Middle 50%', density=True)
    ax2.axvline(original_df['bat_landing_number'].quantile(0.25), color='blue', linestyle='--', 
                label='Q1 (25%)')
    ax2.axvline(original_df['bat_landing_number'].quantile(0.75), color='blue', linestyle='--', 
                label='Q3 (75%)')
    ax2.set_xlabel('Bat Landing Number', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Bat Landing Numbers\n(Showing Middle 50% Selection)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hours after sunset distribution in middle 50%
    ax3.hist(middle_50_records['hours_after_sunset'], bins=25, alpha=0.7, color='green', 
             edgecolor='black', linewidth=0.5)
    ax3.axvline(middle_50_records['hours_after_sunset'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {middle_50_records["hours_after_sunset"].mean():.1f}')
    ax3.axvline(middle_50_records['hours_after_sunset'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f'Median: {middle_50_records["hours_after_sunset"].median():.1f}')
    ax3.set_xlabel('Hours After Sunset', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Hours After Sunset Distribution\n(Middle 50% Records)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Motivation vs Bat Landing scatter plot for middle 50%
    ax4.scatter(middle_50_records['motivation'], middle_50_records['bat_landing_number'], 
                alpha=0.6, s=40, color='purple', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Motivation Level', fontsize=12)
    ax4.set_ylabel('Bat Landing Number', fontsize=12)
    ax4.set_title('Motivation vs Bat Landing\n(Middle 50% Records)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_coef = middle_50_records['motivation'].corr(middle_50_records['bat_landing_number'])
    ax4.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('middle_50_percent_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('middle_50_percent_analysis.png', dpi=300, bbox_inches='tight')
    
    print("MIDDLE 50% VISUALIZATION:")
    print("Comprehensive visualization of middle 50% records saved as:")
    print("- middle_50_percent_analysis.pdf")
    print("- middle_50_percent_analysis.png")
    print()
    
    # Print some key statistics
    print("KEY STATISTICS FOR MIDDLE 50% RECORDS:")
    print(f"Number of records: {len(middle_50_records)}")
    print(f"Bat landing range: {middle_50_records['bat_landing_number'].min():.0f} - {middle_50_records['bat_landing_number'].max():.0f}")
    print(f"Hours after sunset range: {middle_50_records['hours_after_sunset'].min():.1f} - {middle_50_records['hours_after_sunset'].max():.1f}")
    print(f"Motivation range: {middle_50_records['motivation'].min():.2f} - {middle_50_records['motivation'].max():.2f}")
    print(f"Average bat landings: {middle_50_records['bat_landing_number'].mean():.1f}")
    print(f"Average hours after sunset: {middle_50_records['hours_after_sunset'].mean():.1f}")
    print(f"Average motivation: {middle_50_records['motivation'].mean():.2f}")
    print("="*80)
    
    plt.show()

def analyze_bat_landings_vs_motivation(middle_50_records):
    """Analyze average bat landings by motivation level in the middle 50% records"""
    
    # Round motivation to nearest 0.5 for grouping
    middle_50_records = middle_50_records.copy()
    middle_50_records['motivation_rounded'] = (middle_50_records['motivation'] * 2).round() / 2
    
    # Calculate average bat landing numbers for each motivation group
    motivation_analysis = middle_50_records.groupby('motivation_rounded')['bat_landing_number'].agg([
        'mean', 'count', 'std', 'min', 'max'
    ]).reset_index()
    motivation_analysis.columns = ['motivation_level', 'avg_bat_landing', 'count', 'std_dev', 'min_landing', 'max_landing']
    
    # Filter out groups with very few observations (less than 5)
    motivation_analysis = motivation_analysis[motivation_analysis['count'] >= 5]
    
    # Sort by motivation level
    motivation_analysis = motivation_analysis.sort_values('motivation_level')
    
    print("MIDDLE 50% RECORDS - BAT LANDINGS vs MOTIVATION ANALYSIS:")
    print(f"Motivation groups analyzed: {len(motivation_analysis)}")
    print(f"Total records in analysis: {motivation_analysis['count'].sum()}")
    print()
    
    # Print detailed statistics for each motivation level
    print("DETAILED ANALYSIS BY MOTIVATION LEVEL:")
    for _, row in motivation_analysis.iterrows():
        print(f"Motivation {row['motivation_level']:.1f}: Avg={row['avg_bat_landing']:.1f}, "
              f"Count={row['count']:.0f}, Std={row['std_dev']:.1f}, "
              f"Range={row['min_landing']:.0f}-{row['max_landing']:.0f}")
    
    print("="*80)
    
    return motivation_analysis

def create_motivation_bat_landing_bar_chart_filtered_hours():
    """Create a bar chart showing average bat landings vs motivation level for hours_after_sunset 2.5-8"""
    
    # Load the dataset
    print("Loading dataset2_edited.csv for filtered motivation analysis...")
    df = pd.read_csv('dataset2_edited.csv')
    
    # Filter for hours_after_sunset between 2.5 and 8
    filtered_df = df[(df['hours_after_sunset'] >= 2.5) & (df['hours_after_sunset'] <= 8)]
    
    # Clean the data
    clean_df = filtered_df[['hours_after_sunset', 'bat_landing_number', 'motivation']].dropna()
    
    print(f"Original dataset size: {len(df)} records")
    print(f"Records with hours_after_sunset 2.5-8: {len(clean_df)} records")
    print(f"Hours range in filtered data: {clean_df['hours_after_sunset'].min():.1f} to {clean_df['hours_after_sunset'].max():.1f}")
    
    # Round motivation to nearest 0.5 for grouping
    clean_df = clean_df.copy()
    clean_df['motivation_rounded'] = (clean_df['motivation'] * 2).round() / 2
    
    # Calculate average bat landing numbers for each motivation group
    motivation_analysis = clean_df.groupby('motivation_rounded')['bat_landing_number'].agg([
        'mean', 'count', 'std', 'min', 'max'
    ]).reset_index()
    motivation_analysis.columns = ['motivation_level', 'avg_bat_landing', 'count', 'std_dev', 'min_landing', 'max_landing']
    
    # Filter out groups with very few observations (less than 3 for this specific analysis)
    motivation_analysis = motivation_analysis[motivation_analysis['count'] >= 3]
    
    # Sort by motivation level
    motivation_analysis = motivation_analysis.sort_values('motivation_level')
    
    print(f"Motivation groups analyzed: {len(motivation_analysis)}")
    print(f"Total records in analysis: {motivation_analysis['count'].sum()}")
    print()
    
    # Print detailed statistics for each motivation level
    print("BAT LANDINGS vs MOTIVATION (Hours 2.5-8 after sunset):")
    for _, row in motivation_analysis.iterrows():
        print(f"Motivation {row['motivation_level']:.1f}: Avg={row['avg_bat_landing']:.1f}, "
              f"Count={row['count']:.0f}, Std={row['std_dev']:.1f}, "
              f"Range={row['min_landing']:.0f}-{row['max_landing']:.0f}")
    
    print("="*80)
    
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(motivation_analysis['motivation_level'], 
                   motivation_analysis['avg_bat_landing'],
                   width=0.3,
                   color='forestgreen',
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Customize the chart
    plt.xlabel('Estimated Motivation Level', fontsize=14, fontweight='bold')
    plt.ylabel('Average Bat Landing Number', fontsize=14, fontweight='bold')
    plt.title('Average Bat Landings vs Estimated Motivation Level\n(Hours 2.5-8 After Sunset Only)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize grid and layout
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(motivation_analysis['motivation_level'])
    plt.tight_layout()
    
    # Add statistics text box
    overall_mean = motivation_analysis['avg_bat_landing'].mean()
    max_activity = motivation_analysis['avg_bat_landing'].max()
    min_activity = motivation_analysis['avg_bat_landing'].min()
    total_records = motivation_analysis['count'].sum()
    
    stats_text = f'Hours Filter: 2.5-8 after sunset\nTotal Records: {total_records}\nOverall Mean: {overall_mean:.1f}\nMax Activity: {max_activity:.1f}\nMin Activity: {min_activity:.1f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the chart
    plt.savefig('motivation_bat_landing_hours_2_5_8.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('motivation_bat_landing_hours_2_5_8.png', dpi=300, bbox_inches='tight')
    
    print("MOTIVATION vs BAT LANDING (Hours 2.5-8) VISUALIZATION:")
    print("Average bat landings by motivation level (2.5-8 hours after sunset) saved as:")
    print("- motivation_bat_landing_hours_2_5_8.pdf")
    print("- motivation_bat_landing_hours_2_5_8.png")
    
    plt.show()
    
    return motivation_analysis

def create_motivation_bat_landing_bar_chart(motivation_analysis):
    """Create a bar chart showing average bat landings vs motivation level"""
    
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(motivation_analysis['motivation_level'], 
                   motivation_analysis['avg_bat_landing'],
                   width=0.3,
                   color='forestgreen',
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Customize the chart
    plt.xlabel('Estimated Motivation Level', fontsize=14, fontweight='bold')
    plt.ylabel('Average Bat Landing Number', fontsize=14, fontweight='bold')
    plt.title('Average Bat Landings vs Estimated Motivation Level\n(Middle 50% Records)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize grid and layout
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(motivation_analysis['motivation_level'])
    plt.tight_layout()
    
    # Add statistics text box
    overall_mean = motivation_analysis['avg_bat_landing'].mean()
    max_activity = motivation_analysis['avg_bat_landing'].max()
    min_activity = motivation_analysis['avg_bat_landing'].min()
    
    stats_text = f'Overall Mean: {overall_mean:.1f}\nMax Activity: {max_activity:.1f}\nMin Activity: {min_activity:.1f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the chart
    plt.savefig('motivation_bat_landing_bar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('motivation_bat_landing_bar_chart.png', dpi=300, bbox_inches='tight')
    
    print("MOTIVATION vs BAT LANDING BAR CHART:")
    print("Average bat landings by motivation level saved as:")
    print("- motivation_bat_landing_bar_chart.pdf")
    print("- motivation_bat_landing_bar_chart.png")
    
    plt.show()

def create_bat_landing_bar_chart(bat_avg_by_hour):
    """Create a bar chart showing average bat landing numbers by hours after sunset"""
    
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(bat_avg_by_hour['hours_after_sunset'], 
                   bat_avg_by_hour['avg_bat_landing'],
                   width=0.3,
                   color='steelblue',
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Customize the chart
    plt.xlabel('Hours After Sunset', fontsize=14, fontweight='bold')
    plt.ylabel('Average Bat Landing Number', fontsize=14, fontweight='bold')
    plt.title('Average Bat Landing Numbers by Hours After Sunset', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on top of bars (without n count)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize grid and layout
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(bat_avg_by_hour['hours_after_sunset'], rotation=45)
    plt.tight_layout()
    
    # Add statistics text box
    overall_mean = bat_avg_by_hour['avg_bat_landing'].mean()
    max_activity = bat_avg_by_hour['avg_bat_landing'].max()
    min_activity = bat_avg_by_hour['avg_bat_landing'].min()
    
    stats_text = f'Overall Mean: {overall_mean:.1f}\nMax Activity: {max_activity:.1f}\nMin Activity: {min_activity:.1f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the chart
    plt.savefig('bat_landing_by_hour_bar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('bat_landing_by_hour_bar_chart.png', dpi=300, bbox_inches='tight')
    
    print("BAR CHART VISUALIZATION:")
    print("Average bat landing numbers by hour saved as:")
    print("- bat_landing_by_hour_bar_chart.pdf")
    print("- bat_landing_by_hour_bar_chart.png")
    
    plt.show()

def main():
    """Main function to run the complete analysis"""
    
    print("BAT LANDING ANALYSIS BY HOURS AFTER SUNSET")
    print("="*80)
    
    # Analyze bat landing patterns by hour
    bat_avg_by_hour = analyze_bat_landing_by_hour()
    
    # Create bar chart visualization
    create_bat_landing_bar_chart(bat_avg_by_hour)
    
    print("\n" + "="*80)
    print("MIDDLE 50% RECORDS ANALYSIS")
    print("="*80)
    
    # Load original dataframe for comparison
    original_df = pd.read_csv('dataset2_edited.csv')
    original_df = original_df[['hours_after_sunset', 'bat_landing_number', 'motivation']].dropna()
    
    # Extract middle 50% of records
    middle_50_records = extract_middle_50_percent_records()
    
    # Visualize the middle 50% records
    visualize_middle_50_percent_records(middle_50_records, original_df)
    
    # Analyze bat landings vs motivation in middle 50% records
    motivation_analysis = analyze_bat_landings_vs_motivation(middle_50_records)
    
    # Create motivation vs bat landing bar chart
    create_motivation_bat_landing_bar_chart(motivation_analysis)
    
    print("\n" + "="*80)
    print("MOTIVATION vs BAT LANDING (HOURS 2.5-8 FILTER)")
    print("="*80)
    
    # Create filtered analysis for hours 2.5-8 after sunset
    filtered_motivation_analysis = create_motivation_bat_landing_bar_chart_filtered_hours()
    
    print("\nComplete analysis finished successfully!")
    print("="*80)

if __name__ == "__main__":
    main()