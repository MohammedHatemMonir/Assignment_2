import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Data parameters and their types from dataset1.csv
data_types = {
    "start_time": "DateTime",
    "bat_landing_to_food": "Numeric",
    "habit": "Categorical",
    "rat_period_start": "DateTime",
    "rat_period_end": "DateTime",
    "seconds_after_rat_arrival": "Numeric",
    "risk": "Binary",
    "reward": "Binary",
    "month": "Numeric",
    "sunset_time": "DateTime",
    "hours_after_sunset": "Numeric",
    "season": "Numeric"
}

def plot_correlation_chart(csv_path, param1, param2, check_outlier=False):
    """
    Plots a correlation chart between two parameters from a CSV file.
    Handles different data type combinations:
    - Numeric vs Numeric: Scatter plot with correlation coefficient
    - Numeric vs Categorical: Box plot or violin plot
    - Numeric vs Binary: Box plot comparing groups
    - Categorical vs Categorical: Heatmap of cross-tabulation
    - DateTime vs Numeric: Time series plot
    - DateTime vs Categorical: Timeline with categories
    Optionally checks and marks outliers for numeric data.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Get data types for the parameters
    param1_type = data_types.get(param1, "Unknown") #return unknown if not found
    param2_type = data_types.get(param2, "Unknown") #return unknown if not found
    
    # Drop rows with missing values in the selected columns
    data = df[[param1, param2]].dropna()
    
    # Handle different data type combinations
    plt.figure(figsize=(10, 6))
    
    # Case 1: One Numeric, One Binary (prioritize bar chart over scatter plot)
    if (param1_type == "Numeric" and param2_type == "Binary") or \
       (param1_type == "Binary" and param2_type == "Numeric"):
        
        if param1_type == "Binary":
            # Swap to ensure numeric is on y-axis
            param1, param2 = param2, param1
            param1_type, param2_type = param2_type, param1_type
        
        # Create bar chart with mean values for each binary group
        grouped_means = data.groupby(param2)[param1].mean()
        
        # Create bar chart
        bars = plt.bar(['No (0)', 'Yes (1)'], [grouped_means[0], grouped_means[1]], 
                      color=['lightcoral', 'lightblue'], alpha=0.7)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(f'Average {param1} by {param2}')
        plt.ylabel(f'Average {param1}')
        plt.xlabel(param2)
    
    # Case 2: Both Numeric (excluding Binary combinations already handled above)
    elif param1_type == "Numeric" and param2_type == "Numeric":
        if check_outlier:
            # IQR outlier detection for numeric data
            Q1_param1 = data[param1].quantile(0.25)
            Q3_param1 = data[param1].quantile(0.75)
            IQR_param1 = Q3_param1 - Q1_param1
            
            Q1_param2 = data[param2].quantile(0.25)
            Q3_param2 = data[param2].quantile(0.75)
            IQR_param2 = Q3_param2 - Q1_param2
            
            lower_bound_param1 = Q1_param1 - 1.5 * IQR_param1
            upper_bound_param1 = Q3_param1 + 1.5 * IQR_param1
            lower_bound_param2 = Q1_param2 - 1.5 * IQR_param2
            upper_bound_param2 = Q3_param2 + 1.5 * IQR_param2
            
            outliers_param1 = (data[param1] < lower_bound_param1) | (data[param1] > upper_bound_param1)
            outliers_param2 = (data[param2] < lower_bound_param2) | (data[param2] > upper_bound_param2)
            outliers = outliers_param1 | outliers_param2
            
            data['Outlier'] = outliers
            palette = {False: 'blue', True: 'red'}
            sns.scatterplot(data=data, x=param1, y=param2, hue='Outlier', palette=palette)
            plt.title(f'Correlation between {param1} and {param2} (IQR Outliers Highlighted)')
        else:
            sns.scatterplot(data=data, x=param1, y=param2)
            plt.title(f'Correlation between {param1} and {param2}')
        
        # Add correlation coefficient for numeric pairs
        correlation = data[param1].corr(data[param2])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Case 3: Both Binary
    elif param1_type == "Binary" and param2_type == "Binary":
        # Create cross-tabulation for binary vs binary
        crosstab = pd.crosstab(data[param1], data[param2], margins=True)
        
        # Create a more readable version with labels
        binary_crosstab = pd.crosstab(
            data[param1].map({0: f'{param1}=No(0)', 1: f'{param1}=Yes(1)'}),
            data[param2].map({0: f'{param2}=No(0)', 1: f'{param2}=Yes(1)'}),
            margins=True
        )
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(binary_crosstab.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', 
                    cbar_kws={'label': 'Count'})
        plt.title(f'Cross-tabulation: {param1} vs {param2}')
        plt.ylabel(param1)
        plt.xlabel(param2)
        
        # Print detailed breakdown
        print(f"\nBinary vs Binary Analysis:")
        print(f"Cross-tabulation table:")
        print(binary_crosstab)
        print(f"\nInterpretation:")
        print(f"- {param1}=0 & {param2}=0: {crosstab.iloc[0,0]} cases")
        print(f"- {param1}=0 & {param2}=1: {crosstab.iloc[0,1]} cases") 
        print(f"- {param1}=1 & {param2}=0: {crosstab.iloc[1,0]} cases")
        print(f"- {param1}=1 & {param2}=1: {crosstab.iloc[1,1]} cases")
        
        # Calculate correlation for binary pairs
        correlation = data[param1].corr(data[param2])
        plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Case 4: One Numeric, One Categorical
    elif (param1_type == "Numeric" and param2_type == "Categorical") or \
         (param1_type == "Categorical" and param2_type == "Numeric"):
        
        if param1_type == "Categorical":
            # Swap to ensure numeric is on y-axis
            param1, param2 = param2, param1
            param1_type, param2_type = param2_type, param1_type
        
        sns.boxplot(data=data, x=param2, y=param1)
        plt.title(f'Distribution of {param1} by {param2}')
        plt.xticks(rotation=45)
    
    # Case 5: Both Categorical
    elif param1_type == "Categorical" and param2_type == "Categorical":
        # Create cross-tabulation
        crosstab = pd.crosstab(data[param1], data[param2])
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Cross-tabulation: {param1} vs {param2}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    
    # Case 6: DateTime vs Numeric
    elif (param1_type == "DateTime" and param2_type == "Numeric") or \
         (param1_type == "Numeric" and param2_type == "DateTime"):
        
        if param1_type == "Numeric":
            # Swap to ensure DateTime is on x-axis
            param1, param2 = param2, param1
            param1_type, param2_type = param2_type, param1_type
        
        # Convert datetime string to datetime object
        data[param1] = pd.to_datetime(data[param1], format='%d/%m/%Y %H:%M')
        plt.plot(data[param1], data[param2], 'o-', alpha=0.7)
        plt.title(f'Time Series: {param2} over {param1}')
        plt.xticks(rotation=45)
    
    # Case 7: DateTime vs Categorical
    elif (param1_type == "DateTime" and param2_type == "Categorical") or \
         (param1_type == "Categorical" and param2_type == "DateTime"):
        
        if param1_type == "Categorical":
            # Swap to ensure DateTime is on x-axis
            param1, param2 = param2, param1
            param1_type, param2_type = param2_type, param1_type
        
        # Convert datetime string to datetime object
        data[param1] = pd.to_datetime(data[param1], format='%d/%m/%Y %H:%M')
        
        # Create timeline with categories
        for category in data[param2].unique():
            if pd.notna(category) and category != '':
                subset = data[data[param2] == category]
                plt.scatter(subset[param1], [category]*len(subset), label=category, alpha=0.7)
        
        plt.title(f'Timeline: {param2} over {param1}')
        plt.xticks(rotation=45)
        plt.legend()
    
    # Default case
    else:
        print(f"Unsupported combination: {param1_type} vs {param2_type}")
        return
    
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.tight_layout()
    plt.show()
    
    # Print analysis summary
    print(f"\nAnalysis Summary:")
    print(f"Parameter 1: {param1} ({param1_type})")
    print(f"Parameter 2: {param2} ({param2_type})")
    print(f"Data points analyzed: {len(data)}")
    
    if param1_type in ["Numeric", "Binary"] and param2_type in ["Numeric", "Binary"]:
        correlation = data[param1].corr(data[param2])
        print(f"Pearson correlation coefficient: {correlation:.3f}")
        if abs(correlation) > 0.7:
            print("Strong correlation detected!")
        elif abs(correlation) > 0.3:
            print("Moderate correlation detected.")
        else:
            print("Weak or no correlation.")
    
    print(f"Visualization type: {get_chart_type(param1_type, param2_type)}")

def plot_binary_vs_categorical(csv_path, binary_col, categorical_col):
    """
    Creates a simple bar chart showing counts of each category split by binary values.
    For example: habit categories (fast, rat, pick) split by risk (0=no risk, 1=risk).
    
    Args:
        csv_path (str): Path to the CSV file
        binary_col (str): Name of the binary column (should contain 0s and 1s)
        categorical_col (str): Name of the categorical column
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing values in the selected columns
    data = df[[binary_col, categorical_col]].dropna()
    
    # Get unique categories
    categories = data[categorical_col].unique()
    
    # Count occurrences for each combination
    counts_0 = []  # binary = 0 (no risk/no reward)
    counts_1 = []  # binary = 1 (risk/reward)
    
    for category in categories:
        # Count for binary = 0
        count_0 = len(data[(data[categorical_col] == category) & (data[binary_col] == 0)])
        counts_0.append(count_0)
        
        # Count for binary = 1
        count_1 = len(data[(data[categorical_col] == category) & (data[binary_col] == 1)])
        counts_1.append(count_1)
    
    # Create bar chart
    x = range(len(categories))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    
    # Create bars
    bars1 = plt.bar([i - width/2 for i in x], counts_0, width, 
                    label=f'No {binary_col.title()} (0)', color='lightcoral', alpha=0.8)
    bars2 = plt.bar([i + width/2 for i in x], counts_1, width, 
                    label=f'{binary_col.title()} (1)', color='lightblue', alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    # Customize chart
    plt.title(f'{categorical_col.title()} Distribution by {binary_col.title()}')
    plt.xlabel(f'{categorical_col.title()} Categories')
    plt.ylabel('Count')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print(f"\nBinary vs Categorical Analysis:")
    print(f"Binary column: {binary_col}")
    print(f"Categorical column: {categorical_col}")
    print(f"Total data points: {len(data)}")
    print("\nDetailed breakdown:")
    
    for i, category in enumerate(categories):
        print(f"{category}: No {binary_col} = {counts_0[i]}, {binary_col.title()} = {counts_1[i]}")

def get_chart_type(type1, type2):
    """Helper function to determine chart type based on data types"""
    if (type1 == "Numeric" and type2 == "Binary") or (type1 == "Binary" and type2 == "Numeric"):
        return "Bar Chart (Binary Groups)"
    elif type1 == "Numeric" and type2 == "Numeric":
        return "Scatter Plot"
    elif type1 == "Binary" and type2 == "Binary":
        return "Heatmap (Binary Cross-tabulation)"
    elif (type1 == "Numeric" and type2 == "Categorical") or (type1 == "Categorical" and type2 == "Numeric"):
        return "Box Plot"
    elif type1 == "Categorical" and type2 == "Categorical":
        return "Heatmap (Cross-tabulation)"
    elif (type1 == "DateTime" and type2 == "Numeric") or (type1 == "Numeric" and type2 == "DateTime"):
        return "Time Series Plot"
    elif (type1 == "DateTime" and type2 == "Categorical") or (type1 == "Categorical" and type2 == "DateTime"):
        return "Timeline Scatter"
    else:
        return "Unsupported"

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

def analyze_risk_for_rat_habit(csv_path, habit_value="rat"):
    """
    Analyze and visualize risk values (0 or 1) for entries with a specific habit.
    Creates a bar chart showing the count of risk=0 and risk=1 for the specified habit entries.
    
    Args:
        csv_path (str): Path to the CSV file
        habit_value (str): The habit value to filter by (default: "rat")
    
    Returns:
        dict: Dictionary containing counts and percentages
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter data for the specified habit
    habit_data = df[df['habit'] == habit_value]
    
    if len(habit_data) == 0:
        print(f"No data found where habit = '{habit_value}'")
        return None
    
    # Count risk values for the specified habit
    risk_counts = habit_data['risk'].value_counts().sort_index()
    
    # Calculate percentages
    total_habit_entries = len(habit_data)
    risk_percentages = (risk_counts / total_habit_entries * 100).round(2)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(['Risk = 0', 'Risk = 1'], 
                   [risk_counts.get(0, 0), risk_counts.get(1, 0)], 
                   color=['lightgreen', 'lightcoral'], 
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        risk_value = i  # 0 for first bar, 1 for second bar
        count = risk_counts.get(risk_value, 0)
        percentage = risk_percentages.get(risk_value, 0)
        
        # Add count and percentage labels
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(count)}\n({percentage}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # Customize the chart
    plt.title(f'Risk Distribution for Bat Entries with Habit = "{habit_value}"', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Entries', fontsize=12)
    plt.xlabel('Risk Level', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add a text box with summary statistics
    summary_text = f'Total "{habit_value}" habit entries: {total_habit_entries}'
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("="*50)
    print(f"RISK ANALYSIS FOR HABIT = '{habit_value.upper()}'")
    print("="*50)
    print(f"Total entries where habit = '{habit_value}': {total_habit_entries}")
    print(f"Risk = 0 (Low Risk): {risk_counts.get(0, 0)} entries ({risk_percentages.get(0, 0)}%)")
    print(f"Risk = 1 (High Risk): {risk_counts.get(1, 0)} entries ({risk_percentages.get(1, 0)}%)")
    print("="*50)
    
    # Additional insights
    if risk_counts.get(1, 0) > risk_counts.get(0, 0):
        print(f"INSIGHT: More '{habit_value}' habit entries have HIGH RISK (1) than low risk (0)")
    elif risk_counts.get(0, 0) > risk_counts.get(1, 0):
        print(f"INSIGHT: More '{habit_value}' habit entries have LOW RISK (0) than high risk (1)")
    else:
        print(f"INSIGHT: Equal number of high and low risk entries for '{habit_value}' habit")
    
    print("="*50)
    
    # Return results as dictionary
    return {
        'total_habit_entries': total_habit_entries,
        'risk_0_count': risk_counts.get(0, 0),
        'risk_1_count': risk_counts.get(1, 0),
        'risk_0_percentage': risk_percentages.get(0, 0),
        'risk_1_percentage': risk_percentages.get(1, 0),
        'raw_data': habit_data
    }

# Example usage:
# Original correlation chart


# plot_correlation_chart('dataset1.csv', 'risk', 'reward', check_outlier=True)

# New binary vs categorical analysis
# plot_binary_vs_categorical('dataset1.csv', 'risk', 'habit')
# plot_binary_vs_categorical('dataset1.csv', 'reward', 'habit')

def analyze_bat_landing_over_time_for_rat_habit(csv_path):
    """
    Analyze and visualize how bat_landing_to_food changes over time for entries where habit contains 'rat'.
    Creates a time series plot showing the bat landing time trends for any habit containing 'rat'.
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        dict: Dictionary containing analysis results
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter data for any habit that contains 'rat' (case-insensitive)
    rat_data = df[df['habit'].str.contains('rat', case=False, na=False)].copy()
    
    if len(rat_data) == 0:
        print("No data found where habit contains 'rat'")
        return None
    
    # Print what habits were found containing 'rat'
    unique_rat_habits = rat_data['habit'].unique()
    print(f"Found habits containing 'rat': {list(unique_rat_habits)}")
    
    # Convert start_time to datetime - use flexible parsing to handle different formats
    rat_data['start_time'] = pd.to_datetime(rat_data['start_time'], format='mixed')
    
    # Sort by start_time
    rat_data = rat_data.sort_values('start_time')
    
    # Remove any NaN values in bat_landing_to_food
    rat_data = rat_data.dropna(subset=['bat_landing_to_food'])
    
    # Store original data count for comparison
    original_count = len(rat_data)
    
    # Remove outliers using IQR method
    Q1 = rat_data['bat_landing_to_food'].quantile(0.25)
    Q3 = rat_data['bat_landing_to_food'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds (1.5 * IQR is the standard multiplier)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers_mask = (rat_data['bat_landing_to_food'] < lower_bound) | (rat_data['bat_landing_to_food'] > upper_bound)
    outliers = rat_data[outliers_mask]
    
    # Remove outliers
    rat_data = rat_data[~outliers_mask].copy()
    
    # Print outlier removal summary
    outliers_removed = original_count - len(rat_data)
    print(f"\nOUTLIER REMOVAL SUMMARY:")
    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Original data points: {original_count}")
    print(f"Outliers removed: {outliers_removed}")
    print(f"Data points after outlier removal: {len(rat_data)}")
    
    if outliers_removed > 0:
        print(f"Outlier values removed: {sorted(outliers['bat_landing_to_food'].tolist())}")
    
    # Create the visualization
    plt.figure(figsize=(14, 8)) 
    
    # Create time series plot
    plt.subplot(2, 1, 1)
    
    # Plot connected line through all data points with markers
    plt.plot(rat_data['start_time'], rat_data['bat_landing_to_food'], 
             'o-', alpha=0.8, markersize=5, linewidth=1.5, color='darkblue', 
             markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=1)
    
    # Add trend line using numeric indices for calculation but plot against actual dates
    from scipy import stats
    import numpy as np
    
    # Check if we have enough data points for meaningful statistical analysis
    if len(rat_data) < 3:
        print("Warning: Too few data points for reliable trend analysis")
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
        trend_line = [rat_data['bat_landing_to_food'].mean()] * len(rat_data)
    else:
        # Use observation order (not actual time differences) for trend analysis
        x_numeric = np.array(range(len(rat_data)))
        y_values = rat_data['bat_landing_to_food'].values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
        trend_line = slope * x_numeric + intercept
        
        # Double-check p-value calculation with alternative method
        # Calculate t-statistic and degrees of freedom for verification
        n = len(rat_data)
        df = n - 2  # degrees of freedom for linear regression
        
        if std_err > 0:
            t_stat = slope / std_err
            # Two-tailed p-value
            p_value_check = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            print(f"\nSTATISTICAL VALIDATION:")
            print(f"Sample size: {n}")
            print(f"Degrees of freedom: {df}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"Standard error of slope: {std_err:.6f}")
            print(f"P-value from linregress: {p_value:.6f}")
            print(f"P-value verification: {p_value_check:.6f}")
            
            # Use the more precise calculation if they differ significantly
            if abs(p_value - p_value_check) > 0.001:
                print(f"Note: Using verified p-value calculation")
                p_value = p_value_check
    
    plt.plot(rat_data['start_time'], trend_line, '--', color='red', linewidth=2, 
             label=f'Trend (slope={slope:.3f})')
    
    plt.title('Bat Landing to Food Time Over Time (Habits containing "rat")', fontsize=14, fontweight='bold')
    plt.xlabel('Observation Sequence (Chronological Order)')
    plt.ylabel('Bat Landing to Food (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Use a more compact x-axis approach - limit number of date labels to avoid crowding
    import matplotlib.dates as mdates
    ax = plt.gca()
    
    # Set fewer tick locations to reduce gaps appearance
    if len(rat_data) > 20:
        # Show every nth date to avoid overcrowding
        step = max(1, len(rat_data) // 10)
        tick_positions = rat_data['start_time'].iloc[::step]
        ax.set_xticks(tick_positions)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    plt.xticks(rotation=45)
    
    # Tighten x-axis limits to reduce visual gaps
    plt.xlim(rat_data['start_time'].min(), rat_data['start_time'].max())
    
    plt.legend()
    
    # Add text box showing date range and number of records
    date_info = f'Records: {len(rat_data)} entries\nFrom: {rat_data["start_time"].min().strftime("%d/%m/%Y %H:%M")}\nTo: {rat_data["start_time"].max().strftime("%d/%m/%Y %H:%M")}'
    plt.text(0.02, 0.98, date_info, transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontsize=9)
    
    # Create histogram for distribution
    plt.subplot(2, 1, 2)
    plt.hist(rat_data['bat_landing_to_food'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Distribution of Bat Landing to Food Times (Habits containing "rat")', fontsize=14, fontweight='bold')
    plt.xlabel('Bat Landing to Food (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean, median, and mode
    mean_value = rat_data['bat_landing_to_food'].mean()
    median_value = rat_data['bat_landing_to_food'].median()
    mode_result = rat_data['bat_landing_to_food'].mode()
    mode_value = mode_result[0] if len(mode_result) > 0 else median_value  # Use first mode if multiple exist
    
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.axvline(mode_value, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode_value:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats_summary = {
        'total_entries': len(rat_data),
        'mean_bat_landing': mean_value,
        'median_bat_landing': median_value,
        'mode_bat_landing': mode_value,
        'std_bat_landing': rat_data['bat_landing_to_food'].std(),
        'min_bat_landing': rat_data['bat_landing_to_food'].min(),
        'max_bat_landing': rat_data['bat_landing_to_food'].max(),
        'trend_slope': slope,
        'trend_r_value': r_value,
        'trend_p_value': p_value,
        'date_range': {
            'start': rat_data['start_time'].min(),
            'end': rat_data['start_time'].max()
        }
    }
    
    # Print detailed analysis
    print("="*70)
    print("BAT LANDING TO FOOD TIME ANALYSIS FOR HABITS CONTAINING 'RAT'")
    print("="*70)
    print(f"Total entries with habits containing 'rat': {stats_summary['total_entries']}")
    print(f"Habits found: {list(unique_rat_habits)}")
    print(f"Date range: {stats_summary['date_range']['start'].strftime('%d/%m/%Y %H:%M')} to {stats_summary['date_range']['end'].strftime('%d/%m/%Y %H:%M')}")
    print()
    print("DESCRIPTIVE STATISTICS:")
    print(f"Mean bat landing time: {stats_summary['mean_bat_landing']:.2f} seconds")
    print(f"Median bat landing time: {stats_summary['median_bat_landing']:.2f} seconds")
    print(f"Mode bat landing time: {stats_summary['mode_bat_landing']:.2f} seconds")
    print(f"Standard deviation: {stats_summary['std_bat_landing']:.2f} seconds")
    print(f"Minimum time: {stats_summary['min_bat_landing']:.2f} seconds")
    print(f"Maximum time: {stats_summary['max_bat_landing']:.2f} seconds")
    print()
    print("TREND ANALYSIS:")
    print(f"Trend slope: {stats_summary['trend_slope']:.6f} seconds per observation")
    print(f"Standard error of slope: {std_err:.6f}")
    print(f"Correlation coefficient (r): {stats_summary['trend_r_value']:.4f}")
    print(f"R-squared (coefficient of determination): {stats_summary['trend_r_value']**2:.4f}")
    print(f"P-value: {stats_summary['trend_p_value']:.6f}")
    
    # Statistical significance
    if stats_summary['trend_p_value'] < 0.001:
        significance_level = "highly significant (p < 0.001)"
    elif stats_summary['trend_p_value'] < 0.01:
        significance_level = "very significant (p < 0.01)"
    elif stats_summary['trend_p_value'] < 0.05:
        significance_level = "significant (p < 0.05)"
    elif stats_summary['trend_p_value'] < 0.1:
        significance_level = "marginally significant (p < 0.1)"
    else:
        significance_level = "not statistically significant (p >= 0.1)"
    
    print(f"Statistical significance: {significance_level}")
    
    # Confidence level
    confidence_percentage = (1 - stats_summary['trend_p_value']) * 100
    if confidence_percentage > 99:
        confidence_percentage = 99.9
    print(f"Confidence level: {confidence_percentage:.1f}%")
    
    # Trend interpretation
    if abs(stats_summary['trend_r_value']) > 0.7:
        trend_strength = "Strong"
    elif abs(stats_summary['trend_r_value']) > 0.5:
        trend_strength = "Moderate-to-strong"
    elif abs(stats_summary['trend_r_value']) > 0.3:
        trend_strength = "Moderate"
    elif abs(stats_summary['trend_r_value']) > 0.1:
        trend_strength = "Weak-to-moderate"
    else:
        trend_strength = "Very weak"

    if stats_summary['trend_slope'] > 0:
        trend_direction = "increasing"
    elif stats_summary['trend_slope'] < 0:
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"

    print(f"Trend interpretation: {trend_strength} {trend_direction} trend over time")
    print(f"Effect size (R²): {(stats_summary['trend_r_value']**2)*100:.1f}% of variance explained by time")
    
    # Practical meaning
    if abs(stats_summary['trend_slope']) > 0.001:
        if stats_summary['trend_slope'] > 0:
            print(f"Practical meaning: Bat landing time increases by {stats_summary['trend_slope']:.3f} seconds per observation")
        else:
            print(f"Practical meaning: Bat landing time decreases by {abs(stats_summary['trend_slope']):.3f} seconds per observation")
    else:
        print("Practical meaning: Bat landing time remains essentially constant over time")
    
    print("="*60)
    
    return stats_summary

def analyze_time_difference_over_time(csv_path, habit_filter, time_field1, time_field2, analysis_name=None):
    """
    Analyze and visualize how the time difference between two datetime fields changes over time 
    for entries where habit contains the specified filter.
    
    Args:
        csv_path (str): Path to the CSV file
        habit_filter (str): String to filter habits (case-insensitive contains search)
        time_field1 (str): First datetime field name (e.g., 'start_time')
        time_field2 (str): Second datetime field name (e.g., 'rat_period_start')
        analysis_name (str): Optional custom name for the analysis (auto-generated if None)
    
    Returns:
        dict: Dictionary containing analysis results
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter data for any habit that contains the specified filter (case-insensitive)
    filtered_data = df[df['habit'].str.contains(habit_filter, case=False, na=False)].copy()
    
    if len(filtered_data) == 0:
        print(f"No data found where habit contains '{habit_filter}'")
        return None
    
    # Print what habits were found containing the filter
    unique_habits = filtered_data['habit'].unique()
    print(f"Found habits containing '{habit_filter}': {list(unique_habits)}")
    
    # Convert datetime fields to datetime objects - use flexible parsing to handle different formats
    try:
        filtered_data[time_field1] = pd.to_datetime(filtered_data[time_field1], format='mixed')
        filtered_data[time_field2] = pd.to_datetime(filtered_data[time_field2], format='mixed')
    except Exception as e:
        print(f"Error converting datetime fields: {e}")
        return None
    
    # Calculate time difference in seconds (time_field1 - time_field2)
    filtered_data['time_difference'] = (filtered_data[time_field1] - filtered_data[time_field2]).dt.total_seconds()
    
    # Sort by the first time field for chronological analysis
    filtered_data = filtered_data.sort_values(time_field1)
    
    # Remove any NaN values in the calculated time difference
    filtered_data = filtered_data.dropna(subset=['time_difference'])
    
    # Store original data count for comparison
    original_count = len(filtered_data)
    
    # Remove outliers using IQR method
    Q1 = filtered_data['time_difference'].quantile(0.25)
    Q3 = filtered_data['time_difference'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds (1.5 * IQR is the standard multiplier)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers_mask = (filtered_data['time_difference'] < lower_bound) | (filtered_data['time_difference'] > upper_bound)
    outliers = filtered_data[outliers_mask]
    
    # Remove outliers
    filtered_data = filtered_data[~outliers_mask].copy()
    
    # Print outlier removal summary
    outliers_removed = original_count - len(filtered_data)
    print(f"\nOUTLIER REMOVAL SUMMARY:")
    print(f"Q1 (25th percentile): {Q1:.2f} seconds")
    print(f"Q3 (75th percentile): {Q3:.2f} seconds")
    print(f"IQR: {IQR:.2f} seconds")
    print(f"Lower bound: {lower_bound:.2f} seconds")
    print(f"Upper bound: {upper_bound:.2f} seconds")
    print(f"Original data points: {original_count}")
    print(f"Outliers removed: {outliers_removed}")
    print(f"Data points after outlier removal: {len(filtered_data)}")
    
    if outliers_removed > 0:
        print(f"Outlier values removed: {sorted(outliers['time_difference'].tolist())}")
    
    # Create analysis name if not provided
    if analysis_name is None:
        analysis_name = f"Time Difference: {time_field1} - {time_field2}"
    
    # Create the visualization
    plt.figure(figsize=(14, 8)) 
    
    # Create time series plot
    plt.subplot(2, 1, 1)
    
    # Plot connected line through all data points with markers
    plt.plot(filtered_data[time_field1], filtered_data['time_difference'], 
             'o-', alpha=0.8, markersize=5, linewidth=1.5, color='darkblue', 
             markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=1)
    
    # Add trend line using numeric indices for calculation but plot against actual dates
    from scipy import stats
    import numpy as np
    
    # Check if we have enough data points for meaningful statistical analysis
    if len(filtered_data) < 3:
        print("Warning: Too few data points for reliable trend analysis")
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
        trend_line = [filtered_data['time_difference'].mean()] * len(filtered_data)
    else:
        # Use observation order (not actual time differences) for trend analysis
        x_numeric = np.array(range(len(filtered_data)))
        y_values = filtered_data['time_difference'].values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
        trend_line = slope * x_numeric + intercept
        
        # Double-check p-value calculation with alternative method
        # Calculate t-statistic and degrees of freedom for verification
        n = len(filtered_data)
        df = n - 2  # degrees of freedom for linear regression
        
        if std_err > 0:
            t_stat = slope / std_err
            # Two-tailed p-value
            p_value_check = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            print(f"\nSTATISTICAL VALIDATION:")
            print(f"Sample size: {n}")
            print(f"Degrees of freedom: {df}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"Standard error of slope: {std_err:.6f}")
            print(f"P-value from linregress: {p_value:.6f}")
            print(f"P-value verification: {p_value_check:.6f}")
            
            # Use the more precise calculation if they differ significantly
            if abs(p_value - p_value_check) > 0.001:
                print(f"Note: Using verified p-value calculation")
                p_value = p_value_check
    
    plt.plot(filtered_data[time_field1], trend_line, '--', color='red', linewidth=2, 
             label=f'Trend (slope={slope:.3f})')
    
    plt.title(f'{analysis_name} Over Time (Habits containing "{habit_filter}")', fontsize=14, fontweight='bold')
    plt.xlabel('Observation Sequence (Chronological Order)')
    plt.ylabel('Time Difference (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Use a more compact x-axis approach - limit number of date labels to avoid crowding
    import matplotlib.dates as mdates
    ax = plt.gca()
    
    # Set fewer tick locations to reduce gaps appearance
    if len(filtered_data) > 20:
        # Show every nth date to avoid overcrowding
        step = max(1, len(filtered_data) // 10)
        tick_positions = filtered_data[time_field1].iloc[::step]
        ax.set_xticks(tick_positions)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    plt.xticks(rotation=45)
    
    # Tighten x-axis limits to reduce visual gaps
    plt.xlim(filtered_data[time_field1].min(), filtered_data[time_field1].max())
    
    plt.legend()
    
    # Add text box showing date range and number of records
    date_info = f'Records: {len(filtered_data)} entries\nFrom: {filtered_data[time_field1].min().strftime("%d/%m/%Y %H:%M")}\nTo: {filtered_data[time_field1].max().strftime("%d/%m/%Y %H:%M")}'
    plt.text(0.02, 0.98, date_info, transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontsize=9)
    
    # Create histogram for distribution
    plt.subplot(2, 1, 2)
    plt.hist(filtered_data['time_difference'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title(f'Distribution of {analysis_name} (Habits containing "{habit_filter}")', fontsize=14, fontweight='bold')
    plt.xlabel('Time Difference (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean, median, and mode
    mean_value = filtered_data['time_difference'].mean()
    median_value = filtered_data['time_difference'].median()
    mode_result = filtered_data['time_difference'].mode()
    mode_value = mode_result[0] if len(mode_result) > 0 else median_value  # Use first mode if multiple exist
    
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.axvline(mode_value, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode_value:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats_summary = {
        'total_entries': len(filtered_data),
        'mean_time_diff': mean_value,
        'median_time_diff': median_value,
        'mode_time_diff': mode_value,
        'std_time_diff': filtered_data['time_difference'].std(),
        'min_time_diff': filtered_data['time_difference'].min(),
        'max_time_diff': filtered_data['time_difference'].max(),
        'trend_slope': slope,
        'trend_r_value': r_value,
        'trend_p_value': p_value,
        'date_range': {
            'start': filtered_data[time_field1].min(),
            'end': filtered_data[time_field1].max()
        },
        'analysis_name': analysis_name,
        'habit_filter': habit_filter,
        'time_fields': (time_field1, time_field2)
    }
    
    # Print detailed analysis
    print("="*80)
    print(f"{analysis_name.upper()} ANALYSIS FOR HABITS CONTAINING '{habit_filter.upper()}'")
    print("="*80)
    print(f"Total entries with habits containing '{habit_filter}': {stats_summary['total_entries']}")
    print(f"Habits found: {list(unique_habits)}")
    print(f"Analyzing: {time_field1} - {time_field2}")
    print(f"Date range: {stats_summary['date_range']['start'].strftime('%d/%m/%Y %H:%M')} to {stats_summary['date_range']['end'].strftime('%d/%m/%Y %H:%M')}")
    print()
    print("DESCRIPTIVE STATISTICS:")
    print(f"Mean time difference: {stats_summary['mean_time_diff']:.2f} seconds")
    print(f"Median time difference: {stats_summary['median_time_diff']:.2f} seconds")
    print(f"Mode time difference: {stats_summary['mode_time_diff']:.2f} seconds")
    print(f"Standard deviation: {stats_summary['std_time_diff']:.2f} seconds")
    print(f"Minimum time difference: {stats_summary['min_time_diff']:.2f} seconds")
    print(f"Maximum time difference: {stats_summary['max_time_diff']:.2f} seconds")
    print()
    print("TREND ANALYSIS:")
    print(f"Trend slope: {stats_summary['trend_slope']:.6f} seconds per observation")
    print(f"Standard error of slope: {std_err:.6f}")
    print(f"Correlation coefficient (r): {stats_summary['trend_r_value']:.4f}")
    print(f"R-squared (coefficient of determination): {stats_summary['trend_r_value']**2:.4f}")
    print(f"P-value: {stats_summary['trend_p_value']:.6f}")
    
    # Statistical significance
    if stats_summary['trend_p_value'] < 0.001:
        significance_level = "highly significant (p < 0.001)"
    elif stats_summary['trend_p_value'] < 0.01:
        significance_level = "very significant (p < 0.01)"
    elif stats_summary['trend_p_value'] < 0.05:
        significance_level = "significant (p < 0.05)"
    elif stats_summary['trend_p_value'] < 0.1:
        significance_level = "marginally significant (p < 0.1)"
    else:
        significance_level = "not statistically significant (p >= 0.1)"
    
    print(f"Statistical significance: {significance_level}")
    
    # Confidence level
    confidence_percentage = (1 - stats_summary['trend_p_value']) * 100
    if confidence_percentage > 99:
        confidence_percentage = 99.9
    print(f"Confidence level: {confidence_percentage:.1f}%")
    
    # Trend interpretation
    if abs(stats_summary['trend_r_value']) > 0.7:
        trend_strength = "Strong"
    elif abs(stats_summary['trend_r_value']) > 0.5:
        trend_strength = "Moderate-to-strong"
    elif abs(stats_summary['trend_r_value']) > 0.3:
        trend_strength = "Moderate"
    elif abs(stats_summary['trend_r_value']) > 0.1:
        trend_strength = "Weak-to-moderate"
    else:
        trend_strength = "Very weak"

    if stats_summary['trend_slope'] > 0:
        trend_direction = "increasing"
    elif stats_summary['trend_slope'] < 0:
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"

    print(f"Trend interpretation: {trend_strength} {trend_direction} trend over time")
    print(f"Effect size (R²): {(stats_summary['trend_r_value']**2)*100:.1f}% of variance explained by time")
    
    # Practical meaning
    if abs(stats_summary['trend_slope']) > 0.001:
        if stats_summary['trend_slope'] > 0:
            print(f"Practical meaning: Time difference increases by {stats_summary['trend_slope']:.3f} seconds per observation")
        else:
            print(f"Practical meaning: Time difference decreases by {abs(stats_summary['trend_slope']):.3f} seconds per observation")
    else:
        print("Practical meaning: Time difference remains essentially constant over time")
    
    print("="*80)
    
    return stats_summary

# Risk analysis for different habit values
# analyze_risk_for_rat_habit('dataset1.csv', 'rat')    # Analyze for 'rat' habit
# analyze_risk_for_rat_habit('dataset1.csv', 'fast')   # Analyze for 'fast' habit  
# analyze_risk_for_rat_habit('dataset1.csv', 'pick')   # Analyze for 'pick' habit
# analyze_risk_for_rat_habit('dataset1_merged.csv', "fast")             # Uses default 'rat' habit

# analyze_risk_for_rat_habit('dataset1_merged.csv', "pick")             # Uses default 'rat' habit
# print(get_unique_habit_values('dataset1_merged.csv'))

# Analyze bat landing to food time over time for 'rat' habit
# analyze_bat_landing_over_time_for_rat_habit('dataset1_merged.csv')

# Analyze time difference between start_time and rat_period_start for habits containing "rat"
analyze_time_difference_over_time('dataset1_merged.csv', 'rat', 'start_time', 'rat_period_start', 
                                 'Start Time to Rat Period Start')

def analyze_numeric_field_over_time_no_filter(csv_path, numeric_field, time_field='start_time', analysis_name=None):
    """
    Analyze and visualize how a numeric field changes over time without any habit filtering.
    
    Args:
        csv_path (str): Path to the CSV file
        numeric_field (str): The numeric field to analyze (e.g., 'seconds_after_rat_arrival')
        time_field (str): The datetime field to use for time axis (default: 'start_time')
        analysis_name (str): Optional custom name for the analysis (auto-generated if None)
    
    Returns:
        dict: Dictionary containing analysis results
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Available columns in dataset: {list(df.columns)}")
    
    # Check if the numeric field exists
    if numeric_field not in df.columns:
        print(f"Error: Column '{numeric_field}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Use all data without filtering
    data = df.copy()
    
    # Convert time field to datetime - use flexible parsing to handle different formats
    try:
        data[time_field] = pd.to_datetime(data[time_field], format='mixed')
    except Exception as e:
        print(f"Error converting datetime field '{time_field}': {e}")
        return None
    
    # Sort by the time field for chronological analysis
    data = data.sort_values(time_field)
    
    # Remove any NaN values in the numeric field
    data = data.dropna(subset=[numeric_field])
    
    print(f"Total data points available: {len(data)}")
    
    # Store original data count for comparison
    original_count = len(data)
    
    # Remove outliers using IQR method
    Q1 = data[numeric_field].quantile(0.25)
    Q3 = data[numeric_field].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds (1.5 * IQR is the standard multiplier)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers_mask = (data[numeric_field] < lower_bound) | (data[numeric_field] > upper_bound)
    outliers = data[outliers_mask]
    
    # Remove outliers
    data = data[~outliers_mask].copy()
    
    # Print outlier removal summary
    outliers_removed = original_count - len(data)
    print(f"\nOUTLIER REMOVAL SUMMARY:")
    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Original data points: {original_count}")
    print(f"Outliers removed: {outliers_removed}")
    print(f"Data points after outlier removal: {len(data)}")
    
    if outliers_removed > 0:
        print(f"Outlier values removed: {sorted(outliers[numeric_field].tolist())}")
    
    # Create analysis name if not provided
    if analysis_name is None:
        analysis_name = f"{numeric_field} Over Time"
    
    # Create the visualization
    plt.figure(figsize=(14, 8)) 
    
    # Create time series plot
    plt.subplot(2, 1, 1)
    
    # Plot connected line through all data points with markers
    plt.plot(data[time_field], data[numeric_field], 
             'o-', alpha=0.8, markersize=5, linewidth=1.5, color='darkblue', 
             markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=1)
    
    # Add trend line using numeric indices for calculation but plot against actual dates
    from scipy import stats
    import numpy as np
    
    # Check if we have enough data points for meaningful statistical analysis
    if len(data) < 3:
        print("Warning: Too few data points for reliable trend analysis")
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
        trend_line = [data[numeric_field].mean()] * len(data)
    else:
        # Use observation order (not actual time differences) for trend analysis
        x_numeric = np.array(range(len(data)))
        y_values = data[numeric_field].values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
        trend_line = slope * x_numeric + intercept
        
        # Double-check p-value calculation with alternative method
        # Calculate t-statistic and degrees of freedom for verification
        n = len(data)
        df = n - 2  # degrees of freedom for linear regression
        
        if std_err > 0:
            t_stat = slope / std_err
            # Two-tailed p-value
            p_value_check = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            print(f"\nSTATISTICAL VALIDATION:")
            print(f"Sample size: {n}")
            print(f"Degrees of freedom: {df}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"Standard error of slope: {std_err:.6f}")
            print(f"P-value from linregress: {p_value:.6f}")
            print(f"P-value verification: {p_value_check:.6f}")
            
            # Use the more precise calculation if they differ significantly
            if abs(p_value - p_value_check) > 0.001:
                print(f"Note: Using verified p-value calculation")
                p_value = p_value_check
    
    plt.plot(data[time_field], trend_line, '--', color='red', linewidth=2, 
             label=f'Trend (slope={slope:.3f})')
    
    plt.title(f'{analysis_name} (All Data - No Filtering)', fontsize=14, fontweight='bold')
    plt.xlabel('Observation Sequence (Chronological Order)')
    plt.ylabel(numeric_field.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    
    # Use a more compact x-axis approach - limit number of date labels to avoid crowding
    import matplotlib.dates as mdates
    ax = plt.gca()
    
    # Set fewer tick locations to reduce gaps appearance
    if len(data) > 20:
        # Show every nth date to avoid overcrowding
        step = max(1, len(data) // 10)
        tick_positions = data[time_field].iloc[::step]
        ax.set_xticks(tick_positions)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    plt.xticks(rotation=45)
    
    # Tighten x-axis limits to reduce visual gaps
    plt.xlim(data[time_field].min(), data[time_field].max())
    
    plt.legend()
    
    # Add text box showing date range and number of records
    date_info = f'Records: {len(data)} entries\nFrom: {data[time_field].min().strftime("%d/%m/%Y %H:%M")}\nTo: {data[time_field].max().strftime("%d/%m/%Y %H:%M")}'
    plt.text(0.02, 0.98, date_info, transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontsize=9)
    
    # Create histogram for distribution
    plt.subplot(2, 1, 2)
    plt.hist(data[numeric_field], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title(f'Distribution of {analysis_name} (All Data - No Filtering)', fontsize=14, fontweight='bold')
    plt.xlabel(numeric_field.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean, median, and mode
    mean_value = data[numeric_field].mean()
    median_value = data[numeric_field].median()
    mode_result = data[numeric_field].mode()
    mode_value = mode_result[0] if len(mode_result) > 0 else median_value  # Use first mode if multiple exist
    
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.axvline(mode_value, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode_value:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats_summary = {
        'total_entries': len(data),
        'mean_value': mean_value,
        'median_value': median_value,
        'mode_value': mode_value,
        'std_value': data[numeric_field].std(),
        'min_value': data[numeric_field].min(),
        'max_value': data[numeric_field].max(),
        'trend_slope': slope,
        'trend_r_value': r_value,
        'trend_p_value': p_value,
        'date_range': {
            'start': data[time_field].min(),
            'end': data[time_field].max()
        },
        'analysis_name': analysis_name,
        'numeric_field': numeric_field
    }
    
    # Print detailed analysis
    print("="*80)
    print(f"{analysis_name.upper()} ANALYSIS (ALL DATA - NO FILTERING)")
    print("="*80)
    print(f"Total entries analyzed: {stats_summary['total_entries']}")
    print(f"Analyzing field: {numeric_field}")
    print(f"Date range: {stats_summary['date_range']['start'].strftime('%d/%m/%Y %H:%M')} to {stats_summary['date_range']['end'].strftime('%d/%m/%Y %H:%M')}")
    print()
    print("DESCRIPTIVE STATISTICS:")
    print(f"Mean {numeric_field}: {stats_summary['mean_value']:.2f}")
    print(f"Median {numeric_field}: {stats_summary['median_value']:.2f}")
    print(f"Mode {numeric_field}: {stats_summary['mode_value']:.2f}")
    print(f"Standard deviation: {stats_summary['std_value']:.2f}")
    print(f"Minimum: {stats_summary['min_value']:.2f}")
    print(f"Maximum: {stats_summary['max_value']:.2f}")
    print()
    print("TREND ANALYSIS:")
    print(f"Trend slope: {stats_summary['trend_slope']:.6f} per observation")
    print(f"Standard error of slope: {std_err:.6f}")
    print(f"Correlation coefficient (r): {stats_summary['trend_r_value']:.4f}")
    print(f"R-squared (coefficient of determination): {stats_summary['trend_r_value']**2:.4f}")
    print(f"P-value: {stats_summary['trend_p_value']:.6f}")
    
    # Statistical significance
    if stats_summary['trend_p_value'] < 0.001:
        significance_level = "highly significant (p < 0.001)"
    elif stats_summary['trend_p_value'] < 0.01:
        significance_level = "very significant (p < 0.01)"
    elif stats_summary['trend_p_value'] < 0.05:
        significance_level = "significant (p < 0.05)"
    elif stats_summary['trend_p_value'] < 0.1:
        significance_level = "marginally significant (p < 0.1)"
    else:
        significance_level = "not statistically significant (p >= 0.1)"
    
    print(f"Statistical significance: {significance_level}")
    
    # Confidence level
    confidence_percentage = (1 - stats_summary['trend_p_value']) * 100
    if confidence_percentage > 99:
        confidence_percentage = 99.9
    print(f"Confidence level: {confidence_percentage:.1f}%")
    
    # Trend interpretation
    if abs(stats_summary['trend_r_value']) > 0.7:
        trend_strength = "Strong"
    elif abs(stats_summary['trend_r_value']) > 0.5:
        trend_strength = "Moderate-to-strong"
    elif abs(stats_summary['trend_r_value']) > 0.3:
        trend_strength = "Moderate"
    elif abs(stats_summary['trend_r_value']) > 0.1:
        trend_strength = "Weak-to-moderate"
    else:
        trend_strength = "Very weak"

    if stats_summary['trend_slope'] > 0:
        trend_direction = "increasing"
    elif stats_summary['trend_slope'] < 0:
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"

    print(f"Trend interpretation: {trend_strength} {trend_direction} trend over time")
    print(f"Effect size (R²): {(stats_summary['trend_r_value']**2)*100:.1f}% of variance explained by time")
    
    # Practical meaning
    if abs(stats_summary['trend_slope']) > 0.001:
        if stats_summary['trend_slope'] > 0:
            print(f"Practical meaning: {numeric_field} increases by {stats_summary['trend_slope']:.3f} per observation")
        else:
            print(f"Practical meaning: {numeric_field} decreases by {abs(stats_summary['trend_slope']):.3f} per observation")
    else:
        print(f"Practical meaning: {numeric_field} remains essentially constant over time")
    
    print("="*80)
    
    return stats_summary

# Analyze seconds_after_rat_arrival over time (no habit filtering)
analyze_numeric_field_over_time_no_filter('dataset1_merged.csv', 'seconds_after_rat_arrival', 'start_time', 
                                          'Seconds After Rat Arrival Over Time')

def analyze_numeric_correlation(csv_path, field1, field2, analysis_name=None):
    """
    Analyze and visualize the correlation between two numeric fields.
    
    Args:
        csv_path (str): Path to the CSV file
        field1 (str): First numeric field (e.g., 'seconds_after_rat_arrival')
        field2 (str): Second numeric field (e.g., 'avg_rat_arrival_number')
        analysis_name (str): Optional custom name for the analysis (auto-generated if None)
    
    Returns:
        dict: Dictionary containing analysis results
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Available columns in dataset: {list(df.columns)}")
    
    # Check if both fields exist
    if field1 not in df.columns:
        print(f"Error: Column '{field1}' not found in dataset")
        return None
    
    if field2 not in df.columns:
        print(f"Error: Column '{field2}' not found in dataset")
        return None
    
    # Use all data without filtering
    data = df[[field1, field2]].copy()
    
    # Remove any NaN values in either field
    data = data.dropna(subset=[field1, field2])
    
    print(f"Total data points available: {len(data)}")
    
    # Store original data count for comparison
    original_count = len(data)
    
    # Remove outliers using IQR method for both fields
    Q1_field1 = data[field1].quantile(0.25)
    Q3_field1 = data[field1].quantile(0.75)
    IQR_field1 = Q3_field1 - Q1_field1
    
    Q1_field2 = data[field2].quantile(0.25)
    Q3_field2 = data[field2].quantile(0.75)
    IQR_field2 = Q3_field2 - Q1_field2
    
    # Define outlier bounds for both fields
    lower_bound_field1 = Q1_field1 - 1.5 * IQR_field1
    upper_bound_field1 = Q3_field1 + 1.5 * IQR_field1
    lower_bound_field2 = Q1_field2 - 1.5 * IQR_field2
    upper_bound_field2 = Q3_field2 + 1.5 * IQR_field2
    
    # Identify outliers (outlier in either field)
    outliers_mask = ((data[field1] < lower_bound_field1) | (data[field1] > upper_bound_field1) |
                     (data[field2] < lower_bound_field2) | (data[field2] > upper_bound_field2))
    outliers = data[outliers_mask]
    
    # Remove outliers
    data = data[~outliers_mask].copy()
    
    # Print outlier removal summary
    outliers_removed = original_count - len(data)
    print(f"\nOUTLIER REMOVAL SUMMARY:")
    print(f"{field1} - Q1: {Q1_field1:.2f}, Q3: {Q3_field1:.2f}, IQR: {IQR_field1:.2f}")
    print(f"{field1} - Lower bound: {lower_bound_field1:.2f}, Upper bound: {upper_bound_field1:.2f}")
    print(f"{field2} - Q1: {Q1_field2:.2f}, Q3: {Q3_field2:.2f}, IQR: {IQR_field2:.2f}")
    print(f"{field2} - Lower bound: {lower_bound_field2:.2f}, Upper bound: {upper_bound_field2:.2f}")
    print(f"Original data points: {original_count}")
    print(f"Outliers removed: {outliers_removed}")
    print(f"Data points after outlier removal: {len(data)}")
    
    # Create analysis name if not provided
    if analysis_name is None:
        analysis_name = f"{field1} vs {field2}"
    
    # Create the visualization
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(data[field1], data[field2], alpha=0.6, color='darkblue', s=30)
    plt.xlabel(field1.replace('_', ' ').title())
    plt.ylabel(field2.replace('_', ' ').title())
    plt.title(f'Scatter Plot: {analysis_name}', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Calculate correlation and add trend line
    correlation = data[field1].corr(data[field2])
    from scipy import stats
    import numpy as np
    
    if len(data) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[field1], data[field2])
        trend_line = slope * data[field1] + intercept
        plt.plot(data[field1], trend_line, '--', color='red', linewidth=2, alpha=0.8)
        
        # Add correlation info box
        plt.text(0.05, 0.95, f'r = {correlation:.3f}\np = {p_value:.6f}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
    else:
        slope, intercept, r_value, p_value, std_err = 0, 0, correlation, 1, 0
    
    # Create histogram for field1
    plt.subplot(2, 2, 2)
    plt.hist(data[field1], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title(f'Distribution of {field1.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.xlabel(field1.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add mean, median, mode lines
    mean1 = data[field1].mean()
    median1 = data[field1].median()
    mode1_result = data[field1].mode()
    mode1 = mode1_result[0] if len(mode1_result) > 0 else median1
    
    plt.axvline(mean1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean1:.2f}')
    plt.axvline(median1, color='green', linestyle='--', linewidth=2, label=f'Median: {median1:.2f}')
    plt.axvline(mode1, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode1:.2f}')
    plt.legend(fontsize=8)
    
    # Create histogram for field2
    plt.subplot(2, 2, 3)
    plt.hist(data[field2], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title(f'Distribution of {field2.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.xlabel(field2.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add mean, median, mode lines
    mean2 = data[field2].mean()
    median2 = data[field2].median()
    mode2_result = data[field2].mode()
    mode2 = mode2_result[0] if len(mode2_result) > 0 else median2
    
    plt.axvline(mean2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean2:.2f}')
    plt.axvline(median2, color='green', linestyle='--', linewidth=2, label=f'Median: {median2:.2f}')
    plt.axvline(mode2, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode2:.2f}')
    plt.legend(fontsize=8)
    
    # Create joint distribution (2D density plot)
    plt.subplot(2, 2, 4)
    plt.hist2d(data[field1], data[field2], bins=15, cmap='Blues', alpha=0.7)
    plt.colorbar(label='Count')
    plt.xlabel(field1.replace('_', ' ').title())
    plt.ylabel(field2.replace('_', ' ').title())
    plt.title('Joint Distribution (2D Histogram)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats_summary = {
        'total_entries': len(data),
        'field1_mean': mean1,
        'field1_median': median1,
        'field1_mode': mode1,
        'field1_std': data[field1].std(),
        'field1_min': data[field1].min(),
        'field1_max': data[field1].max(),
        'field2_mean': mean2,
        'field2_median': median2,
        'field2_mode': mode2,
        'field2_std': data[field2].std(),
        'field2_min': data[field2].min(),
        'field2_max': data[field2].max(),
        'correlation_r': correlation,
        'regression_slope': slope,
        'regression_p_value': p_value,
        'regression_r_value': r_value,
        'analysis_name': analysis_name,
        'field1_name': field1,
        'field2_name': field2
    }
    
    # Print detailed analysis
    print("="*80)
    print(f"{analysis_name.upper()} CORRELATION ANALYSIS (ALL DATA - NO FILTERING)")
    print("="*80)
    print(f"Total entries analyzed: {stats_summary['total_entries']}")
    print(f"Analyzing: {field1} vs {field2}")
    print()
    print(f"DESCRIPTIVE STATISTICS - {field1.upper()}:")
    print(f"Mean {field1}: {stats_summary['field1_mean']:.2f}")
    print(f"Median {field1}: {stats_summary['field1_median']:.2f}")
    print(f"Mode {field1}: {stats_summary['field1_mode']:.2f}")
    print(f"Standard deviation: {stats_summary['field1_std']:.2f}")
    print(f"Minimum: {stats_summary['field1_min']:.2f}")
    print(f"Maximum: {stats_summary['field1_max']:.2f}")
    print()
    print(f"DESCRIPTIVE STATISTICS - {field2.upper()}:")
    print(f"Mean {field2}: {stats_summary['field2_mean']:.2f}")
    print(f"Median {field2}: {stats_summary['field2_median']:.2f}")
    print(f"Mode {field2}: {stats_summary['field2_mode']:.2f}")
    print(f"Standard deviation: {stats_summary['field2_std']:.2f}")
    print(f"Minimum: {stats_summary['field2_min']:.2f}")
    print(f"Maximum: {stats_summary['field2_max']:.2f}")
    print()
    print("CORRELATION ANALYSIS:")
    print(f"Correlation coefficient (r): {stats_summary['correlation_r']:.4f}")
    print(f"R-squared (coefficient of determination): {stats_summary['correlation_r']**2:.4f}")
    print(f"Regression slope: {stats_summary['regression_slope']:.6f}")
    print(f"P-value: {stats_summary['regression_p_value']:.6f}")
    
    # Statistical significance
    if stats_summary['regression_p_value'] < 0.001:
        significance_level = "highly significant (p < 0.001)"
    elif stats_summary['regression_p_value'] < 0.01:
        significance_level = "very significant (p < 0.01)"
    elif stats_summary['regression_p_value'] < 0.05:
        significance_level = "significant (p < 0.05)"
    elif stats_summary['regression_p_value'] < 0.1:
        significance_level = "marginally significant (p < 0.1)"
    else:
        significance_level = "not statistically significant (p >= 0.1)"
    
    print(f"Statistical significance: {significance_level}")
    
    # Confidence level
    confidence_percentage = (1 - stats_summary['regression_p_value']) * 100
    if confidence_percentage > 99:
        confidence_percentage = 99.9
    print(f"Confidence level: {confidence_percentage:.1f}%")
    
    # Correlation interpretation
    if abs(stats_summary['correlation_r']) > 0.7:
        correlation_strength = "Strong"
    elif abs(stats_summary['correlation_r']) > 0.5:
        correlation_strength = "Moderate-to-strong"
    elif abs(stats_summary['correlation_r']) > 0.3:
        correlation_strength = "Moderate"
    elif abs(stats_summary['correlation_r']) > 0.1:
        correlation_strength = "Weak-to-moderate"
    else:
        correlation_strength = "Very weak"

    if stats_summary['correlation_r'] > 0:
        correlation_direction = "positive"
    elif stats_summary['correlation_r'] < 0:
        correlation_direction = "negative"
    else:
        correlation_direction = "no"

    print(f"Correlation interpretation: {correlation_strength} {correlation_direction} correlation")
    print(f"Effect size (R²): {(stats_summary['correlation_r']**2)*100:.1f}% of variance explained")
    
    # Practical meaning
    if abs(stats_summary['correlation_r']) > 0.1:
        if stats_summary['correlation_r'] > 0:
            print(f"Practical meaning: As {field1} increases, {field2} tends to increase")
        else:
            print(f"Practical meaning: As {field1} increases, {field2} tends to decrease")
    else:
        print(f"Practical meaning: Little to no linear relationship between {field1} and {field2}")
    
    print("="*80)
    
    return stats_summary

# Analyze correlation between seconds_after_rat_arrival and avg_rat_arrival_number
analyze_numeric_correlation('dataset1_merged.csv', 'seconds_after_rat_arrival', 'avg_rat_arrival_number', 
                            'Seconds After Rat Arrival vs Avg Rat Arrival Number')