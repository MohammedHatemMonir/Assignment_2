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

# Example usage:
# Original correlation chart
plot_correlation_chart('dataset1.csv', 'risk', 'reward', check_outlier=True)

# New binary vs categorical analysis
# plot_binary_vs_categorical('dataset1.csv', 'risk', 'habit')
# plot_binary_vs_categorical('dataset1.csv', 'reward', 'habit')
