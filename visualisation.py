import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_correlation_chart(csv_path, param1, param2, check_outlier=False):
    """
    Plots a correlation chart between two parameters from a CSV file.
    Optionally checks and marks outliers.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing values in the selected columns
    data = df[[param1, param2]].dropna()

    # Outlier detection (using Z-score)
    if check_outlier:
        z_scores = stats.zscore(data)
        abs_z_scores = abs(z_scores)
        outliers = (abs_z_scores > 3).any(axis=1)
        data['Outlier'] = outliers
        palette = {False: 'blue', True: 'red'}
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x=param1, y=param2, hue='Outlier', palette=palette)
        plt.title(f'Correlation between {param1} and {param2} (Outliers Highlighted)')
    else:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x=param1, y=param2)
        plt.title(f'Correlation between {param1} and {param2}')
    
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_correlation_chart('dataset1.csv', 'bat_landing_to_food', 'seconds_after_rat_arrival', check_outlier=False)
