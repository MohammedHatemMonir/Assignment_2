import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load the edited dataset with motivation column
df = pd.read_csv('dataset2_edited.csv')

print("=== BAT LANDING vs MOTIVATION ANALYSIS ===\n")
print(f"Original dataset shape: {df.shape}")

# Focus on bat_landing_number and motivation columns
bat_landing = df['bat_landing_number']
motivation = df['motivation']

print(f"Initial data points: {len(df)}")

# Remove outliers using IQR method
def remove_outliers_iqr(data, column_name):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    print(f"{column_name} outliers removed: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
    print(f"  Outlier range: < {lower_bound:.3f} or > {upper_bound:.3f}")
    
    return clean_data.index

# Remove outliers for both variables
bat_clean_indices = remove_outliers_iqr(bat_landing, "Bat landing")
motivation_clean_indices = remove_outliers_iqr(motivation, "Motivation")

# Get intersection of clean indices (data points that are clean in both variables)
clean_indices = bat_clean_indices.intersection(motivation_clean_indices)
df_clean = df.loc[clean_indices].copy()

print(f"\nFinal clean dataset: {len(df_clean)} data points ({len(df_clean)/len(df)*100:.2f}% of original)")

# Extract clean data for analysis
bat_clean = df_clean['bat_landing_number']
motivation_clean = df_clean['motivation']

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS:")
print("="*60)

# Bat landing statistics
print("\nBAT LANDING STATISTICS:")
print(f"Mean bat landing number: {bat_clean.mean():.2f}")
print(f"Median bat landing number: {bat_clean.median():.2f}")
print(f"Mode bat landing number: {bat_clean.mode().iloc[0]:.2f}" if len(bat_clean.mode()) > 0 else "Mode: No unique mode")
print(f"Standard deviation: {bat_clean.std():.2f}")
print(f"Minimum number: {bat_clean.min():.2f}")
print(f"Maximum number: {bat_clean.max():.2f}")

# Motivation statistics
print("\nMOTIVATION STATISTICS:")
print(f"Mean motivation: {motivation_clean.mean():.2f}")
print(f"Median motivation: {motivation_clean.median():.2f}")
print(f"Mode motivation: {motivation_clean.mode().iloc[0]:.2f}" if len(motivation_clean.mode()) > 0 else "Mode: No unique mode")
print(f"Standard deviation: {motivation_clean.std():.2f}")
print(f"Minimum motivation: {motivation_clean.min():.2f}")
print(f"Maximum motivation: {motivation_clean.max():.2f}")

# Correlation and trend analysis
correlation_coeff, p_value = pearsonr(bat_clean, motivation_clean)
r_squared = correlation_coeff**2

# Linear regression for trend analysis
slope, intercept, r_value, p_val, std_err = stats.linregress(bat_clean, motivation_clean)

print("\n" + "="*60)
print("TREND ANALYSIS:")
print("="*60)
print(f"Trend slope: {slope:.6f} motivation units per bat landing")
print(f"Standard error of slope: {std_err:.6f}")
print(f"Correlation coefficient (r): {correlation_coeff:.4f}")
print(f"R-squared (coefficient of determination): {r_squared:.4f}")
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
if abs(correlation_coeff) < 0.1:
    trend_strength = "negligible"
elif abs(correlation_coeff) < 0.3:
    trend_strength = "weak"
elif abs(correlation_coeff) < 0.5:
    trend_strength = "weak-to-moderate"
elif abs(correlation_coeff) < 0.7:
    trend_strength = "moderate"
elif abs(correlation_coeff) < 0.9:
    trend_strength = "strong"
else:
    trend_strength = "very strong"

trend_direction = "increasing" if slope > 0 else "decreasing"
print(f"Trend interpretation: {trend_strength.capitalize()} {trend_direction} trend")
print(f"Effect size (R²): {r_squared*100:.1f}% of variance explained by bat landings")
print(f"Practical meaning: Motivation changes by {slope:.6f} units per additional bat landing")

# Create visualization - Bar chart showing average landings per motivation level
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create bins for motivation levels and calculate average bat landings for each bin
motivation_bins = pd.cut(motivation_clean, bins=10)
grouped_data = df_clean.groupby(motivation_bins)['bat_landing_number'].mean().reset_index()

# Extract bin centers for x-axis labels
bin_centers = []
bin_labels = []
for interval in grouped_data['motivation']:
    center = (interval.left + interval.right) / 2
    bin_centers.append(center)
    bin_labels.append(f'{interval.left:.2f}-{interval.right:.2f}')

# Create the bar chart
bars = ax.bar(range(len(grouped_data)), grouped_data['bat_landing_number'], 
              color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

# Customize the chart
ax.set_xlabel('Estimated Motivation Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Bat Landings per Motivation', fontsize=12, fontweight='bold')
ax.set_title('Average Bat Landings by Motivation Level', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(grouped_data)))
ax.set_xticklabels(bin_labels, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Add correlation info as text box
textstr = f'Correlation: r = {correlation_coeff:.3f}\nR² = {r_squared:.3f}\nP-value = {p_value:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('bat_motivation_bar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('bat_motivation_bar_chart.pdf', bbox_inches='tight')
plt.show()

print(f"\n" + "="*60)
print("VISUALIZATION SUMMARY:")
print("="*60)
print("✓ Bar chart showing average bat landings by motivation level created")
print("✓ Chart saved as 'bat_motivation_bar_chart.png' and 'bat_motivation_bar_chart.pdf'")

# Additional summary statistics
print(f"\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"• Dataset contains {len(df_clean):,} clean observations")
print(f"• Correlation between bat landings and motivation: {correlation_coeff:.4f}")
print(f"• {r_squared*100:.1f}% of motivation variance explained by bat landings")
print(f"• Relationship is statistically {significance}")
print(f"• For every additional bat landing, motivation changes by {slope:.6f} units")


