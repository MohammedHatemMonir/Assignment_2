import os
import sys
import math
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
	# scipy provides p-value and standard error easily
	from scipy import stats as scistats
	_HAVE_SCIPY = True
except Exception:
	_HAVE_SCIPY = False


def iqr_filter(df: pd.DataFrame, cols: Tuple[str, str], k: float = 1.5) -> pd.DataFrame:
	"""
	Remove outliers from the provided columns using the IQR rule.
	Keeps rows where BOTH column values lie within their respective [Q1 - k*IQR, Q3 + k*IQR] fences.
	"""
	filtered = df.copy()
	masks = []
	for c in cols:
		q1 = filtered[c].quantile(0.25)
		q3 = filtered[c].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - k * iqr
		upper = q3 + k * iqr
		masks.append((filtered[c] >= lower) & (filtered[c] <= upper))
	mask_all = np.logical_and.reduce(masks)
	return filtered[mask_all]


def descriptive_stats(series: pd.Series) -> Dict[str, float]:
	s = series.dropna()
	stats: Dict[str, float] = {}
	stats["mean"] = float(s.mean()) if len(s) else float("nan")
	stats["median"] = float(s.median()) if len(s) else float("nan")
	mode_vals = s.mode()
	stats["mode"] = float(mode_vals.iloc[0]) if len(mode_vals) else float("nan")
	stats["std"] = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
	stats["min"] = float(s.min()) if len(s) else float("nan")
	stats["max"] = float(s.max()) if len(s) else float("nan")
	return stats


def regression_xy(x: pd.Series, y: pd.Series):
	"""
	Linear regression y ~ a + b*x.
	Returns: slope, intercept, r, r2, pvalue, stderr
	Fallback without SciPy: slope, intercept via polyfit; r via corr; pvalue/stderr may be None.
	"""
	x = pd.to_numeric(x, errors="coerce").astype(float)
	y = pd.to_numeric(y, errors="coerce").astype(float)
	mask = x.notna() & y.notna()
	x = x[mask]
	y = y[mask]
	if len(x) < 2:
		return math.nan, math.nan, math.nan, math.nan, None, None

	if _HAVE_SCIPY:
		res = scistats.linregress(x.values, y.values)
		slope = float(res.slope)
		intercept = float(res.intercept)
		r = float(res.rvalue)
		r2 = r * r
		pvalue = float(res.pvalue)
		stderr = float(res.stderr)
		return slope, intercept, r, r2, pvalue, stderr
	else:
		# Fallbacks without SciPy
		slope, intercept = np.polyfit(x.values, y.values, 1)
		r = float(pd.Series(x).corr(pd.Series(y)))
		r2 = r * r
		return float(slope), float(intercept), r, r2, None, None


def significance_label(p: float) -> Tuple[str, str]:
	if p is None or math.isnan(p):
		return ("unknown", "N/A")
	if p < 0.001:
		return ("highly significant (p < 0.001)", "99.9%")
	if p < 0.01:
		return ("very significant (p < 0.01)", "99%")
	if p < 0.05:
		return ("significant (p < 0.05)", "95%")
	return ("not significant (p >= 0.05)", "<95%")


def compare_and_plot(df: pd.DataFrame, x_col: str, y_col: str,
					 title: str, x_label: str, y_label: str,
					 y_unit: str = "") -> None:
	"""
	Clean data (numeric + IQR), compute stats and linear trend, print a report, and show a scatter plot.
	"""
	# Validate columns
	for c in (x_col, y_col):
		if c not in df.columns:
			print(f"ERROR: Missing required column: {c}")
			return

	work = df.copy()
	work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
	work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
	work = work[[x_col, y_col]].dropna()

	# Outlier handling via IQR
	df_filtered = iqr_filter(work, (x_col, y_col), k=1.5)

	# Compute descriptive stats for y
	stats_y = descriptive_stats(df_filtered[y_col])

	# Regression and correlation between x and y
	slope, intercept, r, r2, pvalue, stderr = regression_xy(df_filtered[x_col], df_filtered[y_col])
	signif, conf_level = significance_label(pvalue if pvalue is not None else float("nan"))

	# Units suffix for printing
	unit = y_unit

	# Print report
	print(f"DESCRIPTIVE STATISTICS for {y_col}:")
	print(f"Mean: {stats_y['mean']:.2f}{unit}")
	print(f"Median: {stats_y['median']:.2f}{unit}")
	mode_val = stats_y['mode']
	mode_str = f"{mode_val:.2f}{unit}" if not math.isnan(mode_val) else "N/A"
	print(f"Mode: {mode_str}")
	if not math.isnan(stats_y['std']):
		print(f"Standard deviation: {stats_y['std']:.2f}{unit}")
	else:
		print("Standard deviation: N/A")
	print(f"Minimum: {stats_y['min']:.2f}{unit}")
	print(f"Maximum: {stats_y['max']:.2f}{unit}")
	print()

	print(f"TREND ANALYSIS ({x_col} vs {y_col}):")
	if not math.isnan(slope):
		per_x_unit = f"{unit} per unit of {x_col}" if unit else f"per unit of {x_col}"
		print(f"Trend slope: {slope:.6f} {per_x_unit}")
	else:
		print("Trend slope: N/A")
	if stderr is not None:
		print(f"Standard error of slope: {stderr:.6f}")
	else:
		print("Standard error of slope: N/A")
	if not math.isnan(r):
		print(f"Correlation coefficient (r): {r:.4f}")
		print(f"R-squared (coefficient of determination): {r2:.4f}")
	else:
		print("Correlation coefficient (r): N/A")
		print("R-squared (coefficient of determination): N/A")
	if pvalue is not None:
		print(f"P-value: {pvalue:.6f}")
	else:
		print("P-value: N/A")
	print(f"Statistical significance: {signif}")
	print(f"Confidence level: {conf_level}")

	trend_direction = "increasing" if not math.isnan(slope) and slope > 0 else ("decreasing" if not math.isnan(slope) and slope < 0 else "flat")
	strength = "weak" if not math.isnan(r2) and r2 < 0.06 else ("moderate" if not math.isnan(r2) and r2 < 0.25 else "strong")
	print(f"Trend interpretation: {strength.capitalize()} {trend_direction} relationship")
	if not math.isnan(r2):
		print(f"Effect size (R²): {r2*100:.1f}% of variance in {y_col} explained by {x_col}")
	else:
		print("Effect size (R²): N/A")
	if not math.isnan(slope):
		if unit:
			print(f"Practical meaning: {y_col} changes by {slope:.3f}{unit} per 1 unit increase in {x_col}")
		else:
			print(f"Practical meaning: {y_col} changes by {slope:.3f} per 1 unit increase in {x_col}")
	else:
		print("Practical meaning: N/A")
	print()

	# Plot scatter with regression line
	x = df_filtered[x_col]
	y = df_filtered[y_col]
	plt.figure(figsize=(8, 6))
	plt.scatter(x, y, alpha=0.6, edgecolor="k", linewidth=0.3)

	if not math.isnan(slope) and not math.isnan(intercept):
		xs = np.linspace(x.min(), x.max(), 100)
		ys = slope * xs + intercept
		plt.plot(xs, ys, color="red", linewidth=2, label="Linear fit")

	if not math.isnan(r):
		annot = f"r={r:.2f}, R²={r2:.2f}"
		if pvalue is not None:
			annot += f", p={pvalue:.3g}"
		plt.text(0.02, 0.98, annot, transform=plt.gca().transAxes,
				 va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(loc="best")
	plt.tight_layout()
	plt.show()


def compare_and_bar(df: pd.DataFrame, x_col: str, y_col: str,
					title: str, x_label: str, y_label: str,
					agg: str = "mean") -> None:
	"""
	Build a bar chart of y aggregated per rounded integer of x (IQR outliers removed).
	"""
	# Validate columns
	for c in (x_col, y_col):
		if c not in df.columns:
			print(f"ERROR: Missing required column: {c}")
			return

	work = df.copy()
	work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
	work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
	work = work[[x_col, y_col]].dropna()

	# Remove outliers using IQR before grouping
	work = iqr_filter(work, (x_col, y_col), k=1.5)

	# Round x to nearest integer for grouping
	work["x_int"] = work[x_col].round().astype(int)

	if agg not in ("mean", "sum"):
		agg = "mean"

	grouped = (
		work.groupby("x_int")
		.agg(
			count_records=(y_col, "count"),
			sum_y=(y_col, "sum"),
			mean_y=(y_col, "mean"),
			sum_x=(x_col, "sum"),
			mean_x=(x_col, "mean"),
		)
		.reset_index()
		.sort_values("x_int")
	)

	if grouped.empty:
		print("No data available for bar chart after cleaning.")
		return

	# Print grouped metrics
	print(f"Grouped metrics by rounded {x_col}:")
	for _, row in grouped.iterrows():
		rat_lvl = int(row["x_int"])
		count = int(row["count_records"])
		sum_rats = row["sum_x"]
		sum_bats = row["sum_y"]
		avg_bats = row["mean_y"]
		print(
			f"  {x_col}={rat_lvl}: records={count}, sum_{x_col}={sum_rats:.2f}, "
			f"sum_{y_col}={sum_bats:.2f}, avg_{y_col}={avg_bats:.2f}"
		)
	print()

	# Plot bar chart (show averages on the bars)
	fig, ax = plt.subplots(figsize=(9, 6))
	# Bars show means by default per user request
	bar_series = grouped["mean_y"] if agg == "mean" else grouped["sum_y"]
	bars = ax.bar(grouped["x_int"].astype(str), bar_series, color="#4C78A8")

	# Annotate each bar with the average bat landings (these are the numbers user wants on the graph)
	for i, rect in enumerate(bars):
		avg_val = grouped["mean_y"].iloc[i]
		ax.text(
			rect.get_x() + rect.get_width() / 2.0,
			rect.get_height(),
			f"{avg_val:.2f}",
			ha="center",
			va="bottom",
			fontsize=9,
			color="#222",
		)

	ax.set_title(title)
	ax.set_xlabel(x_label + " (rounded to nearest integer)")
	ax.set_ylabel(y_label)
	fig.tight_layout()
	plt.show()


def boxplot_by_group(
	df: pd.DataFrame,
	value_col: str,
	group_col: str,
	title: str,
	y_label: str,
	group_values=(1, 2, 3),
	round_group: bool = True,
) -> None:
	"""
	Boxplot of `value_col` grouped by (rounded) integer levels of `group_col`.
	Also performs ANOVA test to check for significant differences between groups.
	Default groups shown: 1, 2, 3 (for avg_rat_arrival_number).
	"""
	for c in (value_col, group_col):
		if c not in df.columns:
			print(f"ERROR: Missing required column: {c}")
			return

	work = df.copy()
	work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
	work[group_col] = pd.to_numeric(work[group_col], errors="coerce")
	work = work[[value_col, group_col]].dropna()

	# Round group to nearest integer if required (to collapse 1.5 -> 2 etc.)
	if round_group:
		work["group_lvl"] = work[group_col].round().astype("Int64")
	else:
		work["group_lvl"] = work[group_col].astype("Int64")

	# Keep only requested group values
	work = work[work["group_lvl"].isin(list(group_values))]

	if work.empty:
		print("No data available for boxplot after cleaning.")
		return

	# Prepare data per group
	data = []
	labels = []
	counts = []
	for g in group_values:
		vals = work.loc[work["group_lvl"] == g, value_col].dropna().values
		if len(vals) > 0:
			data.append(vals)
			labels.append(str(g))
			counts.append(len(vals))

	# Print counts per group for transparency
	print(f"DESCRIPTIVE STATISTICS for {value_col} by {group_col}:")
	for lab, cnt in zip(labels, counts):
		print(f"  Group {group_col}={lab}: n={cnt}")
	print()

	# Perform ANOVA test if SciPy is available and there are multiple groups
	if _HAVE_SCIPY and len(data) > 1:
		f_stat, p_value = scistats.f_oneway(*data)
		signif, conf_level = significance_label(p_value)
		print("ANOVA TEST (DIFFERENCE IN MEANS):")
		print(f"F-statistic: {f_stat:.4f}")
		print(f"P-value: {p_value:.6f}")
		print(f"Statistical significance: {signif}")
		print(f"Confidence level: {conf_level}")
		if p_value < 0.05:
			print("Interpretation: There is a statistically significant difference in the means of at least two groups.")
		else:
			print("Interpretation: There is no statistically significant difference between the group means.")
		print()

	plt.figure(figsize=(8, 6))
	plt.boxplot(data, labels=labels, showfliers=False)
	plt.title(title)
	plt.xlabel(group_col)
	plt.ylabel(y_label)
	plt.tight_layout()
	plt.show()


def bar_chart_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    Build a bar chart of the mean of `value_col` for each category in `category_col`.
    Also performs a t-test if there are exactly two groups.
    """
    for c in (category_col, value_col):
        if c not in df.columns:
            print(f"ERROR: Missing required column: {c}")
            return

    work = df.copy()
    work[category_col] = pd.to_numeric(work[category_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[[category_col, value_col]].dropna()

    # Apply IQR filtering on the value column within each category group
    def iqr_filter_by_group(g: pd.DataFrame) -> pd.DataFrame:
        v = g[value_col].dropna()
        if len(v) < 4:
            return g
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return g[(g[value_col] >= lower) & (g[value_col] <= upper)]

    work = work.groupby(category_col, group_keys=False).apply(iqr_filter_by_group)

    # Group by the categorical column
    grouped = work.groupby(category_col).agg(
        count=(value_col, "count"),
        mean_val=(value_col, "mean")
    ).reset_index()

    if grouped.empty:
        print(f"No data available for bar chart of {value_col} by {category_col}.")
        return

    print(f"DESCRIPTIVE STATISTICS for {value_col} by {category_col}:")
    for _, row in grouped.iterrows():
        category = row[category_col]
        count = int(row["count"])
        mean_val = row["mean_val"]
        print(f"  Group {category_col}={category}: n={count}, Mean {value_col}={mean_val:.2f}")
    print()

    # Perform t-test if there are exactly two groups
    groups = [g[value_col].dropna().values for name, g in work.groupby(category_col)]
    if _HAVE_SCIPY and len(groups) == 2:
        t_stat, p_value = scistats.ttest_ind(groups[0], groups[1], equal_var=False) # Welch's t-test
        signif, conf_level = significance_label(p_value)
        print("T-TEST (DIFFERENCE IN MEANS):")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Statistical significance: {signif}")
        print(f"Confidence level: {conf_level}")
        if p_value < 0.05:
            print("Interpretation: There is a statistically significant difference between the group means.")
        else:
            print("Interpretation: There is no statistically significant difference between the group means.")
        print()


    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(grouped[category_col].astype(str), grouped["mean_val"], color=["#4C78A8", "#59A14F"])

    for i, rect in enumerate(bars):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    plt.show()


def motivation_landing_time_by_risk(df: pd.DataFrame):
	"""
	Generates three bar charts for motivation vs. bat landing time,
	separated by risk category (0, 1, and both).
	"""

	def plot_motivation_vs_landing_time(df_in: pd.DataFrame, title: str):
		"""
		Helper function to create a bar chart of average bat landing time vs. motivation level.
		Removes outliers from bat_landing_to_food using IQR within each motivation group.
		"""
		work = df_in.copy()
		# Ensure derived columns
		if "motevation" not in work.columns:
			fa2 = pd.to_numeric(work.get("food_availability"), errors="coerce")
			rn2 = pd.to_numeric(work.get("avg_rat_arrival_number"), errors="coerce")
			work["motevation"] = fa2 / rn2
			work.loc[~np.isfinite(work["motevation"]), "motevation"] = np.nan
		
		# Numeric coercion
		work["motevation"] = pd.to_numeric(work["motevation"], errors="coerce")
		work["bat_landing_to_food"] = pd.to_numeric(work.get("bat_landing_to_food"), errors="coerce")
		work = work.dropna(subset=["motevation", "bat_landing_to_food"]).copy()
		
		if work.empty:
			print(f"No data available for '{title}' after filtering.")
			return
			
		# Group by rounded motivation level
		work["motivation_level"] = work["motevation"].round().astype("Int64")
		if work["motivation_level"].isna().all():
			print(f"No valid motivation levels to group by for '{title}'.")
			return
			
		# Apply IQR filtering ONLY on bat_landing_to_food within each motivation group
		def filter_decision_iqr(g: pd.DataFrame) -> pd.DataFrame:
			y = g["bat_landing_to_food"].dropna()
			if len(y) < 4:
				return g
			q1 = y.quantile(0.25)
			q3 = y.quantile(0.75)
			iqr = q3 - q1
			lower = q1 - 1.5 * iqr
			upper = q3 + 1.5 * iqr
			return g[(g["bat_landing_to_food"] >= lower) & (g["bat_landing_to_food"] <= upper)]

		cleaned = work.groupby("motivation_level", dropna=True, group_keys=False).apply(filter_decision_iqr)
		
		if cleaned.empty:
			print(f"No data left for '{title}' after bat_landing_to_food IQR filtering.")
			return
			
		# Compute averages and descriptive stats
		grouped = (
			cleaned.groupby("motivation_level")
			.agg(
				count=("bat_landing_to_food", "count"),
				avg_bltf=("bat_landing_to_food", "mean"),
				std_bltf=("bat_landing_to_food", "std"),
				min_bltf=("bat_landing_to_food", "min"),
				max_bltf=("bat_landing_to_food", "max"),
			)
			.reset_index()
			.sort_values("motivation_level")
		)
		
		print(f"--- DESCRIPTIVE STATISTICS for {title} ---")
		print("Average bat_landing_to_food by motivation level (outliers removed via IQR):")
		for _, r in grouped.iterrows():
			lvl = int(r["motivation_level"]) if pd.notna(r["motivation_level"]) else "NaN"
			std_val = f"{r['std_bltf']:.2f}" if pd.notna(r['std_bltf']) else "N/A"
			print(f"  Motivation {lvl}: n={int(r['count'])}, mean={r['avg_bltf']:.2f}s, std={std_val}, min={r['min_bltf']:.2f}, max={r['max_bltf']:.2f}")
		print("-" * (len(title) + 30))
		print()
		
		# Bar chart
		fig, ax = plt.subplots(figsize=(10, 6))
		x_vals = grouped["motivation_level"].astype(int).astype(str)
		bars = ax.bar(x_vals, grouped["avg_bltf"], color="#59A14F")
		
		for i, rect in enumerate(bars):
			avg_val = grouped["avg_bltf"].iloc[i]
			ax.text(
				rect.get_x() + rect.get_width() / 2.0,
				rect.get_height(),
				f"{avg_val:.1f}",
				ha="center",
				va="bottom",
				fontsize=9,
				color="#222",
			)
			
		ax.set_title(title)
		ax.set_xlabel("Motivation level (rounded to nearest integer)")
		ax.set_ylabel("Average bat landing to food (seconds)")
		fig.tight_layout()
		plt.show()

	# 1. Plot for all data (risk 0 and 1)
	plot_motivation_vs_landing_time(df, "Motivation vs. Landing Time (All Risk Levels)")

	# 2. Plot for risk = 0 (Risk-Avoidance)
	df_risk0 = df[df["risk"] == 0].copy()
	plot_motivation_vs_landing_time(df_risk0, "Motivation vs. Landing Time (Risk-Avoidance Only)")

	# 3. Plot for risk = 1 (Risk-Taking)
	df_risk1 = df[df["risk"] == 1].copy()
	plot_motivation_vs_landing_time(df_risk1, "Motivation vs. Landing Time (Risk-Taking Only)")


def analyze_reward_at_low_motivation(df: pd.DataFrame):
    """
    Analyzes and visualizes reward outcomes when motivation is near zero and risk is high.
    Specifically, it filters for records where risk is 1 and rounded motivation is 0,
    then creates a bar chart of reward counts (0 vs 1).
    """
    work = df.copy()

    # Ensure motivation column exists
    if "motevation" not in work.columns:
        fa = pd.to_numeric(work.get("food_availability"), errors="coerce")
        rn = pd.to_numeric(work.get("avg_rat_arrival_number"), errors="coerce")
        work["motevation"] = fa / rn
        work.loc[~np.isfinite(work["motevation"]), "motevation"] = np.nan

    # Ensure required columns are numeric
    work["motevation"] = pd.to_numeric(work["motevation"], errors="coerce")
    work["risk"] = pd.to_numeric(work["risk"], errors="coerce")
    work["reward"] = pd.to_numeric(work["reward"], errors="coerce")
    work = work.dropna(subset=["motevation", "risk", "reward"]).copy()

    # Filter for the specific condition: risk=1 and motivation rounds to 0
    work["motivation_level"] = work["motevation"].round().astype("Int64")
    filtered_df = work[(work["risk"] == 1) & (work["motivation_level"] == 0)]

    if filtered_df.empty:
        print("No data available for the condition: risk=1 and motivation near 0.")
        return

    # Calculate statistics: counts and percentages of reward
    reward_counts = filtered_df["reward"].value_counts().sort_index()
    total_count = len(filtered_df)
    
    print("--- REWARD ANALYSIS (Risk=1, Motivation ≈ 0) ---")
    print(f"Total observations for this condition: {total_count}")
    
    for reward_val, count in reward_counts.items():
        percentage = (count / total_count) * 100
        reward_label = "Not Rewarding" if reward_val == 0 else "Rewarding"
        print(f"  Reward = {int(reward_val)} ({reward_label}): n={count} ({percentage:.1f}%)")
    print("-" * 50)
    print()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for plotting
    labels = reward_counts.index.map({0: "Not Rewarding (0)", 1: "Rewarding (1)"}).tolist()
    counts = reward_counts.values
    
    bars = ax.bar(labels, counts, color=["#E15759", "#59A14F"])

    # Add counts on top of bars
    for i, rect in enumerate(bars):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"n={int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Reward Outcome for High-Risk, Low-Motivation Foraging")
    ax.set_xlabel("Reward Outcome")
    ax.set_ylabel("Number of Observations")
    ax.set_ylim(0, max(counts) * 1.15) # Add some space at the top
    fig.tight_layout()
    plt.show()


def main():
	# Inputs
	default_csv = os.path.join(os.path.dirname(__file__), "dataset1_merged.csv")
	csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

	if not os.path.exists(csv_path):
		print(f"ERROR: Data file not found: {csv_path}")
		sys.exit(1)

	# Load once
	df = pd.read_csv(csv_path)

	# # Ensure derived columns exist (compute defensively if missing)
	# if "motevation" not in df.columns:
	# 	# Prefer avg_rat_arrival_number; fallback to rat_arrival_number
	# 	if "avg_rat_arrival_number" not in df.columns and "rat_arrival_number" in df.columns:
	# 		df["avg_rat_arrival_number"] = pd.to_numeric(df["rat_arrival_number"], errors="coerce")
	# 	fa = pd.to_numeric(df.get("food_availability"), errors="coerce")
	# 	rn = pd.to_numeric(df.get("avg_rat_arrival_number"), errors="coerce")
	# 	df["motevation"] = fa / rn
	# 	# Clean infinite values from division by zero
	# 	df.loc[~np.isfinite(df["motevation"]), "motevation"] = np.nan

	# if "decision time" not in df.columns:
	# 	sec_after = pd.to_numeric(df.get("seconds_after_rat_arrival"), errors="coerce")
	# 	to_food = pd.to_numeric(df.get("bat_landing_to_food"), errors="coerce")
	# 	df["decision time"] = sec_after + to_food

	# # No risk or habit filtering per request
	# filtered = df.copy()

	# if filtered.empty:
	# 	print("No data available.")
	# 	return

	# Compare motevation (x) vs decision time (y) with stats and scatter plot
	# compare_and_plot(
	# 	filtered,
	# 	x_col="motevation",
	# 	y_col="bat_landing_to_food",
	# 	title="Motivation vs Bat Landing to Food",
	# 	x_label="Motivation (food_availability / avg_rat_arrival_number)",
	# 	y_label="Bat landing to food (seconds)",
	# 	y_unit=" seconds",
	# )

	# Bar chart: average bat_landing_to_food by rounded motivation level,
	# include all records, and remove outliers ONLY from bat_landing_to_food within each group.
	def bar_bltf_by_motivation(df_in: pd.DataFrame) -> None:
		work = df_in.copy()
		# Ensure derived columns
		if "motevation" not in work.columns:
			fa2 = pd.to_numeric(work.get("food_availability"), errors="coerce")
			rn2 = pd.to_numeric(work.get("avg_rat_arrival_number"), errors="coerce")
			work["motevation"] = fa2 / rn2
			work.loc[~np.isfinite(work["motevation"]), "motevation"] = np.nan
		# Numeric coercion
		work["motevation"] = pd.to_numeric(work["motevation"], errors="coerce")
		work["bat_landing_to_food"] = pd.to_numeric(work.get("bat_landing_to_food"), errors="coerce")
		work = work.dropna(subset=["motevation", "bat_landing_to_food"]).copy()
		if work.empty:
			print("No data available for bar chart after filtering.")
			return
		# Group by rounded motivation level (keeping motivation values intact; no outlier removal on motivation)
		work["motivation_level"] = work["motevation"].round().astype("Int64")
		if work["motivation_level"].isna().all():
			print("No valid motivation levels to group by.")
			return
		# Apply IQR filtering ONLY on bat_landing_to_food within each motivation group
		def filter_decision_iqr(g: pd.DataFrame) -> pd.DataFrame:
			y = g["bat_landing_to_food"].dropna()
			if len(y) < 4:
				return g  # not enough data to compute IQR robustly; keep as is
			q1 = y.quantile(0.25)
			q3 = y.quantile(0.75)
			iqr = q3 - q1
			lower = q1 - 1.5 * iqr
			upper = q3 + 1.5 * iqr
			return g[(g["bat_landing_to_food"] >= lower) & (g["bat_landing_to_food"] <= upper)]

		cleaned = work.groupby("motivation_level", dropna=True, group_keys=False).apply(filter_decision_iqr)
		if cleaned.empty:
			print("No data left after bat_landing_to_food IQR filtering by motivation level.")
			return
		# Compute averages
		grouped = (
			cleaned.groupby("motivation_level")
			.agg(count=("bat_landing_to_food", "count"), avg_bltf=("bat_landing_to_food", "mean"))
			.reset_index()
			.sort_values("motivation_level")
		)
		print("Average bat_landing_to_food by motivation level (IQR on bat_landing_to_food only):")
		for _, r in grouped.iterrows():
			lvl = int(r["motivation_level"]) if pd.notna(r["motivation_level"]) else "NaN"
			print(f"  Motivation {lvl}: n={int(r['count'])}, avg bat_landing_to_food={r['avg_bltf']:.2f} seconds")
		print()
		# Bar chart
		fig, ax = plt.subplots(figsize=(9, 6))
		x_vals = grouped["motivation_level"].astype(int).astype(str)
		bars = ax.bar(x_vals, grouped["avg_bltf"], color="#59A14F")
		for i, rect in enumerate(bars):
			avg_val = grouped["avg_bltf"].iloc[i]
			ax.text(
				rect.get_x() + rect.get_width() / 2.0,
				rect.get_height(),
				f"{avg_val:.1f}",
				ha="center",
				va="bottom",
				fontsize=9,
				color="#222",
			)
		ax.set_title("Average Bat Landing to Food by Motivation Level")
		ax.set_xlabel("Motivation level (rounded to nearest integer)")
		ax.set_ylabel("Average bat landing to food (seconds)")
		fig.tight_layout()
		plt.show()

	# Run the bar chart on the already filtered subset
	# bar_bltf_by_motivation(filtered)

	# 1) sum_rat_minutes vs bat_landing_to_food
	# compare_and_plot(
	# 	df,
	# 	x_col="sum_rat_minutes",
	# 	y_col="bat_landing_to_food",
	# 	title="Bats vs. Rats: Vigilance vs Rat Activity\n(sum_rat_minutes vs bat_landing_to_food)",
	# 	x_label="Sum of Rat Minutes",
	# 	y_label="Bat Landing to Food (seconds)",
	# 	y_unit=" seconds",
	# )


	# compare_and_bar(
	# 	df,
	# 	x_col="avg_rat_arrival_number",
	# 	y_col="avg_bat_landing_number",
	# 	title="Bats vs. Rats: Avg Bat Landings vs Avg Rat Arrivals",
	# 	x_label="Avg rat arrival number",
	# 	y_label="Average bat landing number",
	# 	agg="mean",
	# )
	
    
	# 2) rat_minutes vs food_availability
	# compare_and_plot(
	# 	df,
	# 	x_col="bat_landing_to_food",
	# 	y_col="avg_rat_arrival_number",
	# 	title="arrival num vs landing to food time (s)",
	# 	x_label="bat landing to food (seconds)",
	# 	y_label="rat arrival number",
	# 	y_unit="",
	# )

	# Boxplot: bat_landing_to_food by avg_rat_arrival_number (1/2/3)
	# boxplot_by_group(
	# 	df,
	# 	value_col="bat_landing_to_food",
	# 	group_col="avg_rat_arrival_number",
	# 	title="Bat Vigilance by Number of Rats",
	# 	y_label="Bat Landing to Food (seconds)",
	# 	group_values=(1, 2, 3),
	# 	round_group=True,
	# )

	# Bar chart example: categorical variable (e.g., treatment group) vs. a numeric outcome
	# Here we use 'avg_rat_arrival_number' as a dummy categorical variable for illustration
	# if "avg_rat_arrival_number" in df.columns and "bat_landing_to_food" in df.columns:
	# 	df["avg_rat_arrival_number"] = df["avg_rat_arrival_number"].astype(str)  # Convert to string for grouping
	# 	bar_chart_by_category(
	# 		df,
	# 		category_col="avg_rat_arrival_number",
	# 		value_col="bat_landing_to_food",
	# 		title="Bat Landing to Food by Rat Arrival Number Group",
	# 		x_label="Rat Arrival Number Group",
	# 		y_label="Average Bat Landing to Food (seconds)",
	# 	)

	# Bar chart: average bat_landing_to_food by risk
	# bar_chart_by_category(
	# 	df,
	# 	category_col="risk",
	# 	value_col="bat_landing_to_food",
	# 	title="Average Bat Landing to Food by Risk Behavior",
	# 	x_label="Risk (0 = Risk-Avoidance, 1 = Risk-Taking)",
	# 	y_label="Average Bat Landing to Food (seconds)",
	# )

	# Generate the three bar charts for motivation vs landing time by risk
	# motivation_landing_time_by_risk(df)

	# Analyze reward outcomes for high-risk, low-motivation scenarios
	analyze_reward_at_low_motivation(df)


if __name__ == "__main__":
	main()

