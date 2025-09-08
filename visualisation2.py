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


def main():
	# Inputs
	default_csv = os.path.join(os.path.dirname(__file__), "dataset1_merged.csv")
	csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

	if not os.path.exists(csv_path):
		print(f"ERROR: Data file not found: {csv_path}")
		sys.exit(1)

	# Load once
	df = pd.read_csv(csv_path)

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
	compare_and_plot(
		df,
		x_col="bat_landing_to_food",
		y_col="avg_rat_arrival_number",
		title="arrival num vs landing to food time (s)",
		x_label="bat landing to food (seconds)",
		y_label="rat arrival number",
		y_unit="",
	)


if __name__ == "__main__":
	main()

