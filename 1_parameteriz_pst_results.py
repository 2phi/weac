import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter, get_common_distributions
from IPython.utils import io
import numpy as np
import os
from scipy.stats import skew, kurtosis

from plot_distribution import distribution

distributions = [
    "gamma",
    "norm",
    "lognorm",
    "expon",
    "beta",
    "weibull_min",
    "cauchy",
    "exponpow",
    "chi2",
]

# Create a directory for plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Load the data
try:
    df = pd.read_csv("pst_to_GIc.csv")
except FileNotFoundError:
    print("pst_to_GIc.csv not found. Please run 1_eval_pst.py first.")
    exit()

print("Data loaded successfully. Starting analysis...")
print(df.info())
print(df.head())

# Exclude rows where the density is unphysically low.
df = df[df["rho_wl"] >= 50]

# Stats
mean = df["GIc"].mean()
std = df["GIc"].std()
skew = skew(df["GIc"])
kurt = kurtosis(df["GIc"])
print(f"Mean: {mean:.3f}, Std: {std:.3f}, Skew: {skew:.3f}, Kurt: {kurt:.3f}")

# --- Part 1: Plotting distributions of individual variables ---

# Fit distributions to GIc
print("\nFitting distributions to GIc...")
hist_bins = np.histogram_bin_edges(df["GIc"], bins=30)  # Try 50, 30, etc.
g_ic_fitter = Fitter(
    df["GIc"].dropna(),
    bins=hist_bins,
    distributions=distributions,
)
with io.capture_output() as captured:
    g_ic_fitter.fit()
print("Best distributions for GIc:")
summary = g_ic_fitter.summary()
print(summary)

# Distribution of GIc
distribution(
    df["GIc"],
    dist_type="lognorm",
    kind="pdf",
    bins=75,
    plot_range=(0, 5),
    save="plots/GIc_pdf.png",
)

rho_bins = np.histogram_bin_edges(df["rho_wl"], bins=25)
# Fit distributions to rho_wl
print("\nFitting distributions to rho_wl...")
rho_wl_fitter = Fitter(
    df["rho_wl"].dropna(),
    bins=rho_bins,
    distributions=distributions,
)
with io.capture_output() as captured:
    rho_wl_fitter.fit()
print("Best distributions for rho_wl:")
summary = rho_wl_fitter.summary()
print(summary)

# Distribution of rho_wl
distribution(
    df["rho_wl"],
    dist_type="beta",
    kind="pdf",
    bins=25,
    plot_range=(50, 400),
    save="plots/rho_wl_pdf.png",
)
# Cumulative distribution of rho_wl
distribution(
    df["rho_wl"],
    dist_type="beta",
    kind="cdf",
    bins=25,
    plot_range=(50, 400),
    save="plots/rho_wl_cdf.png",
)

# Distribution of HH_wl (Hand Hardness) (8 string entries)
plt.figure(figsize=(12, 7))
sns.countplot(y=df["HH_wl"], order=df["HH_wl"].value_counts().index)
plt.title("Distribution of Weak Layer Hand Hardness (HH_wl)")
plt.xlabel("Count")
plt.ylabel("Hand Hardness")
plt.tight_layout()
plt.savefig("plots/HH_wl_distribution.png")
plt.close()

# Distribution of GT_wl (Grain Type)
plt.figure(figsize=(12, 8))
sns.countplot(y=df["GT_wl"], order=df["GT_wl"].value_counts().index)
plt.title("Distribution of Weak Layer Grain Type (GT_wl)")
plt.xlabel("Count")
plt.ylabel("Grain Type")
plt.tight_layout()
plt.savefig("plots/GT_wl_distribution.png")
plt.close()


# Distribution of GS_wl (Grain Size)
plt.figure(figsize=(10, 6))
sns.histplot(df["GS_wl"], kde=True, bins=10, binrange=(0, 10))
plt.title("Distribution of Weak Layer Grain Size (GS_wl)")
plt.xlabel("Grain Size (mm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/GS_wl_distribution.png")
plt.close()


# # --- Part 2: Analyzing relationships with GIc ---

# # From rho_wl to GIc
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x="rho_wl", y="GIc", alpha=0.5)
# plt.title("GIc vs. Weak Layer Density (rho_wl)")
# plt.xlabel("Density (kg/m^3)")
# plt.ylabel("GIc (J/m^2)")
# plt.tight_layout()
# plt.savefig("plots/GIc_vs_rho_wl_scatter.png")
# plt.close()

# # Bin rho_wl and plot GIc distributions
# df["rho_wl_binned"] = pd.qcut(
#     df["rho_wl"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
# )
# plt.figure(figsize=(12, 7))
# sns.boxplot(data=df, x="rho_wl_binned", y="GIc")
# plt.title("GIc Distribution by Weak Layer Density Bins")
# plt.xlabel("Density Bins (Quartiles)")
# plt.ylabel("GIc (J/m^2)")
# plt.tight_layout()
# plt.savefig("plots/GIc_by_rho_wl_bins.png")
# plt.close()


# # From HH_wl (binned) to GIc
# hh_order = df.groupby("HH_wl")["GIc"].median().sort_values().index
# plt.figure(figsize=(12, 7))
# sns.boxplot(data=df, x="HH_wl", y="GIc", order=hh_order)
# plt.title("GIc Distribution by Weak Layer Hand Hardness (HH_wl)")
# plt.xlabel("Hand Hardness")
# plt.ylabel("GIc (J/m^2)")
# plt.tight_layout()
# plt.savefig("plots/GIc_by_HH_wl.png")
# plt.close()

# # Fit distributions for GIc for each HH category
# print("\nFitting distributions to GIc for each Hand Hardness category...")
# hh_categories = df["HH_wl"].dropna().unique()
# for cat in hh_categories:
#     subset = df[df["HH_wl"] == cat]["GIc"].dropna()
#     if len(subset) > 50:  # Only fit if there are enough data points
#         print(f"--- Fitting GIc for HH_wl = {cat} ---")
#         f = Fitter(subset, distributions=get_common_distributions())
#         with io.capture_output() as captured:
#             f.fit()
#         summary = f.summary()
#         print(summary)

# # From GT_wl (binned) to GIc
# gt_order = df.groupby("GT_wl")["GIc"].median().sort_values().index
# plt.figure(figsize=(12, 8))
# sns.boxplot(data=df, x="GT_wl", y="GIc", order=gt_order)
# plt.title("GIc Distribution by Weak Layer Grain Type (GT_wl)")
# plt.xlabel("Grain Type")
# plt.ylabel("GIc (J/m^2)")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("plots/GIc_by_GT_wl.png")
# plt.close()

# # Fit distributions for GIc for each GT category
# print("\nFitting distributions to GIc for each Grain Type category...")
# gt_categories = df["GT_wl"].dropna().unique()
# for cat in gt_categories:
#     subset = df[df["GT_wl"] == cat]["GIc"].dropna()
#     if len(subset) > 50:
#         print(f"--- Fitting GIc for GT_wl = {cat} ---")
#         f = Fitter(subset, distributions=get_common_distributions())
#         with io.capture_output() as captured:
#             f.fit()
#         summary = f.summary()
#         print(summary)

# print("\nAnalysis complete. Plots are saved in the 'plots/' directory.")
