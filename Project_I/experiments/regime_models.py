import sys
sys.path.insert(0, r"c:\Projects Python\Project-I\Project-I\Project_I\src")

import os
os.chdir(r"c:\Projects Python\Project-I\Project-I\Project_I")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt

from project_i.cluster_eval import ClusterEvaluator

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({"figure.figsize": (14, 5), "figure.dpi": 100})

# ============================================================
# 1. Data Loading & Weekly Profiles
# ============================================================

df = pd.read_csv("data/clean_energy_data.csv", index_col="timestamp")
df.index = pd.to_datetime(df.index, utc=True)

# Clean main meter
p999 = df["main_meter_power_kw"].quantile(0.999)
df["main_meter_clean_kw"] = df["main_meter_power_kw"].copy()
df.loc[df["main_meter_clean_kw"] > p999, "main_meter_clean_kw"] = np.nan

# Hourly resampling
hourly = df[["main_meter_clean_kw", "temp_c"]].resample("h").mean()
print(f"Hourly: {len(hourly)} rows")
print(f"  main_meter missing: {hourly['main_meter_clean_kw'].isna().mean():.1%}")
print(f"  temp_c missing:     {hourly['temp_c'].isna().mean():.1%}")

# Weekly profiles (168-dim) — one row per week, columns = dow*24 + hour
hourly_power = hourly["main_meter_clean_kw"]
hourly_df = hourly_power.to_frame()
hourly_df["week"] = hourly_power.index.to_period("W").start_time
hourly_df["dow_hour"] = hourly_power.index.dayofweek * 24 + hourly_power.index.hour

weekly_pivot = hourly_df.pivot_table(
    index="week", columns="dow_hour", values="main_meter_clean_kw", aggfunc="mean"
)

# Keep weeks with >= 120 valid hours
valid_wh = weekly_pivot.notna().sum(axis=1)
weekly_pivot = weekly_pivot[valid_wh >= 120]
weekly_pivot = weekly_pivot.interpolate(axis=1, limit=6).dropna()
dates_weekly = pd.to_datetime(weekly_pivot.index)
print(f"\nWeekly profiles: {len(weekly_pivot)} weeks × {weekly_pivot.shape[1]} dims")


# ============================================================
# 2. Temperature Preparation
# ============================================================

# Weekly mean temperature — interpolate gaps up to 4 weeks
weekly_temp = hourly["temp_c"].resample("W-MON").mean()
weekly_temp.index = weekly_temp.index - pd.Timedelta(days=6)  # align to week start
weekly_temp = weekly_temp.interpolate(method="time", limit=4)

# Align with weekly profiles
temp_aligned = weekly_temp.reindex(dates_weekly)
temp_coverage = temp_aligned.notna().mean()
print(f"Temperature coverage after interpolation: {temp_coverage:.1%}")
print(f"  Available: {temp_aligned.notna().sum()} / {len(temp_aligned)} weeks")


# ============================================================
# 3. PCA Reduction
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(weekly_pivot.values)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: {X_scaled.shape[1]} dims → {X_pca.shape[1]} components "
      f"({pca.explained_variance_ratio_.sum():.1%} variance)")

# Variance per component
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
       pca.explained_variance_ratio_, color="steelblue")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA — Variance Explained per Component")
plt.tight_layout()
plt.show()


# ============================================================
# 4. Temperature-Corrected Version
# ============================================================
# Regress each PC on temperature, keep residuals
# Only for weeks where temperature is available

temp_mask = temp_aligned.notna()
print(f"\nTemperature correction on {temp_mask.sum()} weeks (of {len(dates_weekly)})")

X_corrected = X_pca.copy()
temp_vals = temp_aligned.values

if temp_mask.sum() > 50:
    from sklearn.linear_model import LinearRegression

    # For weeks WITH temperature: regress out temp effect
    temp_present = temp_vals[temp_mask].reshape(-1, 1)
    for j in range(X_pca.shape[1]):
        lr = LinearRegression()
        lr.fit(temp_present, X_pca[temp_mask, j])
        # Subtract predicted (temperature-driven) component for ALL weeks with temp
        X_corrected[temp_mask, j] = X_pca[temp_mask, j] - lr.predict(temp_present)

    # For weeks WITHOUT temperature: keep original PCA values (can't correct)
    print(f"  Corrected {temp_mask.sum()} weeks, kept {(~temp_mask).sum()} uncorrected")
else:
    print("  Not enough temperature data for correction, skipping")


# ============================================================
# 5. Model Comparison — pick your data
# ============================================================
# Set which data to use for all models below:
# X_pca = raw PCA-reduced weekly profiles
# X_corrected = temperature-corrected version

X = X_pca  # <-- change to X_corrected to use temp-corrected data
data_label = "Weekly Profiles (PCA)"
# data_label = "Weekly Profiles (temp-corrected)"


# ============================================================
# Model 1: GMM — Gaussian Mixture Model
# ============================================================
print("\n" + "=" * 60)
print("  MODEL 1: Gaussian Mixture Model")
print("=" * 60)

# BIC/AIC for model selection
K_range = range(2, 16)
bic_scores = []
aic_scores = []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=5, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_range), bic_scores, "bo-", linewidth=2, markersize=6)
ax1.set_xlabel("K")
ax1.set_ylabel("BIC")
ax1.set_title("GMM — BIC (lower is better)")
ax1.set_xticks(list(K_range))

ax2.plot(list(K_range), aic_scores, "ro-", linewidth=2, markersize=6)
ax2.set_xlabel("K")
ax2.set_ylabel("AIC")
ax2.set_title("GMM — AIC (lower is better)")
ax2.set_xticks(list(K_range))
plt.tight_layout()
plt.show()

best_k_bic = list(K_range)[np.argmin(bic_scores)]
best_k_aic = list(K_range)[np.argmin(aic_scores)]
print(f"Best K by BIC: {best_k_bic}, by AIC: {best_k_aic}")

# Fit with chosen K (adjust after inspecting BIC/AIC)
K_GMM = 6
gmm_final = GaussianMixture(n_components=K_GMM, covariance_type="full",
                            n_init=10, random_state=42)
gmm_final.fit(X)
labels_gmm = gmm_final.predict(X)
probs_gmm = gmm_final.predict_proba(X)

# Show soft assignment confidence
max_prob = probs_gmm.max(axis=1)
print(f"GMM assignment confidence: mean={max_prob.mean():.2f}, "
      f"min={max_prob.min():.2f}, <0.7: {(max_prob < 0.7).sum()} weeks")

eval_gmm = ClusterEvaluator(X, labels_gmm, dates_weekly,
                            name=f"GMM K={K_GMM} — {data_label}")
eval_gmm.plot_all()


# ============================================================
# Model 2: Hierarchical Clustering
# ============================================================
print("\n" + "=" * 60)
print("  MODEL 2: Hierarchical Clustering")
print("=" * 60)

# Dendrogram (on a sample if too many points)
n_samples = len(X)
if n_samples > 500:
    # Sample for dendrogram readability
    sample_idx = np.random.RandomState(42).choice(n_samples, 500, replace=False)
    X_sample = X[sample_idx]
else:
    X_sample = X
    sample_idx = np.arange(n_samples)

Z = linkage(X_sample, method="ward")

fig, ax = plt.subplots(figsize=(18, 6))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7 * max(Z[:, 2]))
ax.set_title("Hierarchical Clustering — Ward Linkage (dendrogram)")
ax.set_xlabel("Sample index (truncated)")
ax.set_ylabel("Distance")
plt.tight_layout()
plt.show()

# Fit on full data with chosen K (adjust after inspecting dendrogram)
K_HIER = 6
hc = AgglomerativeClustering(n_clusters=K_HIER, linkage="ward")
labels_hier = hc.fit_predict(X)

eval_hier = ClusterEvaluator(X, labels_hier, dates_weekly,
                             name=f"Hierarchical (Ward) K={K_HIER} — {data_label}")
eval_hier.plot_all()


# ============================================================
# Model 3: HMM — Hidden Markov Model
# ============================================================
print("\n" + "=" * 60)
print("  MODEL 3: Hidden Markov Model")
print("=" * 60)

# HMM needs temporally ordered data (already is — weekly_pivot is sorted)
# Model selection via BIC-like score
hmm_scores = []
K_hmm_range = range(2, 12)

for k in K_hmm_range:
    hmm = GaussianHMM(n_components=k, covariance_type="diag",
                      n_iter=200, random_state=42, verbose=False)
    hmm.fit(X)
    log_likelihood = hmm.score(X)
    n_params = k * X.shape[1] * 2 + k * k + k  # means + covs + transitions + priors
    bic = -2 * log_likelihood + n_params * np.log(len(X))
    hmm_scores.append({"k": k, "log_likelihood": log_likelihood, "bic": bic})

hmm_df = pd.DataFrame(hmm_scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hmm_df["k"], hmm_df["log_likelihood"], "go-", linewidth=2, markersize=6)
ax1.set_xlabel("K (hidden states)")
ax1.set_ylabel("Log-Likelihood")
ax1.set_title("HMM — Log-Likelihood (higher is better)")
ax1.set_xticks(list(K_hmm_range))

ax2.plot(hmm_df["k"], hmm_df["bic"], "mo-", linewidth=2, markersize=6)
ax2.set_xlabel("K (hidden states)")
ax2.set_ylabel("BIC")
ax2.set_title("HMM — BIC (lower is better)")
ax2.set_xticks(list(K_hmm_range))
plt.tight_layout()
plt.show()

best_k_hmm = hmm_df.loc[hmm_df["bic"].idxmin(), "k"]
print(f"Best K by BIC: {best_k_hmm}")

# Fit with chosen K (adjust after inspecting)
K_HMM = 6
hmm_final = GaussianHMM(n_components=K_HMM, covariance_type="diag",
                        n_iter=300, random_state=42, verbose=False)
hmm_final.fit(X)
labels_hmm = hmm_final.predict(X)

# Transition matrix
print(f"\nHMM Transition Matrix (K={K_HMM}):")
trans_df = pd.DataFrame(hmm_final.transmat_,
                        index=[f"From {i}" for i in range(K_HMM)],
                        columns=[f"To {i}" for i in range(K_HMM)])
print(trans_df.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(hmm_final.transmat_, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=[f"State {i}" for i in range(K_HMM)],
            yticklabels=[f"State {i}" for i in range(K_HMM)], ax=ax)
ax.set_title(f"HMM Transition Matrix — K={K_HMM}")
ax.set_xlabel("To State")
ax.set_ylabel("From State")
plt.tight_layout()
plt.show()

eval_hmm = ClusterEvaluator(X, labels_hmm, dates_weekly,
                            name=f"HMM K={K_HMM} — {data_label}")
eval_hmm.plot_all()


# ============================================================
# Model 4: Change Point Detection
# ============================================================
print("\n" + "=" * 60)
print("  MODEL 4: Change Point Detection (PELT)")
print("=" * 60)

# PELT on the PCA-reduced weekly series
# Use L2 cost (multivariate signal)
# pen parameter controls sensitivity — higher = fewer change points

# Try different penalty values
for pen in [5, 10, 20, 50]:
    algo = rpt.Pelt(model="l2", min_size=3).fit(X)
    change_points = algo.predict(pen=pen)
    n_segments = len(change_points)
    print(f"  penalty={pen:>3}: {n_segments} segments, "
          f"change points at weeks: {change_points[:-1]}")

# Pick a penalty (adjust after inspecting)
PEN = 10
algo = rpt.Pelt(model="l2", min_size=3).fit(X)
change_points = algo.predict(pen=PEN)

# Assign segment labels
labels_cp = np.zeros(len(X), dtype=int)
prev = 0
for seg_id, cp in enumerate(change_points):
    labels_cp[prev:cp] = seg_id
    prev = cp

n_segments = len(change_points)
print(f"\nPELT (pen={PEN}): {n_segments} segments")

# Visualize change points on the weekly mean consumption
weekly_mean = weekly_pivot.mean(axis=1)

fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(dates_weekly, weekly_mean.values, linewidth=0.8, color="steelblue", alpha=0.7)
for cp in change_points[:-1]:
    if cp < len(dates_weekly):
        ax.axvline(dates_weekly[cp], color="red", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Mean Weekly Power [kW]")
ax.set_title(f"Change Point Detection (PELT, pen={PEN}) — {n_segments} segments")
plt.tight_layout()
plt.show()

# If too many segments, cluster them to get regime labels
if n_segments > 15:
    # Compute segment means in PCA space, then cluster segments
    segment_features = []
    prev = 0
    for cp in change_points:
        segment_features.append(X[prev:cp].mean(axis=0))
        prev = cp
    segment_features = np.array(segment_features)

    K_SEG = 6
    km_seg = KMeans(n_clusters=K_SEG, n_init=10, random_state=42)
    seg_labels = km_seg.fit_predict(segment_features)

    # Map back to weekly labels
    labels_cp_clustered = np.zeros(len(X), dtype=int)
    prev = 0
    for seg_id, cp in enumerate(change_points):
        labels_cp_clustered[prev:cp] = seg_labels[seg_id]
        prev = cp

    print(f"Segments clustered into {K_SEG} regime types")
    eval_cp = ClusterEvaluator(X, labels_cp_clustered, dates_weekly,
                               name=f"PELT+KMeans K={K_SEG} — {data_label}")
    eval_cp.plot_all()
else:
    eval_cp = ClusterEvaluator(X, labels_cp, dates_weekly,
                               name=f"PELT pen={PEN} — {data_label}")
    eval_cp.plot_all()


# ============================================================
# 6. Side-by-Side Timeline Comparison
# ============================================================
print("\n" + "=" * 60)
print("  COMPARISON — All Models")
print("=" * 60)

all_results = {
    f"GMM (K={K_GMM})": labels_gmm,
    f"Hierarchical (K={K_HIER})": labels_hier,
    f"HMM (K={K_HMM})": labels_hmm,
    f"PELT (pen={PEN})": labels_cp_clustered if n_segments > 15 else labels_cp,
}

# Print scores for each
for name, labels in all_results.items():
    ev = ClusterEvaluator(X, labels, dates_weekly, name=name)
    ev.print_scores()

# Side-by-side timelines
from matplotlib.patches import Patch

n_models = len(all_results)
years = sorted(set(dates_weekly.year))
n_years = len(years)

fig, axes = plt.subplots(n_years, n_models, figsize=(5 * n_models, 0.6 * n_years + 2),
                         sharex=True, squeeze=False)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

for col, (model_name, labels) in enumerate(all_results.items()):
    n_clusters = len(np.unique(labels))
    colors_list = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))[:n_clusters]
    unique_labels = sorted(np.unique(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    for row, year in enumerate(years):
        ax = axes[row, col]
        year_mask = dates_weekly.year == year
        year_dates = dates_weekly[year_mask]
        year_labels = labels[year_mask]
        doy = year_dates.dayofyear

        for d, label in zip(doy, year_labels):
            color = colors_list[label_to_idx[label]]
            ax.barh(0, 7, left=d, height=1, color=color, edgecolor="none")

        ax.set_yticks([])
        ax.set_xlim(1, 366)
        ax.set_ylim(-0.5, 0.5)

        for ms in month_starts[1:]:
            ax.axvline(ms, color="white", linewidth=0.5, alpha=0.5)

        if row == 0:
            ax.set_title(model_name, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(str(year), fontsize=8, rotation=0, labelpad=25, va="center")

        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names if row == n_years - 1 else [], fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)

plt.suptitle("Model Comparison — Regime Timelines", fontsize=14, y=1.01)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.08)
plt.show()
