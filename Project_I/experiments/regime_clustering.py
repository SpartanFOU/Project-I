import sys
sys.path.insert(0, r"c:\Projects Python\Project-I\Project-I\Project_I\src")

import os
os.chdir(r"c:\Projects Python\Project-I\Project-I\Project_I")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from project_i.cluster_eval import ClusterEvaluator

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({"figure.figsize": (14, 5), "figure.dpi": 100})

# ============================================================
# 1. Load & prepare data
# ============================================================

df = pd.read_csv("data/clean_energy_data.csv", index_col="timestamp")
df.index = pd.to_datetime(df.index, utc=True)

# Recreate main_meter_clean_kw (cap at 99.9th percentile)
p999 = df["main_meter_power_kw"].quantile(0.999)
df["main_meter_clean_kw"] = df["main_meter_power_kw"].copy()
df.loc[df["main_meter_clean_kw"] > p999, "main_meter_clean_kw"] = np.nan

# Resample to hourly means
hourly = df["main_meter_clean_kw"].resample("h").mean()
print(f"Hourly series: {len(hourly)} rows, {hourly.isna().mean():.1%} missing")

# ============================================================
# Experiment A — Daily profiles (24-dim)
# ============================================================
# Pivot: one row per day, 24 columns (hours 0-23)

daily_profiles = hourly.to_frame()
daily_profiles["date"] = hourly.index.date
daily_profiles["hour"] = hourly.index.hour

daily_pivot = daily_profiles.pivot_table(
    index="date", columns="hour", values="main_meter_clean_kw", aggfunc="mean"
)

# Drop days with <20 valid hours
valid_hours = daily_pivot.notna().sum(axis=1)
daily_pivot = daily_pivot[valid_hours >= 20]
print(f"Daily profiles: {len(daily_pivot)} days (dropped {(valid_hours < 20).sum()} incomplete days)")

# Interpolate remaining small gaps within each day
daily_pivot = daily_pivot.interpolate(axis=1, limit=4)
daily_pivot = daily_pivot.dropna()
print(f"After interpolation & dropna: {len(daily_pivot)} days")

# Scale
scaler_a = StandardScaler()
X_daily = scaler_a.fit_transform(daily_pivot.values)
dates_daily = pd.to_datetime(daily_pivot.index)

# K selection
print("\n--- Experiment A: Daily profiles (24-dim) ---")
sil_a = ClusterEvaluator.plot_k_selection(X_daily, k_range=range(2, 16), name="Daily Profiles")

# Pick K (inspect the plots, then set K here)
K_A = 6
km_a = KMeans(n_clusters=K_A, n_init=10, random_state=42)
labels_a = km_a.fit_predict(X_daily)

eval_a = ClusterEvaluator(X_daily, labels_a, dates_daily, name=f"Daily Profiles (K={K_A})")
eval_a.plot_all()


# ============================================================
# Experiment B — Weekly profiles (168-dim)
# ============================================================
# Pivot: one row per week, columns = (dayofweek, hour) → 7 * 24 = 168

hourly_df = hourly.to_frame()
hourly_df["week"] = hourly.index.to_period("W").start_time
hourly_df["dow_hour"] = hourly.index.dayofweek * 24 + hourly.index.hour

weekly_pivot = hourly_df.pivot_table(
    index="week", columns="dow_hour", values="main_meter_clean_kw", aggfunc="mean"
)

# Drop weeks with <120 valid hours (~71%)
valid_wh = weekly_pivot.notna().sum(axis=1)
weekly_pivot = weekly_pivot[valid_wh >= 120]
print(f"\nWeekly profiles: {len(weekly_pivot)} weeks (dropped {(valid_wh < 120).sum()} incomplete)")

# Interpolate small gaps, drop remaining NaN rows
weekly_pivot = weekly_pivot.interpolate(axis=1, limit=6)
weekly_pivot = weekly_pivot.dropna()
print(f"After interpolation & dropna: {len(weekly_pivot)} weeks")

# Scale
scaler_b = StandardScaler()
X_weekly = scaler_b.fit_transform(weekly_pivot.values)
dates_weekly = pd.to_datetime(weekly_pivot.index)

# K selection
print("\n--- Experiment B: Weekly profiles (168-dim) ---")
sil_b = ClusterEvaluator.plot_k_selection(X_weekly, k_range=range(2, 16), name="Weekly Profiles")

# Pick K (inspect the plots, then set K here)
for K_B in [4, 6, 8]:
   # K_B = 4
    km_b = KMeans(n_clusters=K_B, n_init=10, random_state=42)
    labels_b = km_b.fit_predict(X_weekly)

    eval_b = ClusterEvaluator(X_weekly, labels_b, dates_weekly, name=f"Weekly Profiles (K={K_B})")
    eval_b.plot_all()


# ============================================================
# Experiment C — Weekly feature vectors
# ============================================================

hourly_ts = hourly.dropna()

def extract_weekly_features(series):
    """Extract feature vector per week from hourly consumption series."""
    weekly_groups = series.groupby(series.index.to_period("W").start_time)

    records = []
    for week_start, group in weekly_groups:
        if len(group) < 120:
            continue

        hours = group.index.hour
        dow = group.index.dayofweek

        records.append({
            "week": week_start,
            "mean": group.mean(),
            "std": group.std(),
            "median": group.median(),
            "min": group.min(),
            "max": group.max(),
            "base_load": group.quantile(0.05),
            "peak_load": group.quantile(0.95),
            "weekday_mean": group[dow < 5].mean() if (dow < 5).any() else np.nan,
            "weekend_mean": group[dow >= 5].mean() if (dow >= 5).any() else np.nan,
            "wd_we_ratio": (group[dow < 5].mean() / group[dow >= 5].mean()
                            if (dow >= 5).any() and group[dow >= 5].mean() > 0
                            else np.nan),
            "night_mean": group[(hours >= 22) | (hours < 6)].mean(),
            "day_mean": group[(hours >= 8) & (hours < 18)].mean(),
        })

    return pd.DataFrame(records).set_index("week")


features_df = extract_weekly_features(hourly_ts)
features_df = features_df.dropna()
print(f"\nWeekly features: {len(features_df)} weeks, {features_df.shape[1]} features")
print(f"Features: {list(features_df.columns)}")

# Scale
scaler_c = StandardScaler()
X_features = scaler_c.fit_transform(features_df.values)
dates_features = pd.to_datetime(features_df.index)

# K selection
print("\n--- Experiment C: Weekly feature vectors ---")
sil_c = ClusterEvaluator.plot_k_selection(X_features, k_range=range(2, 16), name="Weekly Features")

# Pick K (inspect the plots, then set K here)
K_C = 6
km_c = KMeans(n_clusters=K_C, n_init=10, random_state=42)
labels_c = km_c.fit_predict(X_features)

eval_c = ClusterEvaluator(X_features, labels_c, dates_features, name=f"Weekly Features (K={K_C})")
eval_c.plot_all(feature_labels=list(features_df.columns))


# ============================================================
# Experiment D — Combined B + C (weekly profiles + features)
# ============================================================

# Align on common weeks
common_weeks = weekly_pivot.index.intersection(features_df.index)
print(f"\n--- Experiment D: Combined B+C ---")
print(f"Common weeks: {len(common_weeks)} (B={len(weekly_pivot)}, C={len(features_df)})")

X_b_aligned = scaler_b.transform(weekly_pivot.loc[common_weeks].values)
X_c_aligned = scaler_c.transform(features_df.loc[common_weeks].values)
X_combined = np.hstack([X_b_aligned, X_c_aligned])
dates_combined = pd.to_datetime(common_weeks)

print(f"Combined feature matrix: {X_combined.shape}")

# K selection
sil_d = ClusterEvaluator.plot_k_selection(X_combined, k_range=range(2, 16), name="Combined B+C")

# Pick K
K_D = 8
km_d = KMeans(n_clusters=K_D, n_init=10, random_state=42)
labels_d = km_d.fit_predict(X_combined)

eval_d = ClusterEvaluator(X_combined, labels_d, dates_combined, name=f"Combined B+C (K={K_D})")
eval_d.plot_all()
