import sys
sys.path.insert(0, r"c:\Projects Python\Project-I\Project-I\Project_I\src")

import os
os.chdir(r"c:\Projects Python\Project-I\Project-I\Project_I")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

from project_i.cluster_eval import ClusterEvaluator

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({"figure.figsize": (14, 5), "figure.dpi": 100})


# ============================================================
# 1. Load & clean data
# ============================================================

df = pd.read_csv("data/clean_energy_data.csv", index_col="timestamp")
df.index = pd.to_datetime(df.index, utc=True)

p999 = df["main_meter_power_kw"].quantile(0.999)
df["main_meter_clean_kw"] = df["main_meter_power_kw"].clip(upper=p999)
df["solar_irradiance_clean"] = df["solar_irradiance_wm2"].clip(lower=0)

hourly_power  = df["main_meter_clean_kw"].resample("h").mean()
hourly_solar  = df["solar_irradiance_clean"].resample("h").mean()
hourly_temp   = df["temp_c"].resample("h").mean()
hourly_motors = df["motors_power_kw"].resample("h").mean()

print(f"Hourly power: {len(hourly_power)} rows, {hourly_power.isna().mean():.1%} missing")


# ============================================================
# 2. Daily profiles (24-dim, per-day normalised)
#    Mode D pipeline — same as notebook 03 best result
# ============================================================

hourly_df = hourly_power.to_frame()
hourly_df["day"]  = hourly_power.index.normalize()
hourly_df["hour"] = hourly_power.index.hour

daily_pivot = hourly_df.pivot_table(
    index="day", columns="hour", values="main_meter_clean_kw", aggfunc="mean"
)
valid_hours = daily_pivot.notna().sum(axis=1)
daily_pivot = daily_pivot[valid_hours >= 18]
daily_pivot = daily_pivot.interpolate(axis=1, limit=3).dropna()

raw_values  = daily_pivot.values.copy()
dates_daily = pd.DatetimeIndex(daily_pivot.index)

row_means = raw_values.mean(axis=1, keepdims=True)
row_stds  = raw_values.std(axis=1, keepdims=True)
row_stds[row_stds == 0] = 1
X_perday = (raw_values - row_means) / row_stds

print(f"Daily profiles: {len(dates_daily)} days")


# ============================================================
# 3. Mode D context features + PCA
#    (DOW sin/cos, month sin/cos, temp, solar, motors, rolling lags)
# ============================================================

daily_meta = pd.DataFrame(index=dates_daily)

dow   = dates_daily.dayofweek
month = dates_daily.month
daily_meta["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
daily_meta["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
daily_meta["month_sin"] = np.sin(2 * np.pi * month / 12)
daily_meta["month_cos"] = np.cos(2 * np.pi * month / 12)

daily_meta["temp_mean"]   = hourly_temp.resample("D").mean().reindex(dates_daily).values
daily_meta["solar_total"] = hourly_solar.resample("D").sum().reindex(dates_daily).values
daily_meta["motors_mean"] = hourly_motors.resample("D").mean().reindex(dates_daily).values

power_daily = pd.Series(raw_values.mean(axis=1), index=dates_daily)
temp_daily  = pd.Series(daily_meta["temp_mean"].values, index=dates_daily)

for lag in [1, 7, 30]:
    daily_meta[f"power_lag{lag}d"] = (
        power_daily.shift(1).rolling(lag, min_periods=1).mean().values
    )
    daily_meta[f"temp_lag{lag}d"] = (
        temp_daily.shift(1).rolling(lag, min_periods=1).mean().values
    )

WARMUP   = 30
X_shape  = X_perday[WARMUP:]
raw_vals = raw_values[WARMUP:]
dates_D  = dates_daily[WARMUP:]
meta_D   = daily_meta.iloc[WARMUP:].copy().fillna(daily_meta.iloc[WARMUP:].median())

scaler_meta = StandardScaler()
X_D   = np.hstack([X_shape, scaler_meta.fit_transform(meta_D.values)])
pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_D)

print(f"Mode D PCA: {X_D.shape[1]} → {X_pca.shape[1]} dims "
      f"({pca.explained_variance_ratio_.sum():.1%} variance)")


# ============================================================
# 4. Period-level features  (top-down step 1)
#
#    Rolling W-day window computed DIRECTLY from raw data —
#    no dependency on L1 day-type labels.
#
#    Features per day:
#      1. detrended rolling mean consumption   (level w/o multi-year drift)
#      2. rolling std of raw consumption       (variability: semester > break)
#      3. weekday / weekend consumption ratio  (structure: semester >> break)
#      4-5. month sin / cos of endpoint        (season anchor)
#
#    Lessons applied:
#      - detrend to remove the building's increasing baseline (seen in W=28 raw result)
#      - W=28 gives stable, contiguous period blocks (better than W=14)
# ============================================================

W_PERIOD = 14

raw_daily_power = raw_vals.mean(axis=1)
power_series    = pd.Series(raw_daily_power, index=dates_D)
power_detrended = (
    power_series - power_series.rolling(365, min_periods=180, center=True).mean()
).values

dow_arr = dates_D.dayofweek.values


def build_period_features(power_det, power_raw, dow, W):
    """
    3 features — all derived from consumption patterns, no calendar encoding.

    month_sin/cos were removed: they step-change at month boundaries and cause
    the HMM to segment by calendar month rather than by consumption regime.
    The seasonal signal (summer dip, winter peak) is already present in
    detrended_mean and raw_std after the 365-day detrend.

    Features:
      1. detrended rolling mean  — level relative to annual baseline
      2. rolling std (raw)       — variability: semester > break
      3. weekday / weekend ratio — structure: active semester >> holiday period
    """
    rows, idx = [], []
    for d in range(W - 1, len(power_det)):
        det_win = power_det[d - W + 1: d + 1]
        raw_win = power_raw[d - W + 1: d + 1]
        dow_win = dow[d - W + 1: d + 1]

        wd = raw_win[dow_win < 5]
        we = raw_win[dow_win >= 5]
        wday = np.nanmean(wd) if len(wd) > 0 else np.nan
        wend = np.nanmean(we) if len(we) > 0 else np.nan
        ratio = wday / wend if (wend and wend > 0) else np.nan

        rows.append([
            np.nanmean(det_win),
            np.nanstd(raw_win),
            ratio,
        ])
        idx.append(dates_D[d])

    X = pd.DataFrame(rows).fillna(pd.DataFrame(rows).median()).values
    return X, pd.DatetimeIndex(idx)


X_period_raw, dates_period = build_period_features(
    power_detrended, raw_daily_power, dow_arr, W_PERIOD
)
X_period = StandardScaler().fit_transform(X_period_raw)
print(f"\nPeriod feature matrix: {X_period.shape}  (W={W_PERIOD}d)")
print("Features: detrended_mean | raw_std | wd/we_ratio")


# ============================================================
# 5. Period K selection — HMM BIC scan
#
#    HMM is better than K-Means for period detection because it
#    explicitly models temporal transitions between regimes.
#    The transition matrix learns that e.g. "summer break" rarely
#    jumps directly to "winter exam" — enforcing contiguous blocks.
#
#    Model selection: BIC (lower = better).
#    BIC = -2 * log_likelihood + n_params * log(n_samples)
# ============================================================

K_range    = range(2, 10)
hmm_scores = []

print("\n--- Period HMM K selection (BIC) ---")
for k in K_range:
    hmm = GaussianHMM(n_components=k, covariance_type="diag",
                      n_iter=200, random_state=42, verbose=False)
    hmm.fit(X_period)
    ll       = hmm.score(X_period)
    n_params = k * X_period.shape[1] * 2 + k * k + k   # means + diag-covs + transmat + startprob
    bic      = -2 * ll + n_params * np.log(len(X_period))
    hmm_scores.append({"k": k, "log_likelihood": ll, "bic": bic})
    print(f"  K={k}: log-likelihood={ll:.1f}  BIC={bic:.1f}")

hmm_df = pd.DataFrame(hmm_scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hmm_df["k"], hmm_df["log_likelihood"], "bo-", lw=2, ms=6)
ax1.set_xlabel("K (states)")
ax1.set_ylabel("Log-likelihood")
ax1.set_title(f"HMM — Log-likelihood (W={W_PERIOD}d)")
ax1.set_xticks(list(K_range))

ax2.plot(hmm_df["k"], hmm_df["bic"], "ro-", lw=2, ms=6)
ax2.set_xlabel("K (states)")
ax2.set_ylabel("BIC  (lower = better)")
ax2.set_title(f"HMM — BIC (W={W_PERIOD}d)")
ax2.set_xticks(list(K_range))

plt.suptitle("Period HMM — model selection", fontsize=13)
plt.tight_layout()
plt.show()


# ============================================================
# 6. Period clustering — HMM fit
#    Adjust K_PERIOD after inspecting BIC plot above
# ============================================================

K_PERIOD = 6   # <-- adjust after inspecting BIC plot

hmm_period    = GaussianHMM(n_components=K_PERIOD, covariance_type="diag",
                             n_iter=300, random_state=42, verbose=False)
hmm_period.fit(X_period)
labels_period = hmm_period.predict(X_period)

print(f"\nHMM log-likelihood (K={K_PERIOD}): {hmm_period.score(X_period):.2f}")

# Transition matrix
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(hmm_period.transmat_, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=[f"P{i}" for i in range(K_PERIOD)],
            yticklabels=[f"P{i}" for i in range(K_PERIOD)], ax=ax)
ax.set_title(f"HMM Transition Matrix — Period K={K_PERIOD}  (W={W_PERIOD}d)")
plt.tight_layout()
plt.show()

eval_period = ClusterEvaluator(X_period, labels_period, dates_period,
                                name=f"HMM Period K={K_PERIOD} W={W_PERIOD}d")
eval_period.plot_all()


# ============================================================
# 7. Assign HMM period label to every day in dates_D
#    Day d gets the label of the window ending on d.
#    The first W-1 days (before the first window) get the first window's label.
# ============================================================

day_period_labels = np.empty(len(dates_D), dtype=int)
for d_idx in range(len(dates_D)):
    win_idx = max(0, d_idx - (W_PERIOD - 1))
    day_period_labels[d_idx] = labels_period[win_idx]

print("\nPeriod label distribution:")
for p, c in zip(*np.unique(day_period_labels, return_counts=True)):
    print(f"  Period {p}: {c:>4} days ({c / len(day_period_labels):.1%})")


# ============================================================
# 8. Day sub-clustering within each period  (top-down step 2)
#
#    Within each period cluster, re-cluster individual days using
#    Mode D PCA features (shape + DOW + season context).
#    DOW in features lets the algorithm find workday/weekend/holiday
#    sub-types with seasonal confounding already removed.
#
#    Lessons applied:
#      - per-day norm in PCA features → shape-based, not level-based
#      - DOW sin/cos in features → holidays stand out (weekday DOW, weekend shape)
# ============================================================

K_SUB    = 2    # <-- workday / weekend / holiday-like
MIN_DAYS = 50   # minimum days to attempt sub-clustering

sub_labels_per_day = np.full(len(dates_D), -1, dtype=int)  # sub-cluster within period
period_sub_results = {}  # period -> sub_labels array (aligned to that period's days)

for p in sorted(np.unique(day_period_labels)):
    mask      = day_period_labels == p
    n_days    = mask.sum()
    X_sub     = X_pca[mask]
    raw_sub   = raw_vals[mask]
    dates_sub = dates_D[mask]
    dow_sub   = dates_sub.dayofweek

    print(f"\n{'=' * 55}")
    print(f"  Period {p} — {n_days} days")
    print(f"{'=' * 55}")

    if n_days < MIN_DAYS:
        print(f"  Too few days for sub-clustering — kept as single group")
        period_sub_results[p] = np.zeros(n_days, dtype=int)
        sub_labels_per_day[mask] = 0
        continue

    km_sub   = KMeans(n_clusters=K_SUB, n_init=10, random_state=42)
    sub_labs = km_sub.fit_predict(X_sub)
    period_sub_results[p] = sub_labs

    # Write sub-labels back to the full array
    sub_labels_per_day[mask] = sub_labs

    eval_sub = ClusterEvaluator(X_sub, sub_labs, dates_sub,
                                name=f"Period {p} sub-clusters (K={K_SUB})")
    eval_sub.print_scores()

    # --- DOW heatmap ---
    dow_names   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    contingency = pd.DataFrame(
        np.zeros((K_SUB, 7), dtype=int),
        index=[f"Sub-{s}" for s in range(K_SUB)],
        columns=dow_names,
    )
    for lab, d in zip(sub_labs, dow_sub):
        contingency.iloc[lab, d] += 1
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100

    # --- Mean raw profiles per sub-cluster ---
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    sns.heatmap(contingency_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax1)
    ax1.set_title(f"Period {p} — day-of-week % per sub-cluster")
    ax1.set_ylabel("")

    for s in range(K_SUB):
        smask  = sub_labs == s
        mean_p = raw_sub[smask].mean(axis=0)
        std_p  = raw_sub[smask].std(axis=0)
        ax2.plot(np.arange(24), mean_p, color=base_colors[s], linewidth=2,
                 label=f"Sub-{s}  n={smask.sum()}")
        ax2.fill_between(np.arange(24), mean_p - std_p, mean_p + std_p,
                         alpha=0.15, color=base_colors[s])

    ax2.set_xlabel("Hour")
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_ylabel("Power [kW]  (raw)")
    ax2.set_title(f"Period {p} — mean daily profile per sub-cluster")
    ax2.legend(fontsize=9)
    plt.suptitle(f"Period {p} sub-clustering  (K_sub={K_SUB})", fontsize=13)
    plt.tight_layout()
    plt.show()

    # Sub-cluster timeline (within this period's days)
    eval_sub.plot_timeline()


# ============================================================
# 9. Composite timeline
#    Each day = (period, sub-cluster) pair, rendered as a unique colour.
#    tab20 gives 20 distinct colours — enough for 6 periods × 3 sub-types.
# ============================================================

period_list = sorted(np.unique(day_period_labels))
n_composite = K_PERIOD * K_SUB
tab20       = plt.cm.tab20(np.linspace(0, 1, 20))

def composite_color(period, sub):
    return tab20[(period * K_SUB + sub) % 20]

years        = sorted(set(dates_D.year))
month_names  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

fig, axes = plt.subplots(len(years), 1,
                          figsize=(20, 0.7 * len(years) + 2),
                          sharex=True, squeeze=False)
axes = axes.flatten()

for i, year in enumerate(years):
    ax        = axes[i]
    ymask     = dates_D.year == year
    ydates    = dates_D[ymask]
    yperiod   = day_period_labels[ymask]
    ysub      = sub_labels_per_day[ymask]

    for doy, p, s in zip(ydates.dayofyear, yperiod, ysub):
        ax.barh(0, 1, left=doy, height=1,
                color=composite_color(p, s), edgecolor="none")

    ax.set_yticks([])
    ax.set_ylabel(str(year), fontsize=10, rotation=0, labelpad=30, va="center")
    ax.set_xlim(1, 366)
    ax.set_ylim(-0.5, 0.5)
    for ms in month_starts[1:]:
        ax.axvline(ms, color="white", linewidth=0.8, alpha=0.7)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names if i == len(years) - 1 else [], fontsize=9)
    ax.tick_params(axis="x", length=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

patches = [
    mpatches.Patch(facecolor=composite_color(p, s), label=f"P{p}-S{s}")
    for p in period_list
    for s in range(K_SUB if day_period_labels[day_period_labels == p].sum() >= MIN_DAYS else 1)
]
fig.legend(handles=patches, loc="upper right", fontsize=8,
           ncol=min(len(patches), 9), frameon=True, edgecolor="gray")
plt.suptitle(
    f"Composite Regime Timeline — Period K={K_PERIOD}, Day-type K_sub={K_SUB}, W={W_PERIOD}d",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.show()


# ============================================================
# 10. Summary table — composite label statistics + DOW profile
# ============================================================

print(f"\n{'=' * 65}")
print(f"  Composite regime summary  "
      f"(K_period={K_PERIOD}, K_sub={K_SUB}, W={W_PERIOD}d)")
print(f"{'=' * 65}")
print(f"  {'Label':<10} {'Days':>5} {'%':>6}   "
      f"Mon  Tue  Wed  Thu  Fri  Sat  Sun")
print(f"  {'-' * 60}")

for p in period_list:
    pmask   = day_period_labels == p
    sub_arr = period_sub_results[p]
    n_sub   = K_SUB if pmask.sum() >= MIN_DAYS else 1

    for s in range(n_sub):
        if n_sub == 1:
            full_mask = pmask
        else:
            full_mask          = np.zeros(len(dates_D), dtype=bool)
            full_mask[pmask]   = sub_arr == s

        n_days    = full_mask.sum()
        pct       = n_days / len(dates_D) * 100
        dow_counts = np.bincount(dates_D[full_mask].dayofweek, minlength=7)
        dow_str    = "  ".join(f"{d:>3.0f}" for d in dow_counts)
        print(f"  P{p}-Sub{s}    {n_days:>5} {pct:>5.1f}%   {dow_str}")
