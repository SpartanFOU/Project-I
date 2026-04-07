import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def evaluate_regime_forecast(
    df,
    regime_col,
    target_col="main_meter_power_kw",
    test_from="2024-01-01",
    model_cls=None,
    min_train_days=30,
    regime_as_feature=False,
    plot=True,
):
    """
    Evaluate next-day total consumption forecasting: single model vs. per-regime models.

    Parameters
    ----------
    df : pd.DataFrame
        Raw 15-min data. Index must be a DatetimeIndex. Must contain `target_col`,
        `regime_col`, and optionally `temp_c`, `solar_irradiance_wm2`, `motors_power_kw`.
    regime_col : str
        Column with cluster/regime labels (int or str).
    target_col : str
        Column to predict. Default: "main_meter_power_kw".
    test_from : str or Timestamp
        Chronological split point. Everything on or after this date is test set.
    model_cls : sklearn estimator class or None
        Estimator to use (must implement fit/predict). Default: LinearRegression.
    min_train_days : int
        Regimes with fewer training days than this fall back to the single model.
        Only relevant when regime_as_feature=False.
    regime_as_feature : bool
        If True, one-hot encode the regime label and add it as features to a single
        model (instead of training separate per-regime models). Use this for high-
        capacity models like XGBoost that can learn regime interactions internally
        but suffer from data starvation when split. Default: False.
    plot : bool
        Whether to produce diagnostic plots.

    Returns
    -------
    dict with keys:
        "single"      : {"rmse", "mae", "r2"}
        "regime"      : {"rmse", "mae", "r2",
                         "per_regime": {label: {"rmse", "mae", "n_train", "n_test", "fallback"}}}
        "predictions" : DataFrame (date, actual, pred_single, pred_regime, regime)
        "daily_df"    : DataFrame (aggregated daily data used internally)
    """
    if model_cls is None:
        model_cls = LinearRegression

    # ------------------------------------------------------------------ #
    # 1. Daily aggregation                                                 #
    # ------------------------------------------------------------------ #

    p999 = df[target_col].quantile(0.999)
    target_clipped = df[target_col].clip(upper=p999)

    solar_clipped = (
        df["solar_irradiance_wm2"].clip(lower=0)
        if "solar_irradiance_wm2" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Count valid target readings per day for quality filter
    valid_counts = target_clipped.notna().resample("D").sum()

    # temp_c is recorded once per hour — the other 3 slots per hour are NaN.
    # Forward-fill up to 3 steps to propagate each hourly reading to :15/:30/:45
    # before aggregating to daily, giving near-full daily temperature coverage.
    temp_filled = df["temp_c"].ffill(limit=3) if "temp_c" in df.columns else None

    daily = pd.DataFrame({
        "target_sum":   target_clipped.resample("D").sum(min_count=1),
        "temp_mean":    temp_filled.resample("D").mean() if temp_filled is not None
                        else pd.Series(np.nan, index=target_clipped.resample("D").mean().index),
        "solar_total":  solar_clipped.resample("D").sum(min_count=1),
        "motors_mean":  df["motors_power_kw"].resample("D").mean() if "motors_power_kw" in df.columns
                        else pd.Series(np.nan, index=target_clipped.resample("D").mean().index),
        "regime":       df[regime_col].resample("D").apply(
                            lambda x: x.dropna().mode().iloc[0] if x.notna().any() else np.nan
                        ),
    })

    # Drop days with insufficient target readings (< 18 of 96 possible 15-min slots)
    daily = daily[valid_counts >= 18].copy()
    daily = daily.dropna(subset=["target_sum", "regime"])

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                               #
    # ------------------------------------------------------------------ #

    # Predict target_sum for day i+1 using features from day i and before
    ts = daily["target_sum"]

    features = pd.DataFrame(index=daily.index)

    for lag in [1, 7, 14, 28]:
        features[f"power_roll{lag}d"] = ts.shift(1).rolling(lag, min_periods=1).mean()

    for lag in [7, 14]:
        features[f"power_std{lag}d"] = ts.shift(1).rolling(lag, min_periods=1).std()

    # DOW and month of *tomorrow* (the day being predicted)
    tomorrow = daily.index + pd.Timedelta(days=1)
    features["dow_sin"]   = np.sin(2 * np.pi * tomorrow.dayofweek / 7)
    features["dow_cos"]   = np.cos(2 * np.pi * tomorrow.dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * tomorrow.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * tomorrow.month / 12)

    # Temperature features — after ffill on raw 15-min data, daily coverage is near-full
    features["temp_today"]  = daily["temp_mean"].values
    features["temp_roll7d"] = daily["temp_mean"].shift(1).rolling(7, min_periods=1).mean()

    # Solar features
    features["solar_today"]   = daily["solar_total"].values
    features["solar_roll7d"]  = daily["solar_total"].shift(1).rolling(7, min_periods=1).mean()

    # Drop warmup rows
    WARMUP = 30
    features = features.iloc[WARMUP:]
    daily_aligned = daily.iloc[WARMUP:].copy()

    # Target: next-day sum (shift target by -1, then drop last row)
    y_series = daily_aligned["target_sum"].shift(-1)
    valid_target = y_series.notna()
    features = features[valid_target]
    daily_aligned = daily_aligned[valid_target]
    y = y_series[valid_target].values

    # Fill sparse feature columns (temp is ~75% missing) with column median
    feature_medians = features.median()
    features = features.fillna(feature_medians)

    dates = daily_aligned.index
    regimes = daily_aligned["regime"].values
    feature_names = list(features.columns)
    X = features.values

    # Regime-as-feature: one-hot encode regime and append to feature matrix
    if regime_as_feature:
        regime_dummies = pd.get_dummies(daily_aligned["regime"], prefix="regime", dtype=float)
        X_r = np.concatenate([X, regime_dummies.values], axis=1)
    else:
        X_r = None

    # ------------------------------------------------------------------ #
    # 3. Train / test split                                                #
    # ------------------------------------------------------------------ #

    test_cutoff = pd.Timestamp(test_from).tz_localize("UTC") if dates.tz is not None else pd.Timestamp(test_from)
    train_mask = dates < test_cutoff
    test_mask  = ~train_mask

    if train_mask.sum() == 0:
        raise ValueError(f"No training data before test_from={test_from}")
    if test_mask.sum() == 0:
        raise ValueError(f"No test data from test_from={test_from} onward")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    reg_train, reg_test = regimes[train_mask], regimes[test_mask]
    dates_test = dates[test_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if X_r is not None:
        X_r_train, X_r_test = X_r[train_mask], X_r[test_mask]
        scaler_r = StandardScaler()
        X_r_train_s = scaler_r.fit_transform(X_r_train)
        X_r_test_s  = scaler_r.transform(X_r_test)

    # ------------------------------------------------------------------ #
    # 4. Model training                                                    #
    # ------------------------------------------------------------------ #

    # Single model (baseline — no regime info)
    single_model = model_cls()
    single_model.fit(X_train_s, y_train)

    unique_test_regimes = np.unique(reg_test)

    if regime_as_feature:
        # One augmented model — regime encoded as one-hot features
        regime_feature_model = model_cls()
        regime_feature_model.fit(X_r_train_s, y_train)
        regime_models = {}
        fallback_regimes = set()
    else:
        # Per-regime models
        unique_train_regimes = np.unique(reg_train)
        regime_models = {}
        fallback_regimes = set()

        for r in unique_train_regimes:
            mask = reg_train == r
            n = mask.sum()
            if n < min_train_days:
                fallback_regimes.add(r)
                print(f"  [warn] Regime {r}: only {n} train days < min_train_days={min_train_days} — using fallback")
                continue
            m = model_cls()
            m.fit(X_train_s[mask], y_train[mask])
            regime_models[r] = m

        # Warn about regimes in test but not in train
        for r in unique_test_regimes:
            if r not in regime_models and r not in fallback_regimes:
                print(f"  [warn] Regime {r} appears in test set but has no train data — using fallback")
                fallback_regimes.add(r)

    # ------------------------------------------------------------------ #
    # 5. Prediction                                                        #
    # ------------------------------------------------------------------ #

    pred_single = single_model.predict(X_test_s)

    if regime_as_feature:
        pred_regime = regime_feature_model.predict(X_r_test_s)
    else:
        pred_regime = np.empty_like(pred_single)
        for i, r in enumerate(reg_test):
            if r in regime_models:
                pred_regime[i] = regime_models[r].predict(X_test_s[i: i + 1])[0]
            else:
                pred_regime[i] = pred_single[i]

    # ------------------------------------------------------------------ #
    # 6. Evaluation                                                        #
    # ------------------------------------------------------------------ #

    def _metrics(actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae  = mean_absolute_error(actual, predicted)
        r2   = r2_score(actual, predicted)
        return {"rmse": rmse, "mae": mae, "r2": r2}

    single_metrics = _metrics(y_test, pred_single)
    regime_metrics = _metrics(y_test, pred_regime)

    per_regime_stats = {}
    for r in unique_test_regimes:
        mask = reg_test == r
        n_train = (reg_train == r).sum()
        n_test  = mask.sum()
        is_fallback = r in fallback_regimes or r not in regime_models
        if mask.sum() > 1:
            stats = _metrics(y_test[mask], pred_regime[mask])
        else:
            stats = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
        stats["n_train"]  = int(n_train)
        stats["n_test"]   = int(n_test)
        stats["fallback"] = is_fallback
        per_regime_stats[r] = stats

    # ------------------------------------------------------------------ #
    # 7. Print summary                                                     #
    # ------------------------------------------------------------------ #

    delta_rmse = (regime_metrics["rmse"] - single_metrics["rmse"]) / single_metrics["rmse"] * 100

    regime_mode = "feature" if regime_as_feature else "split"
    print(f"\n{'=' * 60}")
    print(f"  Regime Forecast Evaluation  (test from {test_from})")
    print(f"  regime mode: {regime_mode}   |   target: next-day {target_col} sum   |   n_test={test_mask.sum()}")
    print(f"{'=' * 60}")
    print(f"  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>6}")
    print(f"  {'-' * 44}")
    print(f"  {'Single model':<20} {single_metrics['rmse']:>8.1f} "
          f"{single_metrics['mae']:>8.1f} {single_metrics['r2']:>6.3f}")
    print(f"  {'Regime models':<20} {regime_metrics['rmse']:>8.1f} "
          f"{regime_metrics['mae']:>8.1f} {regime_metrics['r2']:>6.3f}"
          f"   (Δ RMSE: {delta_rmse:+.1f}%)")
    print(f"\n  Per-regime breakdown (test set):")
    print(f"  {'Regime':<10} {'RMSE':>8} {'MAE':>8} {'R²':>6} "
          f"{'n_train':>8} {'n_test':>7}  fallback")
    print(f"  {'-' * 62}")
    for r, s in sorted(per_regime_stats.items()):
        rmse_str = f"{s['rmse']:8.1f}" if not np.isnan(s["rmse"]) else "     n/a"
        mae_str  = f"{s['mae']:8.1f}"  if not np.isnan(s["mae"])  else "     n/a"
        r2_str   = f"{s['r2']:6.3f}"   if not np.isnan(s["r2"])   else "   n/a"
        fb_str   = "yes" if s["fallback"] else "no"
        print(f"  {str(r):<10} {rmse_str} {mae_str} {r2_str} "
              f"{s['n_train']:>8} {s['n_test']:>7}  {fb_str}")

    # ------------------------------------------------------------------ #
    # 8. Plots                                                             #
    # ------------------------------------------------------------------ #

    if plot:
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)

        # --- Plot 1: Time series of actual vs predicted ---
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(dates_test, y_test,       color="black",      lw=1.2, label="Actual",        zorder=3)
        ax.plot(dates_test, pred_single,  color="steelblue",  lw=1.0, label="Single model",  alpha=0.8)
        ax.plot(dates_test, pred_regime,  color="tomato",     lw=1.0, label="Regime models", alpha=0.8)
        ax.set_title(f"Next-day forecast — test set  (RMSE single={single_metrics['rmse']:.1f}, "
                     f"regime={regime_metrics['rmse']:.1f})")
        ax.set_ylabel(f"{target_col} sum [kW]")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # --- Plot 2: RMSE per regime (regime model vs single model per regime) ---
        regime_labels = sorted(per_regime_stats.keys())
        rmse_regime_per = []
        rmse_single_per = []
        for r in regime_labels:
            mask = reg_test == r
            rmse_regime_per.append(
                np.sqrt(mean_squared_error(y_test[mask], pred_regime[mask])) if mask.sum() > 1 else np.nan
            )
            rmse_single_per.append(
                np.sqrt(mean_squared_error(y_test[mask], pred_single[mask])) if mask.sum() > 1 else np.nan
            )

        x = np.arange(len(regime_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(regime_labels) * 1.5), 4))
        bars1 = ax.bar(x - width / 2, rmse_single_per, width, label="Single model",  color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + width / 2, rmse_regime_per, width, label="Regime model",  color="tomato",    alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Regime {r}" for r in regime_labels])
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE per regime — single model vs regime-specific model")
        ax.legend()
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}",
                        ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.show()

        # --- Plot 3: Residuals distribution ---
        res_single = y_test - pred_single
        res_regime = y_test - pred_regime
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        ax1.hist(res_single, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax1.axvline(0, color="black", lw=1.2, ls="--")
        ax1.set_title(f"Residuals — single model  (std={res_single.std():.1f})")
        ax1.set_xlabel("Residual [kW]")
        ax2.hist(res_regime, bins=40, color="tomato", alpha=0.8, edgecolor="white")
        ax2.axvline(0, color="black", lw=1.2, ls="--")
        ax2.set_title(f"Residuals — regime models  (std={res_regime.std():.1f})")
        ax2.set_xlabel("Residual [kW]")
        plt.suptitle("Residual distributions — test set", fontsize=12)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # 9. Return                                                            #
    # ------------------------------------------------------------------ #

    predictions = pd.DataFrame({
        "actual":       y_test,
        "pred_single":  pred_single,
        "pred_regime":  pred_regime,
        "regime":       reg_test,
    }, index=dates_test)

    return {
        "single":      single_metrics,
        "regime":      {**regime_metrics, "per_regime": per_regime_stats},
        "predictions": predictions,
        "daily_df":    daily_aligned,
    }


# ------------------------------------------------------------------ #
# Usage example                                                        #
# ------------------------------------------------------------------ #
#
# Step 1: run topdown_regime_clustering.py to get day_period_labels + dates_D
#
# Step 2: attach regime labels back onto the raw 15-min dataframe
#
#   regime_series = pd.Series(day_period_labels, index=dates_D)
#   df["regime"] = regime_series.reindex(df.index.normalize().tz_localize(None)).values
#
# Step 3: call the evaluator
#
#   import sys
#   sys.path.insert(0, r"c:\Projects Python\Project-I\Project-I\Project_I\src")
#   from project_i.regime_forecast import evaluate_regime_forecast
#
#   results = evaluate_regime_forecast(df, regime_col="regime", test_from="2024-01-01")
#
# Step 4: inspect
#
#   results["predictions"].head()
#   results["regime"]["per_regime"]
