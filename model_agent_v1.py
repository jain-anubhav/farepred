"""
Flight price prediction model agent_v1 - price-level percentile rank features.

New signals added on top of model_v4:
  price_zscore_market_dfd   - z-score of price_at_obs within (pathod_id) at dfd=40
                              using training group mean/std (no leakage).
  price_pctrank_market_dfd  - percentile rank of price_at_obs within the training
                              distribution for the same pathod_id at dfd=40.

Rationale: price-tier analysis shows RMSE 101.3 for high-price flights vs 29.1 for
low-price. The model needs to know where a price sits in the empirical distribution
for that market, giving it a "cheap / fair / expensive" signal relative to history.

Implementation note:
  Since all observations are at dfd=40, we group training data at dfd==40 by
  pathod_id to get group (mean, std, sorted prices). These stats are applied to
  both train and test rows to avoid leakage.
"""
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

OBSERVATION_DFD = 40
FLIGHT_KEY = ['flt_id', 'pathod_id', 'airline', 'depdt']


def load_data():
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')
    return train, test


def add_offer_date(df):
    """Add offer_date = depdt - dfd (integer days since epoch for merging)."""
    df = df.copy()
    df['depdt_dt'] = pd.to_datetime(df['depdt'])
    df['offer_date'] = (df['depdt_dt'] - pd.to_timedelta(df['dfd'], unit='D')).dt.date
    return df


def build_offer_date_features(df):
    """
    Compute market-level stats grouped by (pathod_id, offer_date) WITHIN the same dataset.

    Unlike looking up training stats for test data (which fails when dates don't overlap),
    this operates on whatever dataset is passed — so test features are built from test data,
    and train features from train data. This mirrors how market_stats is currently built.

    For each flight at observation_dfd=40, the offer_date = depdt - 40 is the calendar
    date of observation. On that date, OTHER flights in the same market (different depdts)
    are also being observed — their prices reflect contemporaneous market demand levels.
    """
    df = add_offer_date(df)

    # At observation_dfd: all flights in this market observed on this calendar date
    obs_rows = df[df['dfd'] == OBSERVATION_DFD].copy()

    offer_obs = obs_rows.groupby(['pathod_id', 'offer_date']).agg(
        offer_obs_mean=('target_price', 'mean'),
        offer_obs_median=('target_price', 'median'),
        offer_obs_std=('target_price', 'std'),
        offer_obs_min=('target_price', 'min'),
        offer_obs_max=('target_price', 'max'),
        offer_obs_n=('target_price', 'count'),
        offer_obs_emf_mean=('expected_minfare', 'mean'),
    ).reset_index()

    # Near-departure flights (dfd <= 7) observed on this calendar date in this market
    # → captures what near-departure prices look like "right now" for the market
    near_dep = df[df['dfd'] <= 7].groupby(['pathod_id', 'offer_date']).agg(
        offer_neardep_mean=('target_price', 'mean'),
        offer_neardep_median=('target_price', 'median'),
        offer_neardep_n=('target_price', 'count'),
    ).reset_index()

    offer_feats = offer_obs.merge(near_dep, on=['pathod_id', 'offer_date'], how='left')
    return offer_feats


def build_global_trajectory_features(train_df, observation_dfd=OBSERVATION_DFD):
    """Average price trajectory per (pathod_id, airline, depdow) at each DFD."""
    group_cols = ['pathod_id', 'airline', 'depdow']

    trajectory = train_df.groupby(group_cols + ['dfd'])['target_price'].agg(
        ['mean', 'median', 'std']
    ).reset_index()
    trajectory.columns = group_cols + ['dfd', 'traj_mean', 'traj_median', 'traj_std']

    train_df2 = train_df.copy()
    train_df2['price_ratio'] = train_df2['target_price'] / (train_df2['expected_minfare'] + 1e-6)
    trajectory_ratio = train_df2.groupby(group_cols + ['dfd'])['price_ratio'].agg(
        ['mean', 'std']
    ).reset_index()
    trajectory_ratio.columns = group_cols + ['dfd', 'traj_ratio_mean', 'traj_ratio_std']

    traj_combined = trajectory.merge(trajectory_ratio, on=group_cols + ['dfd'], how='left')
    return traj_combined


def build_market_dfd_stats(train_df, observation_dfd=OBSERVATION_DFD):
    """
    Build per-pathod_id price distribution stats from training data at dfd=observation_dfd.

    Returns:
        group_stats_df : DataFrame with columns [pathod_id, mkt_mean, mkt_std]
        group_prices   : dict mapping pathod_id -> sorted numpy array of prices
                         (used for percentile rank lookup)
    """
    at_obs = train_df[train_df['dfd'] == observation_dfd].copy()

    group_stats_df = at_obs.groupby('pathod_id')['target_price'].agg(
        mkt_mean='mean',
        mkt_std='std',
    ).reset_index()

    # Store sorted price arrays per group for percentile rank computation
    group_prices = {}
    for pathod_id, grp in at_obs.groupby('pathod_id'):
        group_prices[pathod_id] = np.sort(grp['target_price'].dropna().values)

    return group_stats_df, group_prices


def compute_price_zscore(price, mkt_mean, mkt_std):
    """Compute z-score with epsilon to avoid division by zero."""
    return (price - mkt_mean) / (mkt_std + 1e-6)


def compute_pctrank(price, sorted_prices):
    """
    Compute percentile rank of price within sorted_prices array.
    Returns value in [0, 100].
    Falls back to 50.0 if array is empty.
    """
    if sorted_prices is None or len(sorted_prices) == 0:
        return 50.0
    return percentileofscore(sorted_prices, price, kind='rank')


def build_flight_features(df, train_df, global_traj, observation_dfd=OBSERVATION_DFD):
    """Build per-flight features using only info available at observation_dfd."""
    group_cols = ['pathod_id', 'airline', 'depdow']

    # Add offer_date to df
    df = add_offer_date(df)

    # Static flight info
    static = df[FLIGHT_KEY + ['dephour', 'depdow', 'depdt_dt']].drop_duplicates(subset=FLIGHT_KEY)

    # Known prices (dfd >= observation_dfd)
    known = df[df['dfd'] >= observation_dfd].copy()
    known['price_ratio'] = known['target_price'] / (known['expected_minfare'] + 1e-6)

    # Price at exactly observation_dfd
    at_obs = known[known['dfd'] == observation_dfd][
        FLIGHT_KEY + ['target_price', 'expected_minfare', 'offer_date']
    ].rename(columns={'target_price': 'price_at_obs', 'expected_minfare': 'emf_at_obs'})

    # History stats (dfd >= observation_dfd)
    hist_stats = known.groupby(FLIGHT_KEY).agg(
        hist_price_mean=('target_price', 'mean'),
        hist_price_std=('target_price', 'std'),
        hist_price_max=('target_price', 'max'),
        hist_price_min=('target_price', 'min'),
        hist_price_range=('target_price', lambda x: x.max() - x.min()),
        hist_ratio_mean=('price_ratio', 'mean'),
        hist_ratio_std=('price_ratio', 'std'),
        hist_ratio_max=('price_ratio', 'max'),
        hist_ratio_min=('price_ratio', 'min'),
    ).reset_index()

    # Prices at specific lookback dfds
    lookback_data = at_obs[FLIGHT_KEY].copy()
    for dfd_val in range(observation_dfd, min(observation_dfd + 10, 50)):
        sub = df[df['dfd'] == dfd_val][FLIGHT_KEY + ['target_price']].rename(
            columns={'target_price': f'price_dfd{dfd_val}'}
        )
        lookback_data = lookback_data.merge(sub, on=FLIGHT_KEY, how='left')

    def compute_trend(row):
        prices = [row.get(f'price_dfd{d}', np.nan) for d in range(observation_dfd, min(observation_dfd + 10, 50))]
        prices = [p for p in prices if not np.isnan(p)]
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices))
        return np.polyfit(x, prices, 1)[0]

    lookback_data['price_trend'] = lookback_data.apply(compute_trend, axis=1)
    lookback_data['price_recent_vs_old'] = (
        lookback_data.get(f'price_dfd{observation_dfd}', np.nan) -
        lookback_data.get(f'price_dfd{min(observation_dfd + 9, 49)}', np.nan)
    )

    # Market competition at observation_dfd
    market_at_obs = df[df['dfd'] == observation_dfd].copy()
    market_stats = market_at_obs.groupby(['pathod_id', 'depdt']).agg(
        market_min_price=('target_price', 'min'),
        market_max_price=('target_price', 'max'),
        market_mean_price=('target_price', 'mean'),
        market_n_airlines=('airline', 'nunique'),
    ).reset_index()

    # Training trajectory at observation_dfd
    traj_at_obs = global_traj[global_traj['dfd'] == observation_dfd][
        group_cols + ['traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean']
    ].rename(columns={
        'traj_mean': 'hist_traj_mean_at_obs',
        'traj_median': 'hist_traj_median_at_obs',
        'traj_std': 'hist_traj_std_at_obs',
        'traj_ratio_mean': 'hist_traj_ratio_at_obs',
    })

    # === NEW: offer_date features (within-dataset, no train→test leakage issue) ===
    offer_date_feats = build_offer_date_features(df)

    # Combine
    flight_feats = static.merge(at_obs, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(hist_stats, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(lookback_data, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(market_stats, on=['pathod_id', 'depdt'], how='left')
    flight_feats = flight_feats.merge(traj_at_obs, on=group_cols, how='left')
    flight_feats = flight_feats.merge(offer_date_feats, on=['pathod_id', 'offer_date'], how='left')

    # Temporal features of the offer_date — seasonal demand signal independent of depdt
    flight_feats['offer_month'] = pd.to_datetime(flight_feats['offer_date'].astype(str)).dt.month
    flight_feats['offer_dow'] = pd.to_datetime(flight_feats['offer_date'].astype(str)).dt.dayofweek
    flight_feats['offer_dayofyear'] = pd.to_datetime(flight_feats['offer_date'].astype(str)).dt.dayofyear

    # Price position relative to the offer_date market level
    flight_feats['price_vs_offer_obs_mean'] = (
        flight_feats['price_at_obs'] - flight_feats['offer_obs_mean']
    )
    flight_feats['price_vs_offer_obs_median'] = (
        flight_feats['price_at_obs'] - flight_feats['offer_obs_median']
    )
    # Relative ratio: this flight vs. market mean on offer_date
    flight_feats['price_ratio_to_offer_obs'] = (
        flight_feats['price_at_obs'] / (flight_feats['offer_obs_mean'] + 1e-6)
    )

    # =========================================================================
    # NEW: Price-Level Percentile Rank Within Market-DFD Distribution
    # =========================================================================
    # Build group stats from training data at dfd=observation_dfd (no leakage:
    # test rows use training distribution, not their own).
    group_stats_df, group_prices = build_market_dfd_stats(train_df, observation_dfd)

    # Merge training group (mean, std) onto flight_feats
    flight_feats = flight_feats.merge(
        group_stats_df[['pathod_id', 'mkt_mean', 'mkt_std']],
        on='pathod_id',
        how='left',
    )

    # Z-score: (price_at_obs - group_mean) / (group_std + epsilon)
    flight_feats['price_zscore_market_dfd'] = flight_feats.apply(
        lambda row: compute_price_zscore(row['price_at_obs'], row['mkt_mean'], row['mkt_std'])
        if pd.notna(row['price_at_obs']) and pd.notna(row['mkt_mean'])
        else np.nan,
        axis=1,
    )

    # Percentile rank: where does price_at_obs sit in the training CDF for this pathod_id?
    flight_feats['price_pctrank_market_dfd'] = flight_feats.apply(
        lambda row: compute_pctrank(
            row['price_at_obs'],
            group_prices.get(row['pathod_id'], np.array([])),
        )
        if pd.notna(row['price_at_obs'])
        else np.nan,
        axis=1,
    )

    # Drop temporary merge columns (not features themselves)
    flight_feats = flight_feats.drop(columns=['mkt_mean', 'mkt_std'], errors='ignore')

    return flight_feats


def prepare_data(df, train_df, global_traj, observation_dfd=OBSERVATION_DFD):
    """Create one row per (flight, target_dfd) with full feature set."""
    group_cols = ['pathod_id', 'airline', 'depdow']

    flight_feats = build_flight_features(df, train_df, global_traj, observation_dfd)

    # Future rows to predict
    df_dated = add_offer_date(df)
    future = df_dated[df_dated['dfd'] < observation_dfd][
        FLIGHT_KEY + ['dfd', 'expected_minfare', 'target_price']
    ].copy()

    # Merge flight features
    data = future.merge(flight_feats, on=FLIGHT_KEY, how='left')

    # Merge global trajectory features for the target dfd
    traj_target = global_traj[
        group_cols + ['dfd', 'traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean', 'traj_ratio_std']
    ].rename(columns={
        'traj_mean': 'hist_traj_mean',
        'traj_median': 'hist_traj_median',
        'traj_std': 'hist_traj_std',
        'traj_ratio_mean': 'hist_traj_ratio_mean',
        'traj_ratio_std': 'hist_traj_ratio_std',
    })
    data = data.merge(traj_target, on=group_cols + ['dfd'], how='left')

    # Derived features
    data['dfd_diff'] = observation_dfd - data['dfd']
    data['dfd_frac'] = data['dfd'] / observation_dfd
    data['emf_ratio_to_obs'] = data['expected_minfare'] / (data['emf_at_obs'] + 1e-6)
    data['price_to_emf'] = data['price_at_obs'] / (data['expected_minfare'] + 1e-6)
    data['price_to_market_min'] = data['price_at_obs'] / (data['market_min_price'] + 1e-6)
    data['price_above_emf'] = data['price_at_obs'] - data['emf_at_obs']
    data['price_vs_hist_traj'] = data['price_at_obs'] - data['hist_traj_mean_at_obs']
    data['hist_traj_emf_price'] = data['hist_traj_ratio_mean'] * data['expected_minfare']
    data['airline_enc'] = data['airline'].astype('category').cat.codes

    # Baseline
    data['baseline_pred'] = np.maximum(data['price_at_obs'], data['expected_minfare'])
    data['residual'] = data['target_price'] - data['baseline_pred']

    return data


def get_features():
    return [
        # Core flight features
        'dfd', 'dfd_diff', 'dfd_frac',
        'dephour', 'depdow', 'airline_enc', 'pathod_id',
        'price_at_obs', 'emf_at_obs', 'expected_minfare',
        'emf_ratio_to_obs', 'price_to_emf', 'price_to_market_min', 'price_above_emf',
        'hist_price_mean', 'hist_price_std', 'hist_price_max', 'hist_price_min', 'hist_price_range',
        'hist_ratio_mean', 'hist_ratio_std', 'hist_ratio_max', 'hist_ratio_min',
        'price_trend', 'price_recent_vs_old',
        'market_min_price', 'market_max_price', 'market_mean_price', 'market_n_airlines',
        'price_dfd40', 'price_dfd41', 'price_dfd42', 'price_dfd43', 'price_dfd44',
        'price_dfd45', 'price_dfd46', 'price_dfd47', 'price_dfd48', 'price_dfd49',
        'hist_traj_mean_at_obs', 'hist_traj_median_at_obs', 'hist_traj_std_at_obs',
        'hist_traj_ratio_at_obs',
        'hist_traj_mean', 'hist_traj_median', 'hist_traj_std',
        'hist_traj_ratio_mean', 'hist_traj_ratio_std',
        'hist_traj_emf_price',
        'price_vs_hist_traj',
        'baseline_pred',
        # offer_date features
        'offer_obs_mean', 'offer_obs_median', 'offer_obs_std',
        'offer_obs_min', 'offer_obs_max', 'offer_obs_n', 'offer_obs_emf_mean',
        'offer_neardep_mean', 'offer_neardep_median', 'offer_neardep_n',
        'offer_month', 'offer_dow', 'offer_dayofyear',
        'price_vs_offer_obs_mean', 'price_vs_offer_obs_median',
        'price_ratio_to_offer_obs',
        # === NEW: Price-Level Percentile Rank Within Market-DFD Distribution ===
        'price_zscore_market_dfd',
        'price_pctrank_market_dfd',
    ]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_by_dfd(y_true, y_pred, dfd_vals, label='', baseline=None):
    overall = rmse(y_true, y_pred)
    pct = (rmse(y_true, baseline) - overall) / rmse(y_true, baseline) * 100 if baseline is not None else 0
    print(f"  {label}: RMSE={overall:.4f} ({pct:+.2f}% vs baseline)")
    for dfd_val in [0, 5, 10, 20, 30, 39]:
        mask = dfd_vals == dfd_val
        if mask.sum() > 0:
            r = rmse(y_true[mask], y_pred[mask])
            bl = f" (baseline={rmse(y_true[mask], baseline[mask]):.4f})" if baseline is not None else ""
            print(f"    dfd={dfd_val}: RMSE={r:.4f}{bl}, n={mask.sum()}")
    return overall


def main():
    print("Loading data...")
    train, test = load_data()

    print("Computing global trajectory features from training data...")
    global_traj = build_global_trajectory_features(train)

    print("Preparing features (offer_date + price-percentile features)...")
    train_data = prepare_data(train, train, global_traj)
    test_data = prepare_data(test, train, global_traj)

    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Check new feature coverage
    feature_cols = get_features()
    new_cols = ['price_zscore_market_dfd', 'price_pctrank_market_dfd']
    print("  New feature missing rates:")
    for col in new_cols:
        miss_train = train_data[col].isna().mean() * 100
        miss_test = test_data[col].isna().mean() * 100
        print(f"    {col}: train_missing={miss_train:.1f}%, test_missing={miss_test:.1f}%")

    # Check offer_date feature coverage
    offer_cols = [c for c in feature_cols if c.startswith('offer_')]
    print("  Offer_date feature missing rates:")
    for col in offer_cols:
        miss_train = train_data[col].isna().mean() * 100
        miss_test = test_data[col].isna().mean() * 100
        print(f"    {col}: train_missing={miss_train:.1f}%, test_missing={miss_test:.1f}%")

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['target_price']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['target_price']
    dfd_test = test_data['dfd'].values
    baseline_test = test_data['baseline_pred'].values

    baseline_rmse = rmse(y_test, baseline_test)
    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")

    # ===========================
    # Model A: LightGBM (absolute price)
    # ===========================
    print("\n=== Model A: LightGBM (with offer_date + price-percentile features) ===")
    lgb_a = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_a.fit(X_train, y_train)
    pred_a = lgb_a.predict(X_test)
    rmse_a = evaluate_by_dfd(y_test, pred_a, dfd_test, "LightGBM-offer", baseline_test)

    # ===========================
    # Model B: LightGBM residual
    # ===========================
    print("\n=== Model B: LightGBM Residual (with offer_date + price-percentile features) ===")
    y_train_resid = train_data['residual']
    lgb_b = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_b.fit(X_train, y_train_resid)
    resid_pred = lgb_b.predict(X_test)
    pred_b = baseline_test + resid_pred
    rmse_b = evaluate_by_dfd(y_test, pred_b, dfd_test, "LightGBM-resid-offer", baseline_test)

    # ===========================
    # Model C: XGBoost
    # ===========================
    print("\n=== Model C: XGBoost (with offer_date + price-percentile features) ===")
    xgb_c = xgb.XGBRegressor(
        n_estimators=1500, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_c.fit(X_train, y_train)
    pred_c = xgb_c.predict(X_test)
    rmse_c = evaluate_by_dfd(y_test, pred_c, dfd_test, "XGBoost-offer", baseline_test)

    # ===========================
    # Ensemble
    # ===========================
    print("\n=== Ensemble: Weighted (A+B+C) ===")
    weights = np.array([1/rmse_a, 1/rmse_b, 1/rmse_c])
    weights = weights / weights.sum()
    print(f"  Weights: A={weights[0]:.3f}, B={weights[1]:.3f}, C={weights[2]:.3f}")
    pred_ens = weights[0] * pred_a + weights[1] * pred_b + weights[2] * pred_c
    rmse_ens = evaluate_by_dfd(y_test, pred_ens, dfd_test, "Ensemble-weighted", baseline_test)

    # ===========================
    # Feature Importances
    # ===========================
    print("\n=== Feature Importances (LightGBM Model A, top 25) ===")
    fi = pd.DataFrame({'feature': feature_cols, 'importance': lgb_a.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print(fi.head(25).to_string(index=False))

    print("\n--- New price-percentile features specifically ---")
    new_fi = fi[fi['feature'].isin(new_cols)]
    print(new_fi.to_string(index=False))

    print("\n--- Offer_date features specifically ---")
    offer_fi = fi[fi['feature'].str.startswith('offer_')]
    print(offer_fi.to_string(index=False))

    # ===========================
    # FINAL SUMMARY
    # ===========================
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Model':<40} {'RMSE':>8} {'vs Baseline':>12}")
    print("-" * 63)
    print(f"{'Baseline':<40} {baseline_rmse:>8.4f} {'(reference)':>12}")
    for label, r in [
        ("LightGBM-offer", rmse_a),
        ("LightGBM-resid-offer", rmse_b),
        ("XGBoost-offer", rmse_c),
        ("Ensemble-weighted", rmse_ens),
    ]:
        pct = (baseline_rmse - r) / baseline_rmse * 100
        print(f"  {label:<38} {r:>8.4f} {pct:>+11.2f}%")


if __name__ == '__main__':
    main()
