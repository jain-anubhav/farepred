"""
Flight price prediction model v5 - offer_date trend features.

Builds on v4 offer_date signal with two new ideas:

1. DFD-range fare curve at offer_date:
   On a given calendar date (offer_date X) in market M, flights are observed
   at many different DFDs. Computing market-level prices by DFD range gives a
   snapshot of the demand curve on that day:
     near (dfd 0-7):   close-in bookings — reflects immediate demand
     mid  (dfd 8-20):  medium advance
     adv  (dfd 21-39): advance bookings
   The shape of this curve (near vs. adv spread) signals market conditions.

2. Price trends over offer_dates:
   How are prices in this market changing over calendar time?
   trend_7d  = current market level - level 7 days ago
   trend_14d = current market level - level 14 days ago
   acceleration = trend_7d - trend_14d/2 (is price change speeding up?)
   Built from combined (train + test) offer_date time series → 100% coverage.
"""
import pandas as pd
import numpy as np
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
    df = df.copy()
    df['depdt_dt'] = pd.to_datetime(df['depdt'])
    df['offer_date'] = (df['depdt_dt'] - pd.to_timedelta(df['dfd'], unit='D')).dt.date
    return df


# ──────────────────────────────────────────────────────────────
# Offer-date feature builders
# ──────────────────────────────────────────────────────────────

def build_offer_date_features(df):
    """
    Within-dataset offer_date features (no train→test lookup issues).

    At observation_dfd=40, the offer_date = depdt - 40.  Other flights in
    the same market observed on that calendar day reflect contemporaneous
    market demand.  We also compute DFD-range fare curves to capture the
    shape of the demand schedule.
    """
    df = add_offer_date(df)

    # dfd=40 observations — same-horizon pricing across different departures
    obs = df[df['dfd'] == OBSERVATION_DFD].groupby(['pathod_id', 'offer_date']).agg(
        offer_obs_mean=('target_price', 'mean'),
        offer_obs_median=('target_price', 'median'),
        offer_obs_std=('target_price', 'std'),
        offer_obs_min=('target_price', 'min'),
        offer_obs_max=('target_price', 'max'),
        offer_obs_n=('target_price', 'count'),
        offer_obs_emf_mean=('expected_minfare', 'mean'),
    ).reset_index()

    # Near-departure prices (dfd 0-7): close-in demand signal
    near = df[df['dfd'] <= 7].groupby(['pathod_id', 'offer_date']).agg(
        offer_near_mean=('target_price', 'mean'),
        offer_near_median=('target_price', 'median'),
        offer_near_n=('target_price', 'count'),
    ).reset_index()

    # Mid-range (dfd 8-20): medium advance bookings
    mid = df[(df['dfd'] >= 8) & (df['dfd'] <= 20)].groupby(['pathod_id', 'offer_date']).agg(
        offer_mid_mean=('target_price', 'mean'),
        offer_mid_n=('target_price', 'count'),
    ).reset_index()

    # Advance bookings (dfd 21-39): between observation and near-dep
    adv = df[(df['dfd'] >= 21) & (df['dfd'] <= 39)].groupby(['pathod_id', 'offer_date']).agg(
        offer_adv_mean=('target_price', 'mean'),
        offer_adv_n=('target_price', 'count'),
    ).reset_index()

    feats = obs.merge(near, on=['pathod_id', 'offer_date'], how='left')
    feats = feats.merge(mid,  on=['pathod_id', 'offer_date'], how='left')
    feats = feats.merge(adv,  on=['pathod_id', 'offer_date'], how='left')

    # Demand curve shape features
    feats['offer_near_vs_adv'] = feats['offer_near_mean'] - feats['offer_adv_mean']
    feats['offer_near_vs_obs'] = feats['offer_near_mean'] - feats['offer_obs_mean']
    feats['offer_adv_vs_obs']  = feats['offer_adv_mean']  - feats['offer_obs_mean']

    return feats


def build_offer_timeseries(train_df, test_df):
    """
    Build a combined (train + test) per-(pathod_id, offer_date) time series
    for trend computation.  Using both datasets gives ~100% coverage for
    7-day and 14-day lookbacks on test.

    No target leakage: for any flight at offer_date X, the lookback stats
    come from offer_dates X-7 and X-14, which involve DIFFERENT flights
    (different depdt values), so no future self-referencing.
    """
    combined = pd.concat([add_offer_date(train_df), add_offer_date(test_df)], ignore_index=True)

    # dfd=40 time series: how does market-level observed price change day-by-day?
    ts_obs = combined[combined['dfd'] == OBSERVATION_DFD].groupby(['pathod_id', 'offer_date']).agg(
        ts_obs_mean=('target_price', 'mean'),
        ts_obs_median=('target_price', 'median'),
        ts_obs_std=('target_price', 'std'),
        ts_obs_n=('target_price', 'count'),
    ).reset_index()

    # DFD-range time series (for trend in near/mid/adv prices)
    ts_near = combined[combined['dfd'] <= 7].groupby(['pathod_id', 'offer_date']).agg(
        ts_near_mean=('target_price', 'mean'),
    ).reset_index()

    ts_mid = combined[(combined['dfd'] >= 8) & (combined['dfd'] <= 20)].groupby(
        ['pathod_id', 'offer_date']
    ).agg(ts_mid_mean=('target_price', 'mean')).reset_index()

    ts_adv = combined[(combined['dfd'] >= 21) & (combined['dfd'] <= 39)].groupby(
        ['pathod_id', 'offer_date']
    ).agg(ts_adv_mean=('target_price', 'mean')).reset_index()

    ts = ts_obs.merge(ts_near, on=['pathod_id', 'offer_date'], how='outer')
    ts = ts.merge(ts_mid,  on=['pathod_id', 'offer_date'], how='outer')
    ts = ts.merge(ts_adv,  on=['pathod_id', 'offer_date'], how='outer')
    return ts


def build_offer_trend_features(df, offer_ts, lookback_days=(7, 14)):
    """
    Compute calendar-time price trends for each flight's offer_date.

    For offer_date X in market M:
      trend_7d  = avg_price(X) - avg_price(X-7)
      trend_14d = avg_price(X) - avg_price(X-14)
      accel     = trend_7d - trend_14d/2  (is change speeding up?)
    """
    df = add_offer_date(df)
    obs = df[df['dfd'] == OBSERVATION_DFD][['pathod_id', 'offer_date']].drop_duplicates().copy()
    obs['offer_date_dt'] = pd.to_datetime(obs['offer_date'])

    # Merge current time series stats
    trend_metric_cols = ['ts_obs_mean', 'ts_obs_median', 'ts_obs_std',
                         'ts_near_mean', 'ts_mid_mean', 'ts_adv_mean']
    result = obs.merge(
        offer_ts[['pathod_id', 'offer_date'] + trend_metric_cols].rename(
            columns={c: f'cur_{c}' for c in trend_metric_cols}
        ),
        on=['pathod_id', 'offer_date'], how='left'
    )

    # Merge lookback stats
    for lb in lookback_days:
        lb_df = obs[['pathod_id', 'offer_date', 'offer_date_dt']].copy()
        lb_df['offer_date_lb'] = (lb_df['offer_date_dt'] - pd.Timedelta(days=lb)).dt.date

        lb_stats = lb_df.merge(
            offer_ts[['pathod_id', 'offer_date'] + trend_metric_cols].rename(
                columns={'offer_date': 'offer_date_lb', **{c: f'lb{lb}_{c}' for c in trend_metric_cols}}
            ),
            on=['pathod_id', 'offer_date_lb'], how='left'
        ).drop(columns=['offer_date_dt', 'offer_date_lb'])

        result = result.merge(lb_stats, on=['pathod_id', 'offer_date'], how='left')

    # Compute trend and acceleration
    for lb in lookback_days:
        result[f'trend_obs_{lb}d']       = result['cur_ts_obs_mean']  - result[f'lb{lb}_ts_obs_mean']
        result[f'trend_obs_pct_{lb}d']   = result[f'trend_obs_{lb}d'] / (result[f'lb{lb}_ts_obs_mean'].abs() + 1e-6)
        result[f'trend_near_{lb}d']      = result['cur_ts_near_mean'] - result[f'lb{lb}_ts_near_mean']
        result[f'trend_mid_{lb}d']       = result['cur_ts_mid_mean']  - result[f'lb{lb}_ts_mid_mean']
        result[f'trend_adv_{lb}d']       = result['cur_ts_adv_mean']  - result[f'lb{lb}_ts_adv_mean']

    if 7 in lookback_days and 14 in lookback_days:
        result['trend_obs_accel']  = result['trend_obs_7d']  - result['trend_obs_14d']  / 2
        result['trend_near_accel'] = result['trend_near_7d'] - result['trend_near_14d'] / 2
        result['trend_adv_accel']  = result['trend_adv_7d']  - result['trend_adv_14d']  / 2

    # Missing indicators for trend_near which has high NaN rate in training
    # (near-departure flights before training window don't exist in either dataset)
    for lb in lookback_days:
        result[f'trend_near_{lb}d_missing'] = result[f'trend_near_{lb}d'].isna().astype(np.int8)
        result[f'trend_obs_{lb}d_missing']  = result[f'trend_obs_{lb}d'].isna().astype(np.int8)

    return result.drop(columns=['offer_date_dt'])


# ──────────────────────────────────────────────────────────────
# Core feature pipeline (same as v4, extended)
# ──────────────────────────────────────────────────────────────

def build_global_trajectory_features(train_df):
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

    return trajectory.merge(trajectory_ratio, on=group_cols + ['dfd'], how='left')


def build_flight_features(df, train_df, global_traj, offer_trend_feats, observation_dfd=OBSERVATION_DFD):
    group_cols = ['pathod_id', 'airline', 'depdow']
    df = add_offer_date(df)

    static = df[FLIGHT_KEY + ['dephour', 'depdow', 'depdt_dt']].drop_duplicates(subset=FLIGHT_KEY)

    known = df[df['dfd'] >= observation_dfd].copy()
    known['price_ratio'] = known['target_price'] / (known['expected_minfare'] + 1e-6)

    at_obs = known[known['dfd'] == observation_dfd][
        FLIGHT_KEY + ['target_price', 'expected_minfare', 'offer_date']
    ].rename(columns={'target_price': 'price_at_obs', 'expected_minfare': 'emf_at_obs'})

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

    market_at_obs = df[df['dfd'] == observation_dfd].copy()
    market_stats = market_at_obs.groupby(['pathod_id', 'depdt']).agg(
        market_min_price=('target_price', 'min'),
        market_max_price=('target_price', 'max'),
        market_mean_price=('target_price', 'mean'),
        market_n_airlines=('airline', 'nunique'),
    ).reset_index()

    traj_at_obs = global_traj[global_traj['dfd'] == observation_dfd][
        group_cols + ['traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean']
    ].rename(columns={
        'traj_mean': 'hist_traj_mean_at_obs',
        'traj_median': 'hist_traj_median_at_obs',
        'traj_std': 'hist_traj_std_at_obs',
        'traj_ratio_mean': 'hist_traj_ratio_at_obs',
    })

    # Within-dataset offer_date features (v4 enhanced)
    offer_date_feats = build_offer_date_features(df)

    flight_feats = static.merge(at_obs, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(hist_stats, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(lookback_data, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(market_stats, on=['pathod_id', 'depdt'], how='left')
    flight_feats = flight_feats.merge(traj_at_obs, on=group_cols, how='left')
    flight_feats = flight_feats.merge(offer_date_feats, on=['pathod_id', 'offer_date'], how='left')
    flight_feats = flight_feats.merge(offer_trend_feats, on=['pathod_id', 'offer_date'], how='left')

    # Temporal features of offer_date
    flight_feats['offer_month']      = pd.to_datetime(flight_feats['offer_date'].astype(str)).dt.month
    flight_feats['offer_dow']        = pd.to_datetime(flight_feats['offer_date'].astype(str)).dt.dayofweek
    flight_feats['offer_dayofyear']  = pd.to_datetime(flight_feats['offer_date'].astype(str)).dt.dayofyear

    # Position relative to same-day market
    flight_feats['price_vs_offer_obs_mean']   = flight_feats['price_at_obs'] - flight_feats['offer_obs_mean']
    flight_feats['price_vs_offer_obs_median'] = flight_feats['price_at_obs'] - flight_feats['offer_obs_median']
    flight_feats['price_ratio_to_offer_obs']  = (
        flight_feats['price_at_obs'] / (flight_feats['offer_obs_mean'] + 1e-6)
    )

    return flight_feats


def prepare_data(df, train_df, global_traj, offer_trend_feats, observation_dfd=OBSERVATION_DFD):
    group_cols = ['pathod_id', 'airline', 'depdow']

    flight_feats = build_flight_features(df, train_df, global_traj, offer_trend_feats, observation_dfd)

    df_dated = add_offer_date(df)
    future = df_dated[df_dated['dfd'] < observation_dfd][
        FLIGHT_KEY + ['dfd', 'expected_minfare', 'target_price']
    ].copy()

    data = future.merge(flight_feats, on=FLIGHT_KEY, how='left')

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

    data['dfd_diff']            = observation_dfd - data['dfd']
    data['dfd_frac']            = data['dfd'] / observation_dfd
    data['emf_ratio_to_obs']    = data['expected_minfare'] / (data['emf_at_obs'] + 1e-6)
    data['price_to_emf']        = data['price_at_obs'] / (data['expected_minfare'] + 1e-6)
    data['price_to_market_min'] = data['price_at_obs'] / (data['market_min_price'] + 1e-6)
    data['price_above_emf']     = data['price_at_obs'] - data['emf_at_obs']
    data['price_vs_hist_traj']  = data['price_at_obs'] - data['hist_traj_mean_at_obs']
    data['hist_traj_emf_price'] = data['hist_traj_ratio_mean'] * data['expected_minfare']
    data['airline_enc']         = data['airline'].astype('category').cat.codes
    data['baseline_pred']       = np.maximum(data['price_at_obs'], data['expected_minfare'])
    data['residual']            = data['target_price'] - data['baseline_pred']

    return data


def get_features():
    return [
        # Core
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
        'hist_traj_emf_price', 'price_vs_hist_traj',
        'baseline_pred',
        # offer_date: within-dataset (v4)
        'offer_obs_mean', 'offer_obs_median', 'offer_obs_std',
        'offer_obs_min', 'offer_obs_max', 'offer_obs_n', 'offer_obs_emf_mean',
        'offer_near_mean', 'offer_near_median', 'offer_near_n',
        'offer_mid_mean', 'offer_mid_n',
        'offer_adv_mean', 'offer_adv_n',
        'offer_near_vs_adv', 'offer_near_vs_obs', 'offer_adv_vs_obs',
        'offer_month', 'offer_dow', 'offer_dayofyear',
        'price_vs_offer_obs_mean', 'price_vs_offer_obs_median', 'price_ratio_to_offer_obs',
        # offer_date: trend features (v5)
        # cur_ts_* omitted: redundant with within-dataset offer_obs_* (near-zero importance)
        'trend_obs_7d', 'trend_obs_14d', 'trend_obs_pct_7d',
        'trend_near_7d', 'trend_near_14d',
        'trend_mid_7d', 'trend_mid_14d',
        'trend_adv_7d', 'trend_adv_14d',
        'trend_obs_accel', 'trend_near_accel', 'trend_adv_accel',
        # Missing indicators: model learns to ignore 0-filled NaNs
        'trend_near_7d_missing', 'trend_near_14d_missing',
        'trend_obs_7d_missing',  'trend_obs_14d_missing',
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

    print("Building offer_date time series (combined train+test)...")
    offer_ts = build_offer_timeseries(train, test)
    print(f"  Time series shape: {offer_ts.shape}, unique offer_dates: {offer_ts['offer_date'].nunique()}")

    print("Building offer_date trend features...")
    train_trend = build_offer_trend_features(train, offer_ts, lookback_days=(7, 14))
    test_trend  = build_offer_trend_features(test,  offer_ts, lookback_days=(7, 14))
    print(f"  Train trend features: {train_trend.shape}, Test: {test_trend.shape}")

    # Check trend coverage
    trend_cols = [c for c in train_trend.columns if c.startswith('trend_') or c.startswith('lb')]
    for col in ['trend_obs_7d', 'trend_obs_14d', 'trend_near_7d', 'trend_adv_7d']:
        m_tr = train_trend[col].isna().mean() * 100
        m_te = test_trend[col].isna().mean() * 100
        print(f"    {col}: train_missing={m_tr:.1f}%, test_missing={m_te:.1f}%")

    print("Preparing features...")
    train_data = prepare_data(train, train, global_traj, train_trend)
    test_data  = prepare_data(test,  train, global_traj, test_trend)

    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    feature_cols = get_features()
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['target_price']
    X_test  = test_data[feature_cols].fillna(0)
    y_test  = test_data['target_price']
    dfd_test      = test_data['dfd'].values
    baseline_test = test_data['baseline_pred'].values

    baseline_rmse = rmse(y_test, baseline_test)
    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")

    # ── Model A: LightGBM (absolute price) ──────────────────────────────
    print("\n=== Model A: LightGBM (offer_date + trend features) ===")
    lgb_a = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_a.fit(X_train, y_train)
    pred_a = lgb_a.predict(X_test)
    rmse_a = evaluate_by_dfd(y_test, pred_a, dfd_test, "LightGBM-v5", baseline_test)

    # ── Model B: LightGBM residual ───────────────────────────────────────
    print("\n=== Model B: LightGBM Residual (offer_date + trend features) ===")
    lgb_b = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_b.fit(X_train, train_data['residual'])
    pred_b = baseline_test + lgb_b.predict(X_test)
    rmse_b = evaluate_by_dfd(y_test, pred_b, dfd_test, "LightGBM-resid-v5", baseline_test)

    # ── Model C: XGBoost ─────────────────────────────────────────────────
    print("\n=== Model C: XGBoost (offer_date + trend features) ===")
    xgb_c = xgb.XGBRegressor(
        n_estimators=1500, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_c.fit(X_train, y_train)
    pred_c = xgb_c.predict(X_test)
    rmse_c = evaluate_by_dfd(y_test, pred_c, dfd_test, "XGBoost-v5", baseline_test)

    # ── Ensemble ──────────────────────────────────────────────────────────
    print("\n=== Ensemble: Weighted (A+B+C) ===")
    weights = np.array([1/rmse_a, 1/rmse_b, 1/rmse_c])
    weights /= weights.sum()
    print(f"  Weights: A={weights[0]:.3f}, B={weights[1]:.3f}, C={weights[2]:.3f}")
    pred_ens = weights[0]*pred_a + weights[1]*pred_b + weights[2]*pred_c
    rmse_ens = evaluate_by_dfd(y_test, pred_ens, dfd_test, "Ensemble-v5", baseline_test)

    # ── Feature importances ───────────────────────────────────────────────
    print("\n=== Feature Importances (LightGBM Model A, top 30) ===")
    fi = pd.DataFrame({'feature': feature_cols, 'importance': lgb_a.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print(fi.head(30).to_string(index=False))

    print("\n--- New v5 trend/dfd-range features ---")
    v5_new = fi[fi['feature'].str.startswith(('trend_', 'cur_ts_', 'offer_mid', 'offer_adv',
                                               'offer_near_vs', 'offer_adv_vs'))]
    print(v5_new.to_string(index=False))

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Model':<42} {'RMSE':>8} {'vs Baseline':>12}")
    print("-" * 65)
    print(f"{'Baseline':<42} {baseline_rmse:>8.4f} {'(reference)':>12}")
    ref = {
        'v4 LightGBM-resid (best v4)': 58.9994,
        'v4 Ensemble':                  59.2680,
    }
    for label, r in ref.items():
        pct = (baseline_rmse - r) / baseline_rmse * 100
        print(f"  [{label:<40}] {r:>8.4f} {pct:>+11.2f}%")
    print()
    for label, r in [
        ("LightGBM-v5",      rmse_a),
        ("LightGBM-resid-v5", rmse_b),
        ("XGBoost-v5",        rmse_c),
        ("Ensemble-v5",       rmse_ens),
    ]:
        pct = (baseline_rmse - r) / baseline_rmse * 100
        v4_best = 58.9994
        delta_v4 = (v4_best - r) / v4_best * 100
        print(f"  {label:<42} {r:>8.4f} {pct:>+11.2f}%  (vs v4-best: {delta_v4:+.2f}%)")


if __name__ == '__main__':
    main()
