"""
Flight price prediction model - Final.

Best approach: Ensemble of LightGBM variants + XGBoost.
Key features:
- Price history at dfd 40-49
- Market competition features
- Historical trajectory features
- Price trend
- Baseline prediction as feature
Post-processing: max(pred, expected_minfare)
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
GROUP_COLS = ['pathod_id', 'airline', 'depdow']


def load_data():
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')
    return train, test


def build_global_trajectory(train_df):
    """Historical average price trajectories per group from training data."""
    train_df2 = train_df.copy()
    train_df2['price_ratio'] = train_df2['target_price'] / (train_df2['expected_minfare'] + 1e-6)

    traj = train_df2.groupby(GROUP_COLS + ['dfd']).agg(
        traj_mean=('target_price', 'mean'),
        traj_median=('target_price', 'median'),
        traj_std=('target_price', 'std'),
        traj_ratio_mean=('price_ratio', 'mean'),
        traj_ratio_std=('price_ratio', 'std'),
    ).reset_index()
    return traj


def build_features(df, global_traj, observation_dfd=OBSERVATION_DFD):
    """Full feature engineering pipeline."""
    # Static per-flight features
    static = df[FLIGHT_KEY + ['dephour', 'depdow']].drop_duplicates(subset=FLIGHT_KEY)

    # Data at and above observation_dfd (known)
    known = df[df['dfd'] >= observation_dfd].copy()
    known['price_ratio'] = known['target_price'] / (known['expected_minfare'] + 1e-6)

    # Price at exactly observation_dfd
    at_obs = known[known['dfd'] == observation_dfd][
        FLIGHT_KEY + ['target_price', 'expected_minfare']
    ].rename(columns={'target_price': 'price_at_obs', 'expected_minfare': 'emf_at_obs'})

    # Known history stats
    hist_stats = known.groupby(FLIGHT_KEY).agg(
        hist_mean=('target_price', 'mean'),
        hist_std=('target_price', 'std'),
        hist_max=('target_price', 'max'),
        hist_min=('target_price', 'min'),
        hist_range=('target_price', lambda x: x.max() - x.min()),
        hist_ratio_mean=('price_ratio', 'mean'),
        hist_ratio_std=('price_ratio', 'std'),
        hist_ratio_max=('price_ratio', 'max'),
        hist_ratio_min=('price_ratio', 'min'),
    ).reset_index()

    # Prices at each lookback dfd
    lookback = at_obs[FLIGHT_KEY].copy()
    for dfd_val in range(observation_dfd, min(observation_dfd + 10, 50)):
        sub = df[df['dfd'] == dfd_val][FLIGHT_KEY + ['target_price']].rename(
            columns={'target_price': f'p{dfd_val}'}
        )
        lookback = lookback.merge(sub, on=FLIGHT_KEY, how='left')

    # Price trend (linear slope over dfd=40..49)
    price_cols_hist = [f'p{d}' for d in range(observation_dfd, min(observation_dfd + 10, 50))]

    def slope(row):
        vals = [row.get(c, np.nan) for c in price_cols_hist]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2:
            return 0.0
        return float(np.polyfit(range(len(vals)), vals, 1)[0])

    lookback['price_trend'] = lookback.apply(slope, axis=1)
    lookback['price_change'] = lookback.get('p40', np.nan) - lookback.get('p49', np.nan)

    # Market competition at observation_dfd
    market_at_obs = df[df['dfd'] == observation_dfd].copy()
    market_stats = market_at_obs.groupby(['pathod_id', 'depdt']).agg(
        mkt_min=('target_price', 'min'),
        mkt_max=('target_price', 'max'),
        mkt_mean=('target_price', 'mean'),
        mkt_med=('target_price', 'median'),
        mkt_std=('target_price', 'std'),
        mkt_n=('airline', 'nunique'),
    ).reset_index()

    # Trajectory at observation_dfd
    traj_at_obs = global_traj[global_traj['dfd'] == observation_dfd][
        GROUP_COLS + ['traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean']
    ].rename(columns={c: c + '_obs' for c in ['traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean']})

    # Combine flight-level features
    ff = static.merge(at_obs, on=FLIGHT_KEY, how='left')
    ff = ff.merge(hist_stats, on=FLIGHT_KEY, how='left')
    ff = ff.merge(lookback, on=FLIGHT_KEY, how='left')
    ff = ff.merge(market_stats, on=['pathod_id', 'depdt'], how='left')
    ff = ff.merge(traj_at_obs, on=GROUP_COLS, how='left')

    # Future rows
    future = df[df['dfd'] < observation_dfd][
        FLIGHT_KEY + ['dfd', 'expected_minfare', 'target_price']
    ].copy()

    # Merge flight features
    data = future.merge(ff, on=FLIGHT_KEY, how='left')

    # Merge target-dfd trajectory
    traj_target = global_traj[GROUP_COLS + ['dfd', 'traj_mean', 'traj_median', 'traj_std',
                                             'traj_ratio_mean', 'traj_ratio_std']]
    data = data.merge(traj_target, on=GROUP_COLS + ['dfd'], how='left')

    # Derived features
    data['dfd_diff'] = observation_dfd - data['dfd']
    data['dfd_frac'] = data['dfd'] / observation_dfd
    data['emf_ratio'] = data['expected_minfare'] / (data['emf_at_obs'] + 1e-6)
    data['price_to_emf'] = data['price_at_obs'] / (data['expected_minfare'] + 1e-6)
    data['price_to_mkt_min'] = data['price_at_obs'] / (data['mkt_min'] + 1e-6)
    data['price_to_mkt_mean'] = data['price_at_obs'] / (data['mkt_mean'] + 1e-6)
    data['price_vs_traj'] = data['price_at_obs'] - data['traj_mean_obs']
    data['traj_emf_pred'] = data['traj_ratio_mean'] * data['expected_minfare']
    data['airline_enc'] = data['airline'].astype('category').cat.codes
    data['baseline'] = np.maximum(data['price_at_obs'], data['expected_minfare'])
    data['residual'] = data['target_price'] - data['baseline']

    return data


def features():
    return [
        'dfd', 'dfd_diff', 'dfd_frac',
        'dephour', 'depdow', 'airline_enc', 'pathod_id',
        'price_at_obs', 'emf_at_obs', 'expected_minfare',
        'emf_ratio', 'price_to_emf', 'price_to_mkt_min', 'price_to_mkt_mean',
        'hist_mean', 'hist_std', 'hist_max', 'hist_min', 'hist_range',
        'hist_ratio_mean', 'hist_ratio_std', 'hist_ratio_max', 'hist_ratio_min',
        'price_trend', 'price_change',
        'mkt_min', 'mkt_max', 'mkt_mean', 'mkt_med', 'mkt_std', 'mkt_n',
        'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49',
        'traj_mean_obs', 'traj_median_obs', 'traj_std_obs', 'traj_ratio_mean_obs',
        'traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean', 'traj_ratio_std',
        'traj_emf_pred', 'price_vs_traj',
        'baseline',
    ]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    print("Loading data...")
    train, test = load_data()

    print("Building trajectory features...")
    global_traj = build_global_trajectory(train)

    print("Building feature matrices...")
    train_data = build_features(train, global_traj)
    test_data = build_features(test, global_traj)

    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")

    feat_cols = features()
    X_tr = train_data[feat_cols].fillna(0)
    y_tr = train_data['target_price']
    X_te = test_data[feat_cols].fillna(0)
    y_te = test_data['target_price']
    dfd_te = test_data['dfd'].values
    emf_te = test_data['expected_minfare'].values
    baseline_te = test_data['baseline'].values

    bl_rmse = rmse(y_te, baseline_te)
    print(f"\nBaseline RMSE: {bl_rmse:.4f}")

    lgb_params_abs = dict(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
        num_leaves=127, reg_alpha=0.05, reg_lambda=0.05,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_params_resid = dict(
        n_estimators=2000, max_depth=7, learning_rate=0.01,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
        num_leaves=63, reg_alpha=0.1, reg_lambda=0.1,
        random_state=123, n_jobs=-1, verbose=-1,
    )
    xgb_params = dict(
        n_estimators=1500, max_depth=7, learning_rate=0.015,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    print("\nTraining Model A (LightGBM absolute)...")
    m_a = lgb.LGBMRegressor(**lgb_params_abs)
    m_a.fit(X_tr, y_tr)
    p_a = np.maximum(m_a.predict(X_te), emf_te)

    print("Training Model B (LightGBM residual)...")
    m_b = lgb.LGBMRegressor(**lgb_params_resid)
    m_b.fit(X_tr, train_data['residual'])
    p_b = np.maximum(baseline_te + m_b.predict(X_te), emf_te)

    print("Training Model C (XGBoost absolute)...")
    m_c = xgb.XGBRegressor(**xgb_params)
    m_c.fit(X_tr, y_tr)
    p_c = np.maximum(m_c.predict(X_te), emf_te)

    print("Training Model D (LightGBM abs, different seed)...")
    params_d = {**lgb_params_abs, 'random_state': 999, 'num_leaves': 63, 'max_depth': 7}
    m_d = lgb.LGBMRegressor(**params_d)
    m_d.fit(X_tr, y_tr)
    p_d = np.maximum(m_d.predict(X_te), emf_te)

    # Ensemble
    p_ens3 = (p_a + p_b + p_c) / 3
    p_ens4 = (p_a + p_b + p_c + p_d) / 4

    print("\n=== RESULTS ===")
    print(f"{'Model':<35} {'RMSE':>8} {'vs Baseline':>12}")
    print("-" * 58)
    print(f"{'Baseline':<35} {bl_rmse:>8.4f} {'---':>12}")

    results = []
    for label, p in [
        ("LightGBM-abs (A)", p_a),
        ("LightGBM-resid (B)", p_b),
        ("XGBoost (C)", p_c),
        ("LightGBM-abs2 (D)", p_d),
        ("Ensemble A+B+C", p_ens3),
        ("Ensemble A+B+C+D", p_ens4),
    ]:
        r = rmse(y_te, p)
        pct = (bl_rmse - r) / bl_rmse * 100
        results.append((label, r, pct))
        print(f"  {label:<33} {r:>8.4f} {pct:>+11.2f}%")

    # Best model breakdown by dfd
    best_label, best_rmse_val, _ = min(results, key=lambda x: x[1])
    best_preds = {
        "LightGBM-abs (A)": p_a,
        "LightGBM-resid (B)": p_b,
        "XGBoost (C)": p_c,
        "LightGBM-abs2 (D)": p_d,
        "Ensemble A+B+C": p_ens3,
        "Ensemble A+B+C+D": p_ens4,
    }
    best_p = best_preds[best_label]
    print(f"\nBest model: {best_label} (RMSE={best_rmse_val:.4f})")
    print("RMSE by dfd:")
    for dfd_val in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        mask = dfd_te == dfd_val
        if mask.sum() > 0:
            r = rmse(y_te[mask], best_p[mask])
            bl = rmse(y_te[mask], baseline_te[mask])
            imp = (bl - r) / bl * 100
            print(f"  dfd={dfd_val:2d}: RMSE={r:.4f} vs baseline={bl:.4f}  ({imp:+.2f}%)")

    # Feature importances from best LightGBM
    print("\nTop 15 Feature Importances (LightGBM-abs):")
    fi = pd.DataFrame({'feature': feat_cols, 'importance': m_a.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print(fi.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
