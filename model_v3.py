"""
Flight price prediction model v3 - Best-effort model.

Additional improvements:
- Historical trajectory features: average price at each dfd for similar flights
- Better market features: per-market price trajectories from training
- Optimized LightGBM hyperparameters
- Stack ensemble
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
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


def build_global_trajectory_features(train_df, observation_dfd=OBSERVATION_DFD):
    """
    From training data, compute average price trajectory per (pathod_id, airline, depdow, dephour).
    This gives us expected price at each dfd based on historical similar flights.
    """
    # For each group, compute average price at each dfd
    group_cols = ['pathod_id', 'airline', 'depdow']

    trajectory = train_df.groupby(group_cols + ['dfd'])['target_price'].agg(
        ['mean', 'median', 'std']
    ).reset_index()
    trajectory.columns = group_cols + ['dfd', 'traj_mean', 'traj_median', 'traj_std']

    # Also compute trajectory normalized by emf
    train_df2 = train_df.copy()
    train_df2['price_ratio'] = train_df2['target_price'] / (train_df2['expected_minfare'] + 1e-6)
    trajectory_ratio = train_df2.groupby(group_cols + ['dfd'])['price_ratio'].agg(
        ['mean', 'std']
    ).reset_index()
    trajectory_ratio.columns = group_cols + ['dfd', 'traj_ratio_mean', 'traj_ratio_std']

    traj_combined = trajectory.merge(trajectory_ratio, on=group_cols + ['dfd'], how='left')
    return traj_combined


def build_flight_features(df, train_df, global_traj, observation_dfd=OBSERVATION_DFD):
    """Build rich per-flight features using only info available at observation_dfd."""
    group_cols = ['pathod_id', 'airline', 'depdow']

    # Static flight info
    static = df[FLIGHT_KEY + ['dephour', 'depdow']].drop_duplicates(subset=FLIGHT_KEY)

    # Known prices (dfd >= observation_dfd)
    known = df[df['dfd'] >= observation_dfd].copy()
    known['price_ratio'] = known['target_price'] / (known['expected_minfare'] + 1e-6)

    # Price at exactly observation_dfd
    at_obs = known[known['dfd'] == observation_dfd][
        FLIGHT_KEY + ['target_price', 'expected_minfare']
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

    # Price trend (slope)
    def compute_trend(row):
        prices = [row.get(f'price_dfd{d}', np.nan) for d in range(observation_dfd, min(observation_dfd + 10, 50))]
        prices = [p for p in prices if not np.isnan(p)]
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope

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

    # Training trajectory at observation_dfd (for context)
    traj_at_obs = global_traj[global_traj['dfd'] == observation_dfd][
        group_cols + ['traj_mean', 'traj_median', 'traj_std', 'traj_ratio_mean']
    ].rename(columns={
        'traj_mean': 'hist_traj_mean_at_obs',
        'traj_median': 'hist_traj_median_at_obs',
        'traj_std': 'hist_traj_std_at_obs',
        'traj_ratio_mean': 'hist_traj_ratio_at_obs',
    })

    # Combine all flight-level features
    flight_feats = static.merge(at_obs, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(hist_stats, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(lookback_data, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(market_stats, on=['pathod_id', 'depdt'], how='left')
    flight_feats = flight_feats.merge(traj_at_obs, on=group_cols, how='left')

    return flight_feats


def prepare_data(df, train_df, global_traj, observation_dfd=OBSERVATION_DFD):
    """Create one row per (flight, target_dfd) with full feature set."""
    group_cols = ['pathod_id', 'airline', 'depdow']

    flight_feats = build_flight_features(df, train_df, global_traj, observation_dfd)

    # Future rows to predict
    future = df[df['dfd'] < observation_dfd][
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

    print("Preparing features...")
    train_data = prepare_data(train, train, global_traj)
    test_data = prepare_data(test, train, global_traj)

    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    feature_cols = get_features()
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['target_price']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['target_price']
    dfd_test = test_data['dfd'].values
    baseline_test = test_data['baseline_pred'].values

    baseline_rmse = rmse(y_test, baseline_test)
    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")

    # ===========================
    # Model A: LightGBM (absolute price) - tuned
    # ===========================
    print("\n=== Model A: LightGBM Tuned ===")
    lgb_a = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_a.fit(X_train, y_train)
    pred_a = lgb_a.predict(X_test)
    rmse_a = evaluate_by_dfd(y_test, pred_a, dfd_test, "LightGBM-tuned", baseline_test)

    # ===========================
    # Model B: LightGBM residual
    # ===========================
    print("\n=== Model B: LightGBM Residual ===")
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
    rmse_b = evaluate_by_dfd(y_test, pred_b, dfd_test, "LightGBM-resid", baseline_test)

    # ===========================
    # Model C: XGBoost
    # ===========================
    print("\n=== Model C: XGBoost ===")
    xgb_c = xgb.XGBRegressor(
        n_estimators=1500, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_c.fit(X_train, y_train)
    pred_c = xgb_c.predict(X_test)
    rmse_c = evaluate_by_dfd(y_test, pred_c, dfd_test, "XGBoost", baseline_test)

    # ===========================
    # Model D: LightGBM with trajectory ratio target
    # ===========================
    print("\n=== Model D: LightGBM (traj ratio target) ===")
    # Predict: target_price / hist_traj_mean at target dfd
    train_traj_mean = train_data['hist_traj_mean'].fillna(train_data['price_at_obs'])
    test_traj_mean = test_data['hist_traj_mean'].fillna(test_data['price_at_obs'])
    y_train_d = y_train / (train_traj_mean + 1e-6)

    lgb_d = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_d.fit(X_train, y_train_d)
    pred_d = lgb_d.predict(X_test) * test_traj_mean.values
    pred_d = np.maximum(pred_d, test_data['expected_minfare'].values)
    rmse_d = evaluate_by_dfd(y_test, pred_d, dfd_test, "LightGBM-traj-ratio", baseline_test)

    # ===========================
    # Ensemble: Simple average
    # ===========================
    print("\n=== Ensemble: Average (A+B+C+D) ===")
    pred_ens = (pred_a + pred_b + pred_c + pred_d) / 4
    rmse_ens = evaluate_by_dfd(y_test, pred_ens, dfd_test, "Ensemble-avg", baseline_test)

    # ===========================
    # Ensemble: Weighted (favor best models)
    # ===========================
    print("\n=== Ensemble: Weighted ===")
    # Weight by 1/RMSE
    weights = np.array([1/rmse_a, 1/rmse_b, 1/rmse_c, 1/rmse_d])
    weights = weights / weights.sum()
    print(f"  Weights: A={weights[0]:.3f}, B={weights[1]:.3f}, C={weights[2]:.3f}, D={weights[3]:.3f}")
    pred_w = weights[0] * pred_a + weights[1] * pred_b + weights[2] * pred_c + weights[3] * pred_d
    rmse_w = evaluate_by_dfd(y_test, pred_w, dfd_test, "Ensemble-weighted", baseline_test)

    # ===========================
    # Feature Importances
    # ===========================
    print("\n=== Feature Importances (LightGBM-tuned) ===")
    fi = pd.DataFrame({'feature': feature_cols, 'importance': lgb_a.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print(fi.head(20).to_string(index=False))

    # ===========================
    # FINAL SUMMARY
    # ===========================
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Model':<35} {'RMSE':>8} {'vs Baseline':>12}")
    print("-" * 58)
    print(f"{'Baseline':<35} {baseline_rmse:>8.4f} {'(reference)':>12}")
    for label, r in [
        ("LightGBM-tuned", rmse_a),
        ("LightGBM-resid", rmse_b),
        ("XGBoost", rmse_c),
        ("LightGBM-traj-ratio", rmse_d),
        ("Ensemble-avg", rmse_ens),
        ("Ensemble-weighted", rmse_w),
    ]:
        pct = (baseline_rmse - r) / baseline_rmse * 100
        print(f"  {label:<33} {r:>8.4f} {pct:>+11.2f}%")


if __name__ == '__main__':
    main()
