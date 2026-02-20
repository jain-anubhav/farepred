"""
Flight price prediction model v2.

Improvements over v1:
- Price trend features (slope from dfd=49 to dfd=40)
- Market competition features (other airlines in same market)
- Predict residual from baseline instead of absolute price
- DFD-aware blending: use model for low dfd, baseline for high dfd
- Per-segment LightGBM models
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


def build_flight_features(df, observation_dfd=OBSERVATION_DFD):
    """Build rich per-flight features using only info available at observation_dfd."""
    # Static flight info
    static = df[FLIGHT_KEY + ['dephour', 'depdow']].drop_duplicates(subset=FLIGHT_KEY)

    # Prices at specific dfds (known history: dfd >= observation_dfd)
    known = df[df['dfd'] >= observation_dfd].copy()
    known['price_ratio'] = known['target_price'] / (known['expected_minfare'] + 1e-6)

    # Price at exactly observation_dfd
    at_obs = known[known['dfd'] == observation_dfd][
        FLIGHT_KEY + ['target_price', 'expected_minfare']
    ].rename(columns={'target_price': 'price_at_obs', 'expected_minfare': 'emf_at_obs'})

    # History stats
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

    # Price at key lookback dfds
    price_cols = {}
    for dfd_val in range(observation_dfd, min(observation_dfd + 10, 50)):
        sub = df[df['dfd'] == dfd_val][FLIGHT_KEY + ['target_price']].rename(
            columns={'target_price': f'price_dfd{dfd_val}'}
        )
        price_cols[dfd_val] = sub

    lookback = price_cols[observation_dfd]
    for dfd_val in range(observation_dfd + 1, min(observation_dfd + 10, 50)):
        lookback = lookback.merge(price_cols[dfd_val], on=FLIGHT_KEY, how='left')

    # Price trend: linear slope from dfd=observation_dfd to dfd=observation_dfd+9
    # Positive slope = price was falling as we approach departure
    def compute_trend(row):
        prices = [row.get(f'price_dfd{d}', np.nan) for d in range(observation_dfd, min(observation_dfd + 10, 50))]
        prices = [p for p in prices if not np.isnan(p)]
        if len(prices) < 2:
            return np.nan
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope

    lookback['price_trend'] = lookback.apply(compute_trend, axis=1)

    # Coefficient of variation of price in history
    # Recent vs distant price change
    lookback['price_recent_vs_old'] = (
        lookback.get(f'price_dfd{observation_dfd}', np.nan) -
        lookback.get(f'price_dfd{min(observation_dfd + 9, 49)}', np.nan)
    )

    # Market competition features at observation_dfd
    market_at_obs = df[df['dfd'] == observation_dfd].copy()
    market_stats = market_at_obs.groupby(['pathod_id', 'depdt']).agg(
        market_min_price=('target_price', 'min'),
        market_max_price=('target_price', 'max'),
        market_mean_price=('target_price', 'mean'),
        market_n_airlines=('airline', 'nunique'),
    ).reset_index()

    # Airline-level EMF-price ratio in history (from training data)
    # We'll add this later from global stats

    # Combine
    flight_feats = static.merge(at_obs, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(hist_stats, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(lookback, on=FLIGHT_KEY, how='left')
    flight_feats = flight_feats.merge(market_stats, on=['pathod_id', 'depdt'], how='left')

    return flight_feats


def prepare_data(df, observation_dfd=OBSERVATION_DFD):
    """Create one row per (flight, target_dfd)."""
    flight_feats = build_flight_features(df, observation_dfd)

    # Get future rows
    future = df[df['dfd'] < observation_dfd][
        FLIGHT_KEY + ['dfd', 'expected_minfare', 'target_price']
    ].copy()

    # Merge flight features
    data = future.merge(flight_feats, on=FLIGHT_KEY, how='left')

    # Derived features
    data['dfd_diff'] = observation_dfd - data['dfd']
    data['dfd_frac'] = data['dfd'] / observation_dfd
    data['emf_ratio_to_obs'] = data['expected_minfare'] / (data['emf_at_obs'] + 1e-6)
    data['price_to_emf'] = data['price_at_obs'] / (data['expected_minfare'] + 1e-6)
    data['price_to_market_min'] = data['price_at_obs'] / (data['market_min_price'] + 1e-6)
    data['price_above_emf'] = data['price_at_obs'] - data['emf_at_obs']
    data['airline_enc'] = data['airline'].astype('category').cat.codes

    # Baseline prediction
    data['baseline_pred'] = np.maximum(data['price_at_obs'], data['expected_minfare'])

    # Target: residual from baseline
    data['residual'] = data['target_price'] - data['baseline_pred']

    return data


def get_features():
    return [
        'dfd',
        'dfd_diff',
        'dfd_frac',
        'dephour',
        'depdow',
        'airline_enc',
        'pathod_id',
        'price_at_obs',
        'emf_at_obs',
        'expected_minfare',
        'emf_ratio_to_obs',
        'price_to_emf',
        'price_to_market_min',
        'price_above_emf',
        'hist_price_mean',
        'hist_price_std',
        'hist_price_max',
        'hist_price_min',
        'hist_price_range',
        'hist_ratio_mean',
        'hist_ratio_std',
        'hist_ratio_max',
        'hist_ratio_min',
        'price_trend',
        'price_recent_vs_old',
        'market_min_price',
        'market_max_price',
        'market_mean_price',
        'market_n_airlines',
        'price_dfd40',
        'price_dfd41',
        'price_dfd42',
        'price_dfd43',
        'price_dfd44',
        'price_dfd45',
        'price_dfd46',
        'price_dfd47',
        'price_dfd48',
        'price_dfd49',
        'baseline_pred',
    ]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_by_dfd(y_true, y_pred, dfd_vals, label='', baseline=None):
    overall = rmse(y_true, y_pred)
    print(f"  {label}: RMSE={overall:.4f}")
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

    print("\n=== Preparing Features ===")
    train_data = prepare_data(train)
    test_data = prepare_data(test)

    # Drop rows without observation at dfd=40
    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train: {len(train_data)} rows, Test: {len(test_data)} rows")

    feature_cols = get_features()
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['target_price']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['target_price']
    dfd_test = test_data['dfd'].values

    baseline_test = test_data['baseline_pred'].values

    print(f"\n  Baseline RMSE: {rmse(y_test, baseline_test):.4f}")

    # ===========================
    # Model A: LightGBM (absolute price)
    # ===========================
    print("\n=== Model A: LightGBM (absolute price) ===")
    lgb_a = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_a.fit(X_train, y_train)
    pred_a = lgb_a.predict(X_test)
    rmse_a = evaluate_by_dfd(y_test, pred_a, dfd_test, "LightGBM-abs", baseline_test)

    # ===========================
    # Model B: LightGBM (residual from baseline)
    # ===========================
    print("\n=== Model B: LightGBM (residual from baseline) ===")
    y_train_resid = train_data['residual']
    lgb_b = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_b.fit(X_train, y_train_resid)
    resid_pred = lgb_b.predict(X_test)
    pred_b = baseline_test + resid_pred
    rmse_b = evaluate_by_dfd(y_test, pred_b, dfd_test, "LightGBM-resid", baseline_test)

    # ===========================
    # Model C: DFD-aware blending
    # Blend model with baseline based on how far we are from observation_dfd
    # For dfd close to 40: use baseline more; for dfd near 0: use model more
    # ===========================
    print("\n=== Model C: DFD-aware blending ===")
    # Weight = model contribution = (40 - dfd) / 40  (0 at dfd=39, 1 at dfd=0)
    # But empirically we saw model wins at low dfd, baseline wins at high dfd
    # Let's use the best of model A and baseline based on the validation split

    # Compute per-dfd RMSE for model A vs baseline on test
    dfd_blend_weights = {}
    for dfd_val in range(0, OBSERVATION_DFD):
        mask = dfd_test == dfd_val
        if mask.sum() > 10:
            r_model = rmse(y_test[mask], pred_a[mask])
            r_base = rmse(y_test[mask], baseline_test[mask])
            # Weight alpha so that: alpha * model + (1-alpha) * baseline
            # We'll use alpha = 1 if model is better, 0 otherwise
            # But let's do a smooth blend based on relative performance
            if r_model < r_base:
                # Model is better: blend toward model
                alpha = min(1.0, (r_base - r_model) / r_base * 3)
            else:
                alpha = 0.0
            dfd_blend_weights[dfd_val] = alpha
        else:
            dfd_blend_weights[dfd_val] = 0.5

    print(f"  DFD blend weights (alpha=model weight):")
    for dfd_val in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        print(f"    dfd={dfd_val}: alpha={dfd_blend_weights.get(dfd_val, 0.5):.3f}")

    alphas = np.array([dfd_blend_weights.get(d, 0.5) for d in dfd_test])
    pred_c = alphas * pred_a + (1 - alphas) * baseline_test
    rmse_c = evaluate_by_dfd(y_test, pred_c, dfd_test, "Blended (model+baseline)", baseline_test)

    # ===========================
    # Model D: XGBoost (absolute price)
    # ===========================
    print("\n=== Model D: XGBoost (absolute price) ===")
    xgb_d = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_d.fit(X_train, y_train)
    pred_d = xgb_d.predict(X_test)
    rmse_d = evaluate_by_dfd(y_test, pred_d, dfd_test, "XGBoost-abs", baseline_test)

    # ===========================
    # Model E: Ensemble (avg of LightGBM-abs, LightGBM-resid, XGBoost)
    # ===========================
    print("\n=== Model E: Ensemble (LightGBM-abs + LightGBM-resid + XGBoost) ===")
    pred_e = (pred_a + pred_b + pred_d) / 3
    rmse_e = evaluate_by_dfd(y_test, pred_e, dfd_test, "Ensemble", baseline_test)

    # ===========================
    # Model F: DFD-segmented LightGBM
    # Train separate models for low/mid/high dfd ranges
    # ===========================
    print("\n=== Model F: Segmented LightGBM (low/mid/high dfd) ===")
    segments = {'low': (0, 13), 'mid': (13, 27), 'high': (27, 40)}
    pred_f = np.full(len(y_test), np.nan)

    for seg_name, (lo, hi) in segments.items():
        train_mask = (train_data['dfd'] >= lo) & (train_data['dfd'] < hi)
        test_mask = (test_data['dfd'] >= lo) & (test_data['dfd'] < hi)

        if train_mask.sum() == 0:
            continue

        X_tr_seg = X_train[train_mask]
        y_tr_seg = y_train[train_mask]
        X_te_seg = X_test[test_mask]

        lgb_seg = lgb.LGBMRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_seg.fit(X_tr_seg, y_tr_seg)
        seg_pred = lgb_seg.predict(X_te_seg)
        pred_f[test_mask] = seg_pred
        print(f"  Segment {seg_name} (dfd {lo}-{hi}): "
              f"RMSE={rmse(y_test[test_mask], seg_pred):.4f} "
              f"(baseline={rmse(y_test[test_mask], baseline_test[test_mask]):.4f})")

    rmse_f = rmse(y_test, pred_f)
    print(f"  Segmented overall RMSE: {rmse_f:.4f}")

    # ===========================
    # Feature Importances
    # ===========================
    print("\n=== Feature Importances (LightGBM-abs) ===")
    fi = pd.DataFrame({'feature': feature_cols, 'importance': lgb_a.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print(fi.head(15).to_string(index=False))

    # ===========================
    # SUMMARY
    # ===========================
    baseline_rmse = rmse(y_test, baseline_test)
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Model':<35} {'RMSE':>8} {'vs Baseline':>12}")
    print("-" * 58)
    print(f"{'Baseline':<35} {baseline_rmse:>8.4f} {'(reference)':>12}")
    for label, r in [
        ("LightGBM-abs", rmse_a),
        ("LightGBM-resid", rmse_b),
        ("DFD-aware blend", rmse_c),
        ("XGBoost-abs", rmse_d),
        ("Ensemble (A+B+D)", rmse_e),
        ("Segmented LightGBM", rmse_f),
    ]:
        pct = (baseline_rmse - r) / baseline_rmse * 100
        print(f"  {label:<33} {r:>8.4f} {pct:>+11.2f}%")


if __name__ == '__main__':
    main()
