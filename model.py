"""
Flight price prediction model.

Task: At dfd=40, predict price for each future dfd (0-39).
Baseline: prediction = max(price_at_dfd40, expected_minfare_at_target_dfd)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

OBSERVATION_DFD = 40  # We observe prices at this dfd
MAX_DFD = 49


def load_data():
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')
    return train, test


def compute_baseline(df, observation_dfd=OBSERVATION_DFD):
    """Baseline: max(price_at_observation_dfd, expected_minfare_at_target_dfd)"""
    # Get price at observation dfd for each flight
    obs = df[df['dfd'] == observation_dfd][
        ['flt_id', 'pathod_id', 'airline', 'depdt', 'target_price']
    ].rename(columns={'target_price': f'price_at_{observation_dfd}'})

    # Get rows to predict (dfd < observation_dfd)
    future = df[df['dfd'] < observation_dfd].copy()
    future = future.merge(obs, on=['flt_id', 'pathod_id', 'airline', 'depdt'], how='left')

    future['baseline_pred'] = np.maximum(
        future[f'price_at_{observation_dfd}'],
        future['expected_minfare']
    )
    return future


def build_flight_features(df, observation_dfd=OBSERVATION_DFD):
    """
    Build per-flight features using only information available at observation_dfd.
    Each flight gets one set of static features based on dfd >= observation_dfd.
    """
    flight_key = ['flt_id', 'pathod_id', 'airline', 'depdt']

    # Static flight info (take from any row, same per flight)
    static_cols = flight_key + ['dephour', 'depdow']
    static = df[static_cols].drop_duplicates(subset=flight_key)

    # Price at exactly observation_dfd
    at_obs = df[df['dfd'] == observation_dfd][
        flight_key + ['target_price', 'expected_minfare']
    ].rename(columns={
        'target_price': 'price_at_obs',
        'expected_minfare': 'emf_at_obs'
    })

    # Stats from dfd >= observation_dfd (known history)
    history = df[df['dfd'] >= observation_dfd].copy()
    history['price_ratio'] = history['target_price'] / (history['expected_minfare'] + 1e-6)

    hist_stats = history.groupby(flight_key).agg(
        hist_price_mean=('target_price', 'mean'),
        hist_price_std=('target_price', 'std'),
        hist_price_max=('target_price', 'max'),
        hist_price_min=('target_price', 'min'),
        hist_ratio_mean=('price_ratio', 'mean'),
        hist_ratio_std=('price_ratio', 'std'),
        hist_ratio_max=('price_ratio', 'max'),
    ).reset_index()

    # Price at specific recent dfds
    for dfd_val in [40, 41, 42, 43, 44, 45]:
        sub = df[df['dfd'] == dfd_val][flight_key + ['target_price']].rename(
            columns={'target_price': f'price_dfd{dfd_val}'}
        )
        if dfd_val == 40:
            history_prices = sub
        else:
            history_prices = history_prices.merge(sub, on=flight_key, how='left')

    # Combine all flight-level features
    flight_feats = static.merge(at_obs, on=flight_key, how='left')
    flight_feats = flight_feats.merge(hist_stats, on=flight_key, how='left')
    flight_feats = flight_feats.merge(history_prices, on=flight_key, how='left')

    return flight_feats


def prepare_training_data(df, observation_dfd=OBSERVATION_DFD):
    """
    Create training rows: one row per (flight, target_dfd) where target_dfd < observation_dfd.
    Features: flight features + target_dfd + expected_minfare at target_dfd.
    Target: actual price at target_dfd.
    """
    flight_key = ['flt_id', 'pathod_id', 'airline', 'depdt']

    # Get flight-level features
    flight_feats = build_flight_features(df, observation_dfd)

    # Get all future rows (what we want to predict)
    future = df[df['dfd'] < observation_dfd][
        flight_key + ['dfd', 'expected_minfare', 'target_price']
    ].copy()

    # Merge flight features onto future rows
    data = future.merge(flight_feats, on=flight_key, how='left')

    # Add derived features
    data['emf_ratio_to_obs'] = data['expected_minfare'] / (data['emf_at_obs'] + 1e-6)
    data['price_to_emf_ratio'] = data['price_at_obs'] / (data['expected_minfare'] + 1e-6)
    data['baseline_pred'] = np.maximum(data['price_at_obs'], data['expected_minfare'])
    data['dfd_diff'] = observation_dfd - data['dfd']

    # Encode airline
    data['airline_enc'] = data['airline'].astype('category').cat.codes

    return data


def get_feature_cols():
    return [
        'dfd',
        'dfd_diff',
        'dephour',
        'depdow',
        'airline_enc',
        'pathod_id',
        'price_at_obs',
        'emf_at_obs',
        'expected_minfare',
        'emf_ratio_to_obs',
        'price_to_emf_ratio',
        'hist_price_mean',
        'hist_price_std',
        'hist_price_max',
        'hist_price_min',
        'hist_ratio_mean',
        'hist_ratio_std',
        'hist_ratio_max',
        'price_dfd40',
        'price_dfd41',
        'price_dfd42',
        'price_dfd43',
        'price_dfd44',
        'price_dfd45',
        'baseline_pred',
    ]


def evaluate(y_true, y_pred, label=''):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"  {label}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    return rmse


def main():
    print("Loading data...")
    train, test = load_data()

    print("\n=== Baseline Model ===")
    test_baseline = compute_baseline(test)
    # Drop rows where price_at_40 is missing (no observation at dfd=40)
    test_baseline = test_baseline.dropna(subset=['baseline_pred'])
    baseline_rmse = evaluate(
        test_baseline['target_price'],
        test_baseline['baseline_pred'],
        "Baseline (test, dfd<40)"
    )
    print(f"  Baseline RMSE by dfd:")
    for dfd_val in [0, 5, 10, 20, 30, 39]:
        sub = test_baseline[test_baseline['dfd'] == dfd_val]
        if len(sub) > 0:
            rmse = np.sqrt(mean_squared_error(sub['target_price'], sub['baseline_pred']))
            print(f"    dfd={dfd_val}: RMSE={rmse:.4f}, n={len(sub)}")

    print("\n=== Preparing Training Data ===")
    train_data = prepare_training_data(train)
    test_data = prepare_training_data(test)

    # Drop rows where price_at_obs is missing (flight has no dfd=40 record)
    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train rows: {len(train_data)}, Test rows: {len(test_data)}")

    feature_cols = get_feature_cols()
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['target_price']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['target_price']

    # Also align baseline on same test rows
    baseline_test_aligned = test_data['baseline_pred']
    baseline_rmse_aligned = evaluate(y_test, baseline_test_aligned, "Baseline (aligned test rows)")

    # ===========================
    # Model 1: XGBoost
    # ===========================
    print("\n=== Model 1: XGBoost ===")
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = evaluate(y_test, xgb_pred, "XGBoost")
        print(f"  XGBoost improvement over baseline: {(baseline_rmse_aligned - xgb_rmse) / baseline_rmse_aligned * 100:.2f}%")

        print("  XGBoost RMSE by dfd:")
        for dfd_val in [0, 5, 10, 20, 30, 39]:
            mask = test_data['dfd'] == dfd_val
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y_test[mask], xgb_pred[mask]))
                bl_rmse = np.sqrt(mean_squared_error(y_test[mask], baseline_test_aligned[mask]))
                print(f"    dfd={dfd_val}: RMSE={rmse:.4f} (baseline={bl_rmse:.4f}), n={mask.sum()}")
    except Exception as e:
        print(f"  XGBoost failed: {e}")
        xgb_rmse = None

    # ===========================
    # Model 2: LightGBM
    # ===========================
    print("\n=== Model 2: LightGBM ===")
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        lgb_pred = lgb_model.predict(X_test)
        lgb_rmse = evaluate(y_test, lgb_pred, "LightGBM")
        print(f"  LightGBM improvement over baseline: {(baseline_rmse_aligned - lgb_rmse) / baseline_rmse_aligned * 100:.2f}%")

        print("  LightGBM RMSE by dfd:")
        for dfd_val in [0, 5, 10, 20, 30, 39]:
            mask = test_data['dfd'] == dfd_val
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y_test[mask], lgb_pred[mask]))
                bl_rmse = np.sqrt(mean_squared_error(y_test[mask], baseline_test_aligned[mask]))
                print(f"    dfd={dfd_val}: RMSE={rmse:.4f} (baseline={bl_rmse:.4f}), n={mask.sum()}")
    except Exception as e:
        print(f"  LightGBM failed: {e}")
        lgb_rmse = None

    # ===========================
    # Model 3: LightGBM with ratio target
    # ===========================
    print("\n=== Model 3: LightGBM (predict price/emf ratio) ===")
    try:
        import lightgbm as lgb
        # Predict the ratio: target_price / expected_minfare
        y_train_ratio = y_train / (train_data['expected_minfare'] + 1e-6)
        y_test_ratio = y_test / (test_data['expected_minfare'] + 1e-6)

        lgb_ratio_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_ratio_model.fit(X_train, y_train_ratio)
        ratio_pred = lgb_ratio_model.predict(X_test)
        # Convert back to price: ratio * expected_minfare, with floor at expected_minfare
        price_pred_ratio = ratio_pred * test_data['expected_minfare'].values
        price_pred_ratio = np.maximum(price_pred_ratio, test_data['expected_minfare'].values)
        lgb_ratio_rmse = evaluate(y_test, price_pred_ratio, "LightGBM (ratio)")
        print(f"  LightGBM ratio improvement over baseline: {(baseline_rmse_aligned - lgb_ratio_rmse) / baseline_rmse_aligned * 100:.2f}%")
    except Exception as e:
        print(f"  LightGBM ratio model failed: {e}")
        lgb_ratio_rmse = None

    # ===========================
    # Model 4: LightGBM with more features
    # ===========================
    print("\n=== Model 4: LightGBM (enhanced features) ===")
    try:
        import lightgbm as lgb

        # Add market-level features
        flight_key = ['flt_id', 'pathod_id', 'airline', 'depdt']
        market_key = ['pathod_id', 'depdt']

        # Market average prices at dfd=40
        market_stats = train[train['dfd'] == 40].groupby('pathod_id').agg(
            market_avg_price=('target_price', 'mean'),
            market_median_price=('target_price', 'median'),
        ).reset_index()

        train_data2 = train_data.merge(market_stats, on='pathod_id', how='left')
        test_data2 = test_data.merge(market_stats, on='pathod_id', how='left')

        # Add historical ratio by airline
        airline_ratio = train[train['dfd'] >= 40].copy()
        airline_ratio['ratio'] = airline_ratio['target_price'] / (airline_ratio['expected_minfare'] + 1e-6)
        airline_stats = airline_ratio.groupby('airline').agg(
            airline_ratio_mean=('ratio', 'mean'),
        ).reset_index()

        train_data2 = train_data2.merge(airline_stats, on='airline', how='left')
        test_data2 = test_data2.merge(airline_stats, on='airline', how='left')

        feature_cols2 = feature_cols + ['market_avg_price', 'market_median_price', 'airline_ratio_mean']

        X_train2 = train_data2[feature_cols2].fillna(0)
        X_test2 = test_data2[feature_cols2].fillna(0)

        lgb_model2 = lgb.LGBMRegressor(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model2.fit(X_train2, y_train)
        lgb_pred2 = lgb_model2.predict(X_test2)
        lgb2_rmse = evaluate(y_test, lgb_pred2, "LightGBM Enhanced")
        print(f"  LightGBM Enhanced improvement over baseline: {(baseline_rmse_aligned - lgb2_rmse) / baseline_rmse_aligned * 100:.2f}%")

        print("  LightGBM Enhanced RMSE by dfd:")
        for dfd_val in [0, 5, 10, 20, 30, 39]:
            mask = test_data['dfd'] == dfd_val
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y_test[mask], lgb_pred2[mask]))
                bl_rmse = np.sqrt(mean_squared_error(y_test[mask], baseline_test_aligned[mask]))
                print(f"    dfd={dfd_val}: RMSE={rmse:.4f} (baseline={bl_rmse:.4f}), n={mask.sum()}")

        # Feature importances
        fi = pd.DataFrame({'feature': feature_cols2, 'importance': lgb_model2.feature_importances_})
        fi = fi.sort_values('importance', ascending=False)
        print("\n  Top 10 feature importances:")
        print(fi.head(10).to_string(index=False))

    except Exception as e:
        print(f"  LightGBM Enhanced failed: {e}")
        lgb2_rmse = None

    # ===========================
    # Summary
    # ===========================
    print("\n=== SUMMARY ===")
    print(f"Baseline RMSE (test, dfd<40): {baseline_rmse:.4f}")
    print(f"Baseline RMSE (aligned):      {baseline_rmse_aligned:.4f}")
    if xgb_rmse:
        pct = (baseline_rmse_aligned - xgb_rmse) / baseline_rmse_aligned * 100
        print(f"XGBoost RMSE:                 {xgb_rmse:.4f}  ({pct:+.2f}% vs baseline)")
    if lgb_rmse:
        pct = (baseline_rmse_aligned - lgb_rmse) / baseline_rmse_aligned * 100
        print(f"LightGBM RMSE:                {lgb_rmse:.4f}  ({pct:+.2f}% vs baseline)")
    if lgb_ratio_rmse:
        pct = (baseline_rmse_aligned - lgb_ratio_rmse) / baseline_rmse_aligned * 100
        print(f"LightGBM Ratio RMSE:          {lgb_ratio_rmse:.4f}  ({pct:+.2f}% vs baseline)")
    if lgb2_rmse:
        pct = (baseline_rmse_aligned - lgb2_rmse) / baseline_rmse_aligned * 100
        print(f"LightGBM Enhanced RMSE:       {lgb2_rmse:.4f}  ({pct:+.2f}% vs baseline)")


if __name__ == '__main__':
    main()
