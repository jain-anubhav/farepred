#!/usr/bin/env python3
"""
Run all original and agent models, collect predictions, and save to CSV.

Output: predictions_all_models.csv with columns:
  - flt_id, pathod_id, airline, depdt, dfd  (identifiers)
  - target_price                              (ground truth)
  - baseline_pred                             (baseline: max(price_at_obs, emf))
  - <model>_lgb_abs                           (LightGBM absolute price pred)
  - <model>_lgb_resid                         (LightGBM residual pred)
  - <model>_xgb                               (XGBoost pred)
  - <model>_ensemble                          (weighted ensemble pred)
"""
import sys
import os
import importlib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/user/farepred')
sys.path.insert(0, '/home/user/farepred')

FLIGHT_KEY = ['flt_id', 'pathod_id', 'airline', 'depdt']
OBSERVATION_DFD = 40


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_data():
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')
    return train, test


def train_and_predict(X_train, y_train, X_test, y_test, baseline_test, train_data,
                      model_label, use_log_ratio=False):
    """Train LightGBM-abs, LightGBM-resid/log-ratio, XGBoost, and return predictions."""

    # Model A: LightGBM absolute price
    lgb_a = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_a.fit(X_train, y_train)
    pred_a = lgb_a.predict(X_test)
    rmse_a = rmse(y_test, pred_a)

    # Model B: LightGBM residual or log-ratio
    lgb_b = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    if use_log_ratio and 'log_ratio' in train_data.columns:
        lgb_b.fit(X_train, train_data['log_ratio'])
        log_ratio_pred = lgb_b.predict(X_test)
        pred_b = (baseline_test + 1) * np.exp(log_ratio_pred) - 1
        pred_b = np.clip(pred_b, 0, baseline_test * 3)
    else:
        lgb_b.fit(X_train, train_data['residual'])
        pred_b = baseline_test + lgb_b.predict(X_test)
    rmse_b = rmse(y_test, pred_b)

    # Model C: XGBoost absolute price
    xgb_c = xgb.XGBRegressor(
        n_estimators=1500, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_c.fit(X_train, y_train)
    pred_c = xgb_c.predict(X_test)
    rmse_c = rmse(y_test, pred_c)

    # Weighted ensemble (weight by 1/RMSE)
    weights = np.array([1 / rmse_a, 1 / rmse_b, 1 / rmse_c])
    weights = weights / weights.sum()
    pred_ens = weights[0] * pred_a + weights[1] * pred_b + weights[2] * pred_c
    rmse_ens = rmse(y_test, pred_ens)
    rmse_bl = rmse(y_test, baseline_test)

    print(f"  {model_label}: baseline={rmse_bl:.4f} | lgb_abs={rmse_a:.4f} | "
          f"lgb_resid={rmse_b:.4f} | xgb={rmse_c:.4f} | ensemble={rmse_ens:.4f}")

    return {
        'lgb_abs': pred_a,
        'lgb_resid': pred_b,
        'xgb': pred_c,
        'ensemble': pred_ens,
    }


# ─── Model runners ────────────────────────────────────────────────────────────

def run_model_v2(train, test):
    """model_v2: prepare_data(df) — no global trajectory."""
    print("\n[model_v2]")
    mod = importlib.import_module('model_v2')

    train_data = mod.prepare_data(train).dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = mod.prepare_data(test).dropna(subset=['price_at_obs']).reset_index(drop=True)

    feat_cols = mod.get_features()
    X_tr = train_data[feat_cols].fillna(0)
    y_tr = train_data['target_price']
    X_te = test_data[feat_cols].fillna(0)
    y_te = test_data['target_price']
    bl_te = test_data['baseline_pred'].values

    preds = train_and_predict(X_tr, y_tr, X_te, y_te, bl_te, train_data, 'v2')
    return test_data[FLIGHT_KEY + ['dfd', 'target_price', 'baseline_pred']], preds


def run_model_v3_style(train, test, module_name, use_log_ratio=False):
    """model_v3, model_v4, model_agent_v*: prepare_data(df, train_df, global_traj)."""
    print(f"\n[{module_name}]")
    mod = importlib.import_module(module_name)

    global_traj = mod.build_global_trajectory_features(train)
    train_data = mod.prepare_data(train, train, global_traj).dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = mod.prepare_data(test,  train, global_traj).dropna(subset=['price_at_obs']).reset_index(drop=True)

    feat_cols = mod.get_features()
    X_tr = train_data[feat_cols].fillna(0)
    y_tr = train_data['target_price']
    X_te = test_data[feat_cols].fillna(0)
    y_te = test_data['target_price']

    baseline_col = 'baseline_pred' if 'baseline_pred' in test_data.columns else 'baseline'
    bl_te = test_data[baseline_col].values

    preds = train_and_predict(X_tr, y_tr, X_te, y_te, bl_te, train_data,
                              module_name, use_log_ratio=use_log_ratio)
    return test_data[FLIGHT_KEY + ['dfd', 'target_price', baseline_col]], preds


def run_model_v5(train, test):
    """model_v5: prepare_data(df, train_df, global_traj, offer_trend_feats)."""
    print("\n[model_v5]")
    mod = importlib.import_module('model_v5')

    global_traj = mod.build_global_trajectory_features(train)
    offer_ts = mod.build_offer_timeseries(train, test)
    train_trend = mod.build_offer_trend_features(train, offer_ts, lookback_days=(7, 14))
    test_trend  = mod.build_offer_trend_features(test,  offer_ts, lookback_days=(7, 14))

    train_data = mod.prepare_data(train, train, global_traj, train_trend).dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = mod.prepare_data(test,  train, global_traj, test_trend).dropna(subset=['price_at_obs']).reset_index(drop=True)

    feat_cols = mod.get_features()
    X_tr = train_data[feat_cols].fillna(0)
    y_tr = train_data['target_price']
    X_te = test_data[feat_cols].fillna(0)
    y_te = test_data['target_price']
    bl_te = test_data['baseline_pred'].values

    preds = train_and_predict(X_tr, y_tr, X_te, y_te, bl_te, train_data, 'v5')
    return test_data[FLIGHT_KEY + ['dfd', 'target_price', 'baseline_pred']], preds


def run_model_final(train, test):
    """model_final: build_features(df, global_traj) with 4-model ensemble."""
    print("\n[model_final]")
    mod = importlib.import_module('model_final')

    global_traj = mod.build_global_trajectory(train)
    train_data = mod.build_features(train, global_traj).dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = mod.build_features(test,  global_traj).dropna(subset=['price_at_obs']).reset_index(drop=True)

    feat_cols = mod.features()
    X_tr = train_data[feat_cols].fillna(0)
    y_tr = train_data['target_price']
    X_te = test_data[feat_cols].fillna(0)
    y_te = test_data['target_price']
    bl_te = test_data['baseline'].values
    emf_te = test_data['expected_minfare'].values

    # Model A: LightGBM absolute
    m_a = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
        num_leaves=127, reg_alpha=0.05, reg_lambda=0.05,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    m_a.fit(X_tr, y_tr)
    p_a = np.maximum(m_a.predict(X_te), emf_te)

    # Model B: LightGBM residual
    m_b = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.01,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
        num_leaves=63, reg_alpha=0.1, reg_lambda=0.1,
        random_state=123, n_jobs=-1, verbose=-1,
    )
    m_b.fit(X_tr, train_data['residual'])
    p_b = np.maximum(bl_te + m_b.predict(X_te), emf_te)

    # Model C: XGBoost
    m_c = xgb.XGBRegressor(
        n_estimators=1500, max_depth=7, learning_rate=0.015,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    m_c.fit(X_tr, y_tr)
    p_c = np.maximum(m_c.predict(X_te), emf_te)

    # Model D: LightGBM abs, different seed
    m_d = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.01,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
        num_leaves=63, reg_alpha=0.05, reg_lambda=0.05,
        random_state=999, n_jobs=-1, verbose=-1,
    )
    m_d.fit(X_tr, y_tr)
    p_d = np.maximum(m_d.predict(X_te), emf_te)

    # Equal-weight 4-model ensemble
    p_ens = (p_a + p_b + p_c + p_d) / 4

    rmse_bl = rmse(y_te, bl_te)
    print(f"  model_final: baseline={rmse_bl:.4f} | lgb_abs={rmse(y_te, p_a):.4f} | "
          f"lgb_resid={rmse(y_te, p_b):.4f} | xgb={rmse(y_te, p_c):.4f} | "
          f"lgb_abs2={rmse(y_te, p_d):.4f} | ensemble={rmse(y_te, p_ens):.4f}")

    preds = {
        'lgb_abs': p_a,
        'lgb_resid': p_b,
        'xgb': p_c,
        'lgb_abs2': p_d,
        'ensemble': p_ens,
    }
    return test_data[FLIGHT_KEY + ['dfd', 'target_price', 'baseline']], preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  RUNNING ALL MODELS — COLLECTING PREDICTIONS")
    print("=" * 70)

    print("\nLoading data...")
    train, test = load_data()
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")

    models_to_run = [
        # (label,          runner,                          kwargs)
        ('v2',            run_model_v2,                    {}),
        ('v3',            run_model_v3_style,              {'module_name': 'model_v3'}),
        ('v4',            run_model_v3_style,              {'module_name': 'model_v4'}),
        ('v5',            run_model_v5,                    {}),
        ('final',         run_model_final,                 {}),
        ('agent_v1',      run_model_v3_style,              {'module_name': 'model_agent_v1'}),
        ('agent_v2',      run_model_v3_style,              {'module_name': 'model_agent_v2'}),
        ('agent_v3',      run_model_v3_style,              {'module_name': 'model_agent_v3'}),
        ('agent_v4',      run_model_v3_style,              {'module_name': 'model_agent_v4'}),
        ('agent_v5',      run_model_v3_style,              {'module_name': 'model_agent_v5',
                                                            'use_log_ratio': True}),
    ]

    # We'll collect all results keyed by label.
    # Each entry: (base_df, preds_dict)
    results = {}
    errors = {}

    for label, runner, kwargs in models_to_run:
        try:
            base_df, preds = runner(train, test, **kwargs)
            results[label] = (base_df, preds)
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()
            errors[label] = str(e)

    # ── Build the combined output ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BUILDING COMBINED CSV")
    print("=" * 70)

    # Use the first successful result as the spine (identifiers + ground truth)
    spine_label = next(iter(results))
    spine_df, _ = results[spine_label]
    # Normalise baseline column name
    if 'baseline' in spine_df.columns and 'baseline_pred' not in spine_df.columns:
        spine_df = spine_df.rename(columns={'baseline': 'baseline_pred'})

    combined = spine_df.copy()
    combined = combined.rename(columns={'baseline_pred': 'baseline_pred'})

    for label, (base_df, preds) in results.items():
        # Normalise baseline column name in base_df
        if 'baseline' in base_df.columns and 'baseline_pred' not in base_df.columns:
            base_df = base_df.rename(columns={'baseline': 'baseline_pred'})

        for pred_name, pred_values in preds.items():
            col_name = f'{label}_{pred_name}'
            tmp = base_df[FLIGHT_KEY + ['dfd']].copy()
            tmp[col_name] = pred_values
            combined = combined.merge(tmp, on=FLIGHT_KEY + ['dfd'], how='left')

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = '/home/user/farepred/predictions_all_models.csv'
    combined.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Shape: {combined.shape}")
    print(f"Columns ({len(combined.columns)}):")
    for c in combined.columns:
        print(f"  {c}")

    if errors:
        print(f"\nFailed models: {list(errors.keys())}")
        for k, v in errors.items():
            print(f"  {k}: {v}")

    # ── Quick RMSE summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RMSE SUMMARY (test set)")
    print("=" * 70)
    y_true = combined['target_price'].values
    bl = combined['baseline_pred'].values
    print(f"  {'baseline':<30} RMSE={rmse(y_true, bl):.4f}")
    for col in combined.columns:
        if col.endswith('_ensemble') or col.endswith('_lgb_abs') or col.endswith('_xgb'):
            preds_col = combined[col].values
            if not np.all(np.isnan(preds_col)):
                r = rmse(y_true[~np.isnan(preds_col)], preds_col[~np.isnan(preds_col)])
                pct = (rmse(y_true, bl) - r) / rmse(y_true, bl) * 100
                print(f"  {col:<30} RMSE={r:.4f} ({pct:+.2f}% vs baseline)")

    return combined


if __name__ == '__main__':
    main()
