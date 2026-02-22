"""
model_two_stage.py

Two-stage flight price prediction model:

  Stage 1 (Classifier) : For each (flight, target_dfd) pair, predict whether
                         the actual price will be BELOW (-1), SAME (0), or
                         ABOVE (+1) the baseline prediction.
                         Baseline = max(price_at_obs, expected_minfare).

  Stage 2 (Regressor)  : For rows predicted as "changed" (direction != 0),
                         predict the actual target_price.
                         Trained on OOF-predicted "changed" rows to avoid
                         leakage from stage-1 in-sample performance.

Final prediction:
  - stage1 == 0 (same)  → baseline_pred
  - stage1 != 0 (changed) → stage2 regressor prediction

Two routing variants are compared:
  Hard routing : binary switch (same → baseline, changed → stage2)
  Soft routing : blend by P(same): P(same)*baseline + (1-P(same))*stage2

All results are compared against:
  - Baseline heuristic: max(price_at_obs, expected_minfare)
  - model_agent_v5 ensemble (LightGBM-abs + LightGBM-log-ratio + XGBoost)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Reuse all feature engineering from model_agent_v5
from model_agent_v5 import (
    load_data,
    prepare_data,
    get_features,
    build_global_trajectory_features,
    rmse,
    evaluate_by_dfd,
)

# ─────────────────────────────────────────────────────────────────────────────
# Direction labeling
# ─────────────────────────────────────────────────────────────────────────────
SAME_THRESHOLD_PCT = 0.05   # |deviation| / baseline <= 5% → "same" (class 0)


def label_direction(target_price, baseline_pred, threshold=SAME_THRESHOLD_PCT):
    """
    Ternary direction label relative to the baseline prediction:
      +1  (up)   : target > baseline * (1 + threshold)
       0  (same) : |target - baseline| / baseline <= threshold
      -1  (down) : target < baseline * (1 - threshold)
    """
    dev_pct = (target_price - baseline_pred) / (baseline_pred + 1e-6)
    return np.where(dev_pct > threshold, 1,
           np.where(dev_pct < -threshold, -1, 0)).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
LGB_CLF_PARAMS = dict(
    n_estimators=1000, max_depth=6, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    num_leaves=63, reg_alpha=0.1, reg_lambda=0.1,
    class_weight='balanced',
    random_state=42, n_jobs=-1, verbose=-1,
)

LGB_REG_PARAMS = dict(
    n_estimators=2000, max_depth=8, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1,
)

XGB_REG_PARAMS = dict(
    n_estimators=1500, max_depth=7, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbosity=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# V5 Ensemble — reproduced as reference benchmark
# ─────────────────────────────────────────────────────────────────────────────
def train_v5_ensemble(X_train, y_train, log_ratio_train,
                      X_test, y_test, baseline_test, dfd_test):
    """Three-model weighted ensemble from model_agent_v5."""
    print("\n=== [Reference] model_agent_v5 Ensemble ===")

    lgb_a = lgb.LGBMRegressor(**LGB_REG_PARAMS)
    lgb_a.fit(X_train, y_train)
    pred_a = lgb_a.predict(X_test)
    rmse_a = evaluate_by_dfd(y_test, pred_a, dfd_test,
                             "  LightGBM-abs", baseline_test)

    lgb_b = lgb.LGBMRegressor(**LGB_REG_PARAMS)
    lgb_b.fit(X_train, log_ratio_train)
    log_ratio_pred = lgb_b.predict(X_test)
    pred_b = (baseline_test + 1) * np.exp(log_ratio_pred) - 1
    pred_b = np.clip(pred_b, 0, baseline_test * 3)
    rmse_b = evaluate_by_dfd(y_test, pred_b, dfd_test,
                             "  LightGBM-log-ratio", baseline_test)

    xgb_c = xgb.XGBRegressor(**XGB_REG_PARAMS)
    xgb_c.fit(X_train, y_train)
    pred_c = xgb_c.predict(X_test)
    rmse_c = evaluate_by_dfd(y_test, pred_c, dfd_test,
                             "  XGBoost-abs", baseline_test)

    weights = np.array([1 / rmse_a, 1 / rmse_b, 1 / rmse_c])
    weights /= weights.sum()
    print(f"  Weights: A={weights[0]:.3f}, B={weights[1]:.3f}, C={weights[2]:.3f}")
    pred_ens = weights[0] * pred_a + weights[1] * pred_b + weights[2] * pred_c
    rmse_ens = evaluate_by_dfd(y_test, pred_ens, dfd_test,
                               "  Ensemble-weighted", baseline_test)
    return pred_ens, rmse_ens


# ─────────────────────────────────────────────────────────────────────────────
# Two-Stage Model
# ─────────────────────────────────────────────────────────────────────────────
def build_stage2_features(X, dir_proba, dir_pred, class_order):
    """
    Augment feature matrix with stage-1 output:
      p_down, p_same, p_up  : predicted class probabilities
      stage1_direction       : predicted direction {-1, 0, +1}
    """
    X2 = X.copy()
    label_map = {-1: 'p_down', 0: 'p_same', 1: 'p_up'}
    for i, c in enumerate(class_order):
        X2[label_map[c]] = dir_proba[:, i]
    X2['stage1_direction'] = dir_pred
    return X2


def train_two_stage(X_train, y_train, direction_train,
                    baseline_train, X_test, y_test,
                    baseline_test, dfd_test):
    """
    Train the two-stage model and evaluate on the test set.

    Returns (hard_pred, soft_pred, rmse_hard, rmse_soft).
    """
    print("\n=== Two-Stage Model ===")

    # ── Stage 1: Direction Classifier ────────────────────────────────────────
    print(f"\n  [Stage 1] Direction Classifier  (LightGBM, 3-class, "
          f"threshold=±{SAME_THRESHOLD_PCT*100:.0f}% of baseline)")

    n_total = len(direction_train)
    for val, lbl in [(-1, 'down'), (0, 'same'), (1, 'up')]:
        n = (direction_train == val).sum()
        print(f"    Train class '{lbl}': {n:>7,}  ({100 * n / n_total:.1f}%)")

    # Final stage-1 model trained on all training data
    clf1 = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
    clf1.fit(X_train, direction_train)
    class_order = clf1.classes_   # typically [-1, 0, 1]

    # OOF stage-1 predictions — 5-fold, used to define stage-2 training set
    print("    Computing 5-fold OOF stage-1 predictions for stage-2 training...")
    oof_dir_pred  = np.zeros(n_total, dtype=int)
    oof_dir_proba = np.zeros((n_total, len(class_order)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(X_train, direction_train):
        fold_clf = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
        fold_clf.fit(X_train.iloc[tr_idx], direction_train[tr_idx])
        oof_dir_pred[val_idx]  = fold_clf.predict(X_train.iloc[val_idx])
        oof_dir_proba[val_idx] = fold_clf.predict_proba(X_train.iloc[val_idx])

    # Stage-1 test predictions
    test_dir_pred  = clf1.predict(X_test)
    test_dir_proba = clf1.predict_proba(X_test)

    # Stage-1 evaluation
    direction_test_true = label_direction(y_test.values, baseline_test)
    routing_acc = (test_dir_pred == direction_test_true).mean()
    print(f"\n    Stage-1 routing accuracy on test: {routing_acc:.4f}")
    print("    Classification report:")
    print(classification_report(
        direction_test_true, test_dir_pred,
        target_names=['down', 'same', 'up'],
        labels=[-1, 0, 1],
    ))

    # ── Stage 2: Magnitude Regressor ─────────────────────────────────────────
    # Feature matrix augmented with stage-1 output
    X_train_s2_all = build_stage2_features(
        X_train, oof_dir_proba, oof_dir_pred, class_order)
    X_test_s2 = build_stage2_features(
        X_test, test_dir_proba, test_dir_pred, class_order)

    # Train on OOF-predicted "changed" samples only
    changed_mask_train = oof_dir_pred != 0
    n_changed = changed_mask_train.sum()
    print(f"  [Stage 2] Price Regressor on OOF-predicted 'changed' samples "
          f"({n_changed:,} / {n_total:,} rows, "
          f"{100 * n_changed / n_total:.1f}%)")

    reg2 = lgb.LGBMRegressor(**LGB_REG_PARAMS)
    reg2.fit(
        X_train_s2_all[changed_mask_train],
        y_train[changed_mask_train],
    )

    # Stage-2 predictions for all test rows
    stage2_pred_all = reg2.predict(X_test_s2)

    # ── Routing ──────────────────────────────────────────────────────────────
    # Hard routing: predicted "same" → baseline, "changed" → stage-2
    hard_pred = np.where(test_dir_pred == 0, baseline_test, stage2_pred_all)
    hard_pred = np.clip(hard_pred, baseline_test * 0.3, baseline_test * 3.0)

    # Soft routing: blend by P(same)
    # final = P(same) * baseline + (1 - P(same)) * stage2
    same_col_idx = list(class_order).index(0)
    p_same = test_dir_proba[:, same_col_idx]
    soft_pred = p_same * baseline_test + (1 - p_same) * stage2_pred_all
    soft_pred = np.clip(soft_pred, baseline_test * 0.3, baseline_test * 3.0)

    print()
    rmse_hard = evaluate_by_dfd(y_test, hard_pred, dfd_test,
                                "  Two-Stage Hard-Routing", baseline_test)
    rmse_soft = evaluate_by_dfd(y_test, soft_pred, dfd_test,
                                "  Two-Stage Soft-Routing", baseline_test)

    # ── RMSE breakdown by TRUE price direction ────────────────────────────────
    print("\n  RMSE breakdown by TRUE price direction (on test set):")
    for val, lbl in [(-1, 'down'), (0, 'same'), (1, 'up')]:
        mask = direction_test_true == val
        if mask.sum() == 0:
            continue
        r_hard = rmse(y_test[mask], hard_pred[mask])
        r_soft = rmse(y_test[mask], soft_pred[mask])
        r_bl   = rmse(y_test[mask], baseline_test[mask])
        print(f"    {lbl:>4} (n={mask.sum():>6,}):  "
              f"baseline={r_bl:.4f}  "
              f"hard={r_hard:.4f} ({(r_bl-r_hard)/r_bl*100:+.1f}%)  "
              f"soft={r_soft:.4f} ({(r_bl-r_soft)/r_bl*100:+.1f}%)")

    # ── Stage-2 feature importances ───────────────────────────────────────────
    s2_feat_cols = list(X_train_s2_all.columns)
    fi = pd.DataFrame({'feature': s2_feat_cols,
                       'importance': reg2.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print("\n  [Stage 2] Top-15 feature importances:")
    print(fi.head(15).to_string(index=False))

    return hard_pred, soft_pred, rmse_hard, rmse_soft


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    train, test = load_data()

    print("Computing global trajectory features from training data...")
    global_traj = build_global_trajectory_features(train)

    print("Preparing features...")
    train_data = prepare_data(train, train, global_traj)
    test_data  = prepare_data(test,  train, global_traj)

    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train rows: {len(train_data):,}   Test rows: {len(test_data):,}")

    feature_cols    = get_features()
    X_train         = train_data[feature_cols].fillna(0)
    y_train         = train_data['target_price']
    X_test          = test_data[feature_cols].fillna(0)
    y_test          = test_data['target_price']
    dfd_test        = test_data['dfd'].values
    baseline_train  = train_data['baseline_pred'].values
    baseline_test   = test_data['baseline_pred'].values
    log_ratio_train = train_data['log_ratio']

    # Direction labels for training
    direction_train = label_direction(y_train.values, baseline_train)

    baseline_rmse = rmse(y_test, baseline_test)
    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")

    # ── V5 Ensemble (reference) ───────────────────────────────────────────────
    _, rmse_v5 = train_v5_ensemble(
        X_train, y_train, log_ratio_train,
        X_test, y_test, baseline_test, dfd_test,
    )

    # ── Two-Stage Model ───────────────────────────────────────────────────────
    _, _, rmse_hard, rmse_soft = train_two_stage(
        X_train, y_train, direction_train,
        baseline_train, X_test, y_test,
        baseline_test, dfd_test,
    )

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("FINAL COMPARISON")
    print("=" * 68)
    print(f"  {'Model':<40} {'RMSE':>8}  {'vs Baseline':>12}  {'vs v5':>10}")
    print("-" * 68)
    print(f"  {'Baseline (max(price, emf))':<40} {baseline_rmse:>8.4f}  {'(reference)':>12}")

    for label, r in [
        ("v5 Ensemble (LGB+LGB-logratio+XGB)", rmse_v5),
        ("Two-Stage: Hard Routing",              rmse_hard),
        ("Two-Stage: Soft Routing",              rmse_soft),
    ]:
        vs_bl  = (baseline_rmse - r) / baseline_rmse * 100
        vs_v5  = (rmse_v5 - r) / rmse_v5 * 100
        tag    = " <-- BETTER than v5" if r < rmse_v5 else ""
        print(f"  {label:<40} {r:>8.4f}  {vs_bl:>+11.2f}%  {vs_v5:>+9.2f}%{tag}")

    print()
    print(f"Two-Stage Hard vs v5: {(rmse_v5 - rmse_hard)/rmse_v5*100:+.2f}%  "
          f"({'IMPROVEMENT' if rmse_hard < rmse_v5 else 'REGRESSION'})")
    print(f"Two-Stage Soft vs v5: {(rmse_v5 - rmse_soft)/rmse_v5*100:+.2f}%  "
          f"({'IMPROVEMENT' if rmse_soft < rmse_v5 else 'REGRESSION'})")


if __name__ == '__main__':
    main()
