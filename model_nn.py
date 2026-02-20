"""
Neural network models for flight price prediction.

Models:
  MLP  - Feedforward network on all tabular features
  CNN  - 1D-CNN over price history (dfd=40..49) + tabular features
  GRU  - Gated Recurrent Unit over price history + tabular features

Compares against the baseline and best GBM ensemble from model_v2.py.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

OBSERVATION_DFD = 40
FLIGHT_KEY = ['flt_id', 'pathod_id', 'airline', 'depdt']
DEVICE = torch.device('cpu')
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Feature engineering  (same as model_v2.py)
# ---------------------------------------------------------------------------

def build_features(df, observation_dfd=OBSERVATION_DFD):
    static = df[FLIGHT_KEY + ['dephour', 'depdow']].drop_duplicates(subset=FLIGHT_KEY)

    known = df[df['dfd'] >= observation_dfd].copy()
    known['price_ratio'] = known['target_price'] / (known['expected_minfare'] + 1e-6)

    at_obs = known[known['dfd'] == observation_dfd][
        FLIGHT_KEY + ['target_price', 'expected_minfare']
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

    # Prices at dfd=40..49
    lookback = at_obs[FLIGHT_KEY].copy()
    for dfd_val in range(observation_dfd, min(observation_dfd + 10, 50)):
        sub = df[df['dfd'] == dfd_val][FLIGHT_KEY + ['target_price']].rename(
            columns={'target_price': f'price_dfd{dfd_val}'}
        )
        lookback = lookback.merge(sub, on=FLIGHT_KEY, how='left')

    def slope(row):
        vals = [row.get(f'price_dfd{d}', np.nan)
                for d in range(observation_dfd, min(observation_dfd + 10, 50))]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2:
            return 0.0
        return float(np.polyfit(range(len(vals)), vals, 1)[0])

    lookback['price_trend'] = lookback.apply(slope, axis=1)
    lookback['price_recent_vs_old'] = (
        lookback.get('price_dfd40', np.nan) - lookback.get('price_dfd49', np.nan)
    )

    market_at_obs = df[df['dfd'] == observation_dfd].copy()
    market_stats = market_at_obs.groupby(['pathod_id', 'depdt']).agg(
        market_min_price=('target_price', 'min'),
        market_max_price=('target_price', 'max'),
        market_mean_price=('target_price', 'mean'),
        market_n_airlines=('airline', 'nunique'),
    ).reset_index()

    ff = static.merge(at_obs, on=FLIGHT_KEY, how='left')
    ff = ff.merge(hist_stats, on=FLIGHT_KEY, how='left')
    ff = ff.merge(lookback, on=FLIGHT_KEY, how='left')
    ff = ff.merge(market_stats, on=['pathod_id', 'depdt'], how='left')

    future = df[df['dfd'] < observation_dfd][
        FLIGHT_KEY + ['dfd', 'expected_minfare', 'target_price']
    ].copy()

    data = future.merge(ff, on=FLIGHT_KEY, how='left')
    data['dfd_diff'] = observation_dfd - data['dfd']
    data['dfd_frac'] = data['dfd'] / observation_dfd
    data['emf_ratio_to_obs'] = data['expected_minfare'] / (data['emf_at_obs'] + 1e-6)
    data['price_to_emf'] = data['price_at_obs'] / (data['expected_minfare'] + 1e-6)
    data['price_to_market_min'] = data['price_at_obs'] / (data['market_min_price'] + 1e-6)
    data['price_above_emf'] = data['price_at_obs'] - data['emf_at_obs']
    data['airline_enc'] = data['airline'].astype('category').cat.codes
    data['baseline_pred'] = np.maximum(data['price_at_obs'], data['expected_minfare'])
    data['residual'] = data['target_price'] - data['baseline_pred']
    return data


TABULAR_FEATURES = [
    'dfd', 'dfd_diff', 'dfd_frac', 'dephour', 'depdow', 'airline_enc', 'pathod_id',
    'price_at_obs', 'emf_at_obs', 'expected_minfare',
    'emf_ratio_to_obs', 'price_to_emf', 'price_to_market_min', 'price_above_emf',
    'hist_price_mean', 'hist_price_std', 'hist_price_max', 'hist_price_min', 'hist_price_range',
    'hist_ratio_mean', 'hist_ratio_std', 'hist_ratio_max', 'hist_ratio_min',
    'price_trend', 'price_recent_vs_old',
    'market_min_price', 'market_max_price', 'market_mean_price', 'market_n_airlines',
    'price_dfd40', 'price_dfd41', 'price_dfd42', 'price_dfd43', 'price_dfd44',
    'price_dfd45', 'price_dfd46', 'price_dfd47', 'price_dfd48', 'price_dfd49',
    'baseline_pred',
]

# Price history columns (10 steps: dfd=40..49) used as the sequence input
HISTORY_COLS = [f'price_dfd{d}' for d in range(40, 50)]

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))

def print_metrics(label, y_true, y_pred, baseline=None):
    r = rmse(y_true, y_pred)
    m = mape(y_true, y_pred)
    if baseline is not None:
        bl_r = rmse(y_true, baseline)
        bl_m = mape(y_true, baseline)
        print(f"  {label}: RMSE={r:.4f} ({(bl_r-r)/bl_r*100:+.2f}%), "
              f"MAPE={m:.4f} ({(bl_m-m)/bl_m*100:+.2f}%)")
    else:
        print(f"  {label}: RMSE={r:.4f}, MAPE={m:.4f}")
    return r, m

# ---------------------------------------------------------------------------
# PyTorch model architectures
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard feedforward network on tabular features."""
    def __init__(self, in_dim, hidden=(512, 256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tab, seq=None):
        return self.net(tab).squeeze(-1)


class CNN1D(nn.Module):
    """
    1D-CNN over the price-history sequence, combined with tabular features.

    Sequence shape: (batch, channels=2, time=10)
      channel 0 = normalised price history (price_dfd / price_at_obs)
      channel 1 = linear dfd encoding [40,41,...,49] / 50
    Tabular branch: same tabular features as MLP.
    """
    def __init__(self, tab_dim, seq_len=10, seq_channels=2, dropout=0.2):
        super().__init__()
        # Convolutional branch
        self.conv = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (batch, 64, 1)
        )
        conv_out = 64

        # Tabular branch
        self.tab_net = nn.Sequential(
            nn.Linear(tab_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        tab_out = 128

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(conv_out + tab_out, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tab, seq):
        c = self.conv(seq).squeeze(-1)   # (batch, 64)
        t = self.tab_net(tab)            # (batch, 128)
        return self.head(torch.cat([c, t], dim=1)).squeeze(-1)


class GRUModel(nn.Module):
    """
    GRU over the price-history sequence, combined with tabular features.

    Sequence shape: (batch, time=10, features=2)
      feature 0 = normalised price (price_dfd / price_at_obs)
      feature 1 = emf_at_obs / price_at_obs (constant across time, provides scale)
    """
    def __init__(self, tab_dim, seq_input=2, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(seq_input, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        gru_out = hidden

        self.tab_net = nn.Sequential(
            nn.Linear(tab_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        tab_out = 128

        self.head = nn.Sequential(
            nn.Linear(gru_out + tab_out, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tab, seq):
        # seq: (batch, time, features) – GRU uses batch_first=True
        _, h = self.gru(seq)         # h: (num_layers, batch, hidden)
        h_last = h[-1]               # (batch, hidden)
        t = self.tab_net(tab)
        return self.head(torch.cat([h_last, t], dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def make_sequence_tensor(data, scaler_price):
    """
    Build normalised sequence tensor (batch, 2, 10) for CNN
    and (batch, 10, 2) for GRU from raw data.
    """
    prices = data[HISTORY_COLS].fillna(0).values.astype(np.float32)
    price_at_obs = data['price_at_obs'].fillna(1).values.astype(np.float32)[:, None]

    # Normalise: each flight's price history divided by its price_at_obs
    norm_prices = prices / (price_at_obs + 1e-6)

    # Second channel: relative dfd position [0..1] – same for all rows
    dfd_pos = np.linspace(0, 1, 10, dtype=np.float32)
    dfd_ch = np.tile(dfd_pos, (len(data), 1))  # (batch, 10)

    # CNN expects (batch, channels, time)
    seq_cnn = np.stack([norm_prices, dfd_ch], axis=1)  # (batch, 2, 10)
    # GRU expects (batch, time, features)
    seq_gru = np.stack([norm_prices, dfd_ch], axis=2)  # (batch, 10, 2)
    return seq_cnn, seq_gru


def train_model(model, X_tr, seq_tr_cnn, seq_tr_gru, y_tr,
                X_va, seq_va_cnn, seq_va_gru, y_va,
                model_type='mlp',
                epochs=100, batch_size=4096, lr=1e-3, patience=10):
    """Train a PyTorch model with early stopping on validation RMSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    # Build tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    if model_type == 'cnn':
        seq_tr = torch.tensor(seq_tr_cnn, dtype=torch.float32)
        seq_va = torch.tensor(seq_va_cnn, dtype=torch.float32)
    elif model_type == 'gru':
        seq_tr = torch.tensor(seq_tr_gru, dtype=torch.float32)
        seq_va = torch.tensor(seq_va_gru, dtype=torch.float32)
    else:
        seq_tr = seq_va = None

    if seq_tr is not None:
        ds = TensorDataset(X_tr_t, seq_tr, y_tr_t)
    else:
        ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            opt.zero_grad()
            if seq_tr is not None:
                xb, sb, yb = batch
                pred = model(xb, sb)
            else:
                xb, yb = batch
                pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if seq_va is not None:
                val_pred = model(X_va_t, seq_va).numpy()
            else:
                val_pred = model(X_va_t).numpy()
        val_rmse = rmse(y_va, val_pred)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}: val RMSE={val_rmse:.4f} (best={best_val_rmse:.4f})")

    model.load_state_dict(best_state)
    return model, best_val_rmse


def predict_model(model, X, seq_cnn, seq_gru, model_type='mlp', batch_size=8192):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    if model_type == 'cnn':
        seq_t = torch.tensor(seq_cnn, dtype=torch.float32)
    elif model_type == 'gru':
        seq_t = torch.tensor(seq_gru, dtype=torch.float32)
    else:
        seq_t = None

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size]
            if seq_t is not None:
                sb = seq_t[i:i+batch_size]
                p = model(xb, sb)
            else:
                p = model(xb)
            preds.append(p.numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')

    print("Building features...")
    train_data = build_features(train)
    test_data  = build_features(test)
    train_data = train_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    test_data  = test_data.dropna(subset=['price_at_obs']).reset_index(drop=True)
    print(f"Train: {len(train_data):,}  Test: {len(test_data):,}")

    y_train = train_data['target_price'].values.astype(np.float32)
    y_test  = test_data['target_price'].values.astype(np.float32)
    baseline_test = test_data['baseline_pred'].values.astype(np.float32)
    dfd_test = test_data['dfd'].values

    # ------------------------------------------------------------------
    # Tabular feature matrix (scaled)
    # ------------------------------------------------------------------
    X_raw_tr = train_data[TABULAR_FEATURES].fillna(0).values.astype(np.float32)
    X_raw_te = test_data[TABULAR_FEATURES].fillna(0).values.astype(np.float32)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_raw_tr)
    X_te_sc = scaler.transform(X_raw_te)

    # ------------------------------------------------------------------
    # Sequence tensors
    # ------------------------------------------------------------------
    seq_tr_cnn, seq_tr_gru = make_sequence_tensor(train_data, scaler)
    seq_te_cnn, seq_te_gru = make_sequence_tensor(test_data, scaler)

    # ------------------------------------------------------------------
    # Validation split (last 15% of train rows)
    # ------------------------------------------------------------------
    n_val = int(len(train_data) * 0.15)
    idx = np.random.default_rng(42).permutation(len(train_data))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr, X_va = X_tr_sc[tr_idx], X_tr_sc[val_idx]
    y_tr, y_va = y_train[tr_idx], y_train[val_idx]
    seq_tr_cnn_s, seq_va_cnn_s = seq_tr_cnn[tr_idx], seq_tr_cnn[val_idx]
    seq_tr_gru_s, seq_va_gru_s = seq_tr_gru[tr_idx], seq_tr_gru[val_idx]

    tab_dim = X_tr_sc.shape[1]

    bl_r = rmse(y_test, baseline_test)
    bl_m = mape(y_test, baseline_test)
    print(f"\nBaseline → RMSE={bl_r:.4f}, MAPE={bl_m:.4f}")

    results = {}

    # ==================================================================
    # GBM reference (best from model_v2)
    # ==================================================================
    print("\n=== GBM Reference (LightGBM ensemble) ===")
    lgb_a = lgb.LGBMRegressor(n_estimators=1000, max_depth=7, learning_rate=0.02,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                               num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
    lgb_a.fit(X_raw_tr, y_train)
    gbm_pred_a = lgb_a.predict(X_raw_te)

    lgb_b = lgb.LGBMRegressor(n_estimators=1000, max_depth=7, learning_rate=0.02,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                               num_leaves=63, random_state=99, n_jobs=-1, verbose=-1)
    lgb_b.fit(X_raw_tr, train_data['residual'].values)
    gbm_pred_b = baseline_test + lgb_b.predict(X_raw_te)

    xgb_c = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              random_state=42, n_jobs=-1, verbosity=0)
    xgb_c.fit(X_raw_tr, y_train)
    gbm_pred_c = xgb_c.predict(X_raw_te)

    gbm_ens = (gbm_pred_a + gbm_pred_b + gbm_pred_c) / 3
    r_gbm, m_gbm = print_metrics("GBM ensemble (A+B+C)", y_test, gbm_ens, baseline_test)
    results['GBM ensemble'] = (r_gbm, m_gbm)

    # ==================================================================
    # Model 1: MLP
    # ==================================================================
    print("\n=== Model 1: MLP ===")
    mlp = MLP(tab_dim, hidden=(512, 256, 128, 64), dropout=0.2).to(DEVICE)
    mlp, _ = train_model(mlp, X_tr, None, None, y_tr,
                         X_va, None, None, y_va,
                         model_type='mlp', epochs=150, batch_size=4096,
                         lr=1e-3, patience=15)
    mlp_pred = predict_model(mlp, X_te_sc, None, None, 'mlp')
    r_mlp, m_mlp = print_metrics("MLP", y_test, mlp_pred, baseline_test)
    results['MLP'] = (r_mlp, m_mlp)

    # ==================================================================
    # Model 2: 1D-CNN
    # ==================================================================
    print("\n=== Model 2: 1D-CNN ===")
    cnn = CNN1D(tab_dim, seq_len=10, seq_channels=2, dropout=0.2).to(DEVICE)
    cnn, _ = train_model(cnn, X_tr, seq_tr_cnn_s, seq_tr_gru_s, y_tr,
                         X_va, seq_va_cnn_s, seq_va_gru_s, y_va,
                         model_type='cnn', epochs=150, batch_size=4096,
                         lr=1e-3, patience=15)
    cnn_pred = predict_model(cnn, X_te_sc, seq_te_cnn, seq_te_gru, 'cnn')
    r_cnn, m_cnn = print_metrics("CNN", y_test, cnn_pred, baseline_test)
    results['CNN'] = (r_cnn, m_cnn)

    # ==================================================================
    # Model 3: GRU
    # ==================================================================
    print("\n=== Model 3: GRU ===")
    gru = GRUModel(tab_dim, seq_input=2, hidden=64, num_layers=2, dropout=0.2).to(DEVICE)
    gru, _ = train_model(gru, X_tr, seq_tr_cnn_s, seq_tr_gru_s, y_tr,
                         X_va, seq_va_cnn_s, seq_va_gru_s, y_va,
                         model_type='gru', epochs=150, batch_size=4096,
                         lr=1e-3, patience=15)
    gru_pred = predict_model(gru, X_te_sc, seq_te_cnn, seq_te_gru, 'gru')
    r_gru, m_gru = print_metrics("GRU", y_test, gru_pred, baseline_test)
    results['GRU'] = (r_gru, m_gru)

    # ==================================================================
    # Model 4: Wider / deeper MLP
    # ==================================================================
    print("\n=== Model 4: Wide MLP ===")
    mlp2 = MLP(tab_dim, hidden=(1024, 512, 256, 128, 64), dropout=0.3).to(DEVICE)
    mlp2, _ = train_model(mlp2, X_tr, None, None, y_tr,
                          X_va, None, None, y_va,
                          model_type='mlp', epochs=150, batch_size=4096,
                          lr=5e-4, patience=15)
    mlp2_pred = predict_model(mlp2, X_te_sc, None, None, 'mlp')
    r_mlp2, m_mlp2 = print_metrics("Wide MLP", y_test, mlp2_pred, baseline_test)
    results['Wide MLP'] = (r_mlp2, m_mlp2)

    # ==================================================================
    # Ensembles: NN alone, NN + GBM
    # ==================================================================
    print("\n=== Ensembles ===")

    nn_ens = (mlp_pred + cnn_pred + gru_pred + mlp2_pred) / 4
    r_nn_ens, m_nn_ens = print_metrics("NN ensemble (all NNs)", y_test, nn_ens, baseline_test)
    results['NN ensemble'] = (r_nn_ens, m_nn_ens)

    nn_gbm = (mlp_pred + cnn_pred + gru_pred + mlp2_pred + gbm_ens * 2) / 6
    r_nn_gbm, m_nn_gbm = print_metrics("NN+GBM ensemble", y_test, nn_gbm, baseline_test)
    results['NN+GBM ensemble'] = (r_nn_gbm, m_nn_gbm)

    # Per-dfd breakdown for best model
    best_name = min(results, key=lambda k: results[k][0])
    preds_map = {
        'GBM ensemble': gbm_ens, 'MLP': mlp_pred, 'CNN': cnn_pred,
        'GRU': gru_pred, 'Wide MLP': mlp2_pred,
        'NN ensemble': nn_ens, 'NN+GBM ensemble': nn_gbm,
    }
    best_pred = preds_map[best_name]
    print(f"\n=== Per-dfd breakdown: {best_name} ===")
    for dfd_val in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        mask = dfd_test == dfd_val
        if mask.sum() > 0:
            r = rmse(y_test[mask], best_pred[mask])
            m = mape(y_test[mask], best_pred[mask])
            bl_r_d = rmse(y_test[mask], baseline_test[mask])
            bl_m_d = mape(y_test[mask], baseline_test[mask])
            print(f"  dfd={dfd_val:2d}: RMSE={r:.4f}({(bl_r_d-r)/bl_r_d*100:+.1f}%) "
                  f"MAPE={m:.4f}({(bl_m_d-m)/bl_m_d*100:+.1f}%)")

    # ==================================================================
    # Final summary
    # ==================================================================
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Model':<25} {'RMSE':>8} {'MAPE':>8} {'RMSE vs BL':>11} {'MAPE vs BL':>11}")
    print("-" * 68)
    print(f"{'Baseline':<25} {bl_r:>8.4f} {bl_m:>8.4f}")
    for name, (r, m) in results.items():
        rp = (bl_r - r) / bl_r * 100
        mp = (bl_m - m) / bl_m * 100
        marker = " <-- best" if name == best_name else ""
        print(f"  {name:<23} {r:>8.4f} {m:>8.4f} {rp:>+10.2f}% {mp:>+10.2f}%{marker}")


if __name__ == '__main__':
    main()
