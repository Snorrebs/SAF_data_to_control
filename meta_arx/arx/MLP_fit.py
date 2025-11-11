#!/usr/bin/env python3
# torch_run_arx.py
#
# Train a SINGLE-model ARX that predicts the PLANT (filtered) directly.
# - Uses only z-scored features (columns ending with "_z", except the target_z).
# - Target name is read from prep metadata, so it works for both H=0 and H>0.
# - Time-based split with an optional GAP to avoid leakage from overlapping windows.
# - Saves model and a small report CSV of predictions vs truth.
#
# Requirements:
#   pip install torch pandas numpy joblib scikit-learn

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load

# --------------------------- CONFIG ---------------------------
IN_CSV        = Path("arx/arx_prep/model_arx_30_5_5.csv")
SCALERS_PATH  = Path("arx/arx_prep/model_arx_scalers_30_5_5.joblib")
MODEL_OUT     = Path("arx/models/arx_one_model.pt")
PRED_CSV_OUT  = Path("arx/models/arx_one_model_predictions.csv")

# NOTE: physical target name is not hardcoded; we read it from metadata.
EPOCHS        = 60
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
TRAIN_FRAC    = 0.80
BATCH_SIZE    = 4096     # tune as needed; set None to use full-batch
HIDDEN        = 64       # width of hidden layers
LAYERS        = 2        # number of hidden layers
SEED          = 7

# --------------------------------------------------------------


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2):
        super().__init__()
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        blocks += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def make_batches(X, y, batch_size):
    if (batch_size is None) or (batch_size <= 0) or (batch_size >= len(X)):
        yield X, y
    else:
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---------- Load data & scalers ----------
    assert IN_CSV.exists(), f"Missing dataset: {IN_CSV}"
    assert SCALERS_PATH.exists(), f"Missing scalers: {SCALERS_PATH}"

    df = pd.read_csv(IN_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    meta = load(SCALERS_PATH)

    # Target names from prep metadata (handles H=0 or H>0)
    y_col   = meta["y_col"]             # e.g., "Tot_Resistance_mOhm_filt" or "y_target"
    y_col_z = y_col + "_z"
    y_scaler = meta["y_scaler"]

    # Use ONLY z-scored features, excluding the target_z
    X_cols_z = [c for c in df.columns if c.endswith("_z") and c != y_col_z]

    # ---- FIX: Drop NaNs only on existing z-features and target_z (no physical target here) ----
    df = df.dropna(subset=X_cols_z + [y_col_z]).copy()

    # ---------- Time-based split with optional GAP to avoid leakage ----------
    cfg = meta.get("config", {})
    H = int(cfg.get("horizon", 0))
    gap = int(H + max(cfg.get("max_ar_lag", 0), cfg.get("max_x_lag", 0)))

    n = len(df)
    split = int(TRAIN_FRAC * n)
    te_start = min(n, split + gap)  # push test forward by the gap
    df_tr = df.iloc[:split].copy()
    df_te = df.iloc[te_start:].copy()
    if len(df_te) == 0:
        raise RuntimeError(
            f"Empty test set after applying gap={gap}. "
            f"Consider reducing TRAIN_FRAC or gap, or use a longer dataset."
        )

    # Tensors
    Xtr = torch.tensor(df_tr[X_cols_z].values, dtype=torch.float32)
    ytr = torch.tensor(df_tr[[y_col_z]].values, dtype=torch.float32)
    Xte = torch.tensor(df_te[X_cols_z].values, dtype=torch.float32)
    yte = torch.tensor(df_te[[y_col_z]].values, dtype=torch.float32)

    # ---------- Model ----------
    model = MLP(in_dim=Xtr.shape[1], hidden=HIDDEN, layers=LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # ---------- Train ----------
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in make_batches(Xtr, ytr, BATCH_SIZE):
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(xb)

        epoch_loss /= len(Xtr)
        if epoch % max(1, EPOCHS // 10) == 0:
            print(f"[epoch {epoch:03d}] train MSE(z) = {epoch_loss:.6f}")

    # ---------- Evaluate (inverse-transform to mΩ) ----------
    model.eval()
    with torch.no_grad():
        yhat_tr_z = model(Xtr).cpu().numpy().reshape(-1, 1)
        yhat_te_z = model(Xte).cpu().numpy().reshape(-1, 1)

    # Back to physical units (mΩ)
    yhat_tr = y_scaler.inverse_transform(yhat_tr_z)[:, 0]
    yhat_te = y_scaler.inverse_transform(yhat_te_z)[:, 0]
    ytrue_tr = y_scaler.inverse_transform(ytr.cpu().numpy())[:, 0]
    ytrue_te = y_scaler.inverse_transform(yte.cpu().numpy())[:, 0]

    def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))

    tr_rmse = rmse(ytrue_tr, yhat_tr)
    te_rmse = rmse(ytrue_te, yhat_te)
    tr_r2   = r2_score(ytrue_tr, yhat_tr)
    te_r2   = r2_score(ytrue_te, yhat_te)

    print(f"[Train] RMSE={tr_rmse:.6f} mΩ, R²={tr_r2:.3f}")
    print(f"[ Test] RMSE={te_rmse:.6f} mΩ, R²={te_r2:.3f}")
    print(f"[Info ] Samples: train={len(df_tr)}, gap={gap}, test={len(df_te)}")

    # ---------- Save model ----------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "X_cols_z": X_cols_z,
        "y_col": y_col,          # <-- from meta (works for H=0 and H>0)
        "y_col_z": y_col_z,
        "config": {
            "hidden": HIDDEN,
            "layers": LAYERS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "train_frac": TRAIN_FRAC,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "gap": gap,
        }
    }, MODEL_OUT)
    print(f"[save] Model -> {MODEL_OUT}")

    # ---------- Save prediction CSV (for quick plotting/audit) ----------
    out_df = pd.DataFrame({
        "timestamp": df_te.index,
        "y_true_mOhm": ytrue_te,
        "y_pred_mOhm": yhat_te,
    }).set_index("timestamp")
    out_df.to_csv(PRED_CSV_OUT)
    print(f"[save] Predictions -> {PRED_CSV_OUT}")


if __name__ == "__main__":
    main()
