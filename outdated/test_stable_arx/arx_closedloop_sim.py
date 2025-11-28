#!/usr/bin/env python3
# arx_closedloop_sim.py
#
# Closed-loop simulation of the stable ARX model on the test segment.

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from stable_arx_runner import StableARXRunnerYOnly  # import from your module

META_PATH = Path("models/model_arx_scalers_1_5_5.meta.joblib")
IN_CSV    = Path("init_data/model_arx_1_5_5.csv")
MODEL_IN  = Path("models/arx_linear_ridge_stable_yonly.joblib")
OUT_CSV   = Path("models/arx_linear_ridge_stable_ar_closedloop_sim.csv")

TRAIN_FRAC   = 0.80
MIN_SAFE_GAP = 60

def main():
    assert IN_CSV.exists(), f"Missing dataset: {IN_CSV}"
    assert META_PATH.exists(), f"Missing meta: {META_PATH}"
    assert MODEL_IN.exists(), f"Missing model bundle: {MODEL_IN}"

    df   = pd.read_csv(IN_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    meta = load(META_PATH)

    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    if df["target_time"].dt.tz is None:
        df["target_time"] = df["target_time"].dt.tz_localize("UTC")
    else:
        df["target_time"] = df["target_time"].dt.tz_convert("UTC")

    df = df.dropna(subset=["target_time"])

    y_col      = meta["y_col"]
    X_cols_all = meta["X_cols"]
    cfg        = meta.get("config", {})
    H          = int(cfg.get("horizon", 0))
    max_ar     = int(cfg.get("max_ar_lag", 0))
    max_x      = int(cfg.get("max_x_lag", 0))

    need_cols = ["target_time", y_col] + X_cols_all
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in IN_CSV: {missing}")

    df = df.dropna(subset=need_cols).copy()

    # time split (same as training)
    n = len(df)
    split_idx = int(TRAIN_FRAC * n)

    gap_feat = max(max_ar, max_x)
    gap = max(H + gap_feat, MIN_SAFE_GAP)

    train_end_idx = df.index[split_idx - 1]
    te_start_idx = split_idx + gap
    if te_start_idx >= n:
        raise RuntimeError(f"Empty test set: split={split_idx}, gap={gap}, n={n}.")

    df_tr = df.iloc[:split_idx].copy()
    df_te = df.iloc[te_start_idx:].copy()
    if len(df_te) == 0:
        raise RuntimeError(f"Empty test set after applying gap={gap}.")

    df_tr = df_tr[df_tr["target_time"] <= train_end_idx]
    if len(df_tr) == 0:
        raise RuntimeError("All training rows trimmed by target_time; adjust H/gap.")

    # load model bundle
    bundle = load(MODEL_IN)
    ar_coeffs = np.asarray(bundle["ar_coeffs"], float).ravel()
    p = len(ar_coeffs)
    exog_cols = bundle["exog_cols"]

    # warmup y_init = last p true y from train
    y_init = df_tr[y_col].tail(max(p, 1)).to_numpy()
    runner = StableARXRunnerYOnly(bundle, y_init=y_init)

    # now simulate over test exogenous features
    y_true = df_te[y_col].to_numpy()
    X_exog_te = df_te[exog_cols].to_numpy()

    y_pred_cl = []
    for t in range(len(df_te)):
        x_t = X_exog_te[t, :]
        y_hat = runner.advance(x_t)
        y_pred_cl.append(y_hat)

    y_pred_cl = np.asarray(y_pred_cl)

    # save results
    out_df = pd.DataFrame({
        "timestamp": df_te.index,
        "y_true_mOhm": y_true,
        "y_pred_cl_mOhm": y_pred_cl,
    }).set_index("timestamp")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV)
    print(f"[save] Closed-loop simulation -> {OUT_CSV}")

    # quick metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred_cl)
    r2  = r2_score(y_true, y_pred_cl)
    print(f"[closed-loop] MSE={mse:.6g}, R²={r2:.3f}")


if __name__ == "__main__":
    main()
