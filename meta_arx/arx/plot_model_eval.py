#!/usr/bin/env python3
# plot_model_eval.py
#
# Usage:
#   python plot_model_eval.py models/arx_linear_ridge_predictions.csv "ARX (Ridge)"
#   python plot_model_eval.py models/arx_one_model_predictions.csv    "MLP"
#
# Creates PNGs next to the CSV.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def acf(x, nlags=200):
    x = np.asarray(x) - np.mean(x)
    denom = np.dot(x, x)
    ac = [1.0]
    for k in range(1, nlags+1):
        num = np.dot(x[:-k], x[k:])
        ac.append(num/denom if denom > 0 else np.nan)
    return np.array(ac)

def rolling_rmse(y_true, y_pred, window):
    e2 = (np.asarray(y_true) - np.asarray(y_pred))**2
    # fast rolling mean via convolution
    w = np.ones(window, dtype=float)/window
    # pad with NaNs to keep alignment simple
    rmse_roll = np.full_like(e2, np.nan, dtype=float)
    if len(e2) >= window:
        m = np.convolve(e2, w, mode="valid")
        rmse_v = np.sqrt(m)
        rmse_roll[window-1:] = rmse_v
    return rmse_roll

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_model_eval.py <pred_csv> <model_name>")
        sys.exit(1)

    pred_csv = Path(sys.argv[1])
    model_name = sys.argv[2]

    assert pred_csv.exists(), f"Missing file: {pred_csv}"
    df = pd.read_csv(pred_csv, parse_dates=["timestamp"]).set_index("timestamp").sort_index()

    y_true = df["y_true_mOhm"].values
    y_pred = df["y_pred_mOhm"].values
    err = y_pred - y_true

    # metrics
    _rmse = rmse(y_true, y_pred)
    _r2   = r2_score(y_true, y_pred)

    print(f"[{model_name}] TEST RMSE={_rmse:.6f} mΩ, R²={_r2:.3f}, N={len(df)}")

    out_dir = pred_csv.parent
    stem = pred_csv.stem

    # 1) Time series: y_true vs y_pred
    plt.figure(figsize=(11,4))
    plt.plot(df.index, y_true, label="True (mΩ)")
    plt.plot(df.index, y_pred, label="Pred (mΩ)", alpha=0.8)
    plt.title(f"{model_name} — True vs Pred (RMSE={_rmse:.4f} mΩ, R²={_r2:.3f})")
    plt.xlabel("Time"); plt.ylabel("Resistance [mΩ]"); plt.legend()
    p1 = out_dir / f"{stem}_timeseries.png"
    plt.tight_layout(); plt.savefig(p1, dpi=150); plt.show() ; plt.close()
    print(f"[save] {p1}")

    # 2) Residuals over time
    plt.figure(figsize=(11,3.5))
    plt.plot(df.index, err)
    plt.axhline(0, linestyle="--")
    plt.title(f"{model_name} — Residuals over time (pred - true)")
    plt.xlabel("Time"); plt.ylabel("Residual [mΩ]")
    p2 = out_dir / f"{stem}_residuals.png"
    plt.tight_layout(); plt.savefig(p2, dpi=150); plt.close()
    print(f"[save] {p2}")

    # 3) Scatter: y_true vs y_pred
    plt.figure(figsize=(4.2,4.2))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    lo, hi = np.percentile(np.concatenate([y_true, y_pred]), [1, 99])
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(f"{model_name} — True vs Pred")
    plt.xlabel("True [mΩ]"); plt.ylabel("Pred [mΩ]")
    p3 = out_dir / f"{stem}_scatter.png"
    plt.tight_layout(); plt.savefig(p3, dpi=150); plt.close()
    print(f"[save] {p3}")

    # 4) Residual histogram
    plt.figure(figsize=(5,3.6))
    plt.hist(err, bins=60)
    mu, sd = np.mean(err), np.std(err)
    plt.title(f"{model_name} — Residual histogram (μ={mu:.4g}, σ={sd:.4g} mΩ)")
    plt.xlabel("Residual [mΩ]"); plt.ylabel("Count")
    p4 = out_dir / f"{stem}_residual_hist.png"
    plt.tight_layout(); plt.savefig(p4, dpi=150); plt.close()
    print(f"[save] {p4}")

    # 5) Residual ACF (whiteness)
    nlags = min(200, max(5, len(err)//20))
    a = acf(err, nlags=nlags)
    plt.figure(figsize=(8,3.2))
    plt.stem(range(len(a)), a)
    plt.title(f"{model_name} — Residual ACF (nlags={nlags})")
    plt.xlabel("Lag"); plt.ylabel("ACF")
    # 95% CI for white noise approx ±1.96/sqrt(N)
    ci = 1.96/np.sqrt(len(err))
    plt.axhspan(-ci, ci, color="gray", alpha=0.2)
    p5 = out_dir / f"{stem}_residual_acf.png"
    plt.tight_layout(); plt.savefig(p5, dpi=150); plt.close()
    print(f"[save] {p5}")

    # 6) Rolling RMSE (assume 1 s sampling; change window to your context)
    window = 300  # 5 minutes
    rrmse = rolling_rmse(y_true, y_pred, window=window)
    plt.figure(figsize=(11,3.5))
    plt.plot(df.index, rrmse)
    plt.title(f"{model_name} — Rolling RMSE (window={window} samples)")
    plt.xlabel("Time"); plt.ylabel("RMSE [mΩ]")
    p6 = out_dir / f"{stem}_rolling_rmse.png"
    plt.tight_layout(); plt.savefig(p6, dpi=150); plt.close()
    print(f"[save] {p6}")

if __name__ == "__main__":
    main()
