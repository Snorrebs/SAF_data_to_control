#!/usr/bin/env python3
"""
03_prep_for_arx.py

Build a SISO ARX regression table from filtered plant data, with z-normalization.

Signals (filtered):
- Target: Tot_Resistance_mOhm_filt
- Input (manipulated): El1_dpos_mps_filt
- Disturbances: El1_kA_filt, RMS_V_transformer_filt

ARX structure (all t in seconds, sampling 1 Hz):
    y(t) ≈
        a1 y(t-1) + a2 y(t-2)
        + b1 u(t-1) + b2 u(t-2) + b3 u(t-3)
        + c1 I1(t-1) + c2 I1(t-2)
        + d1 Vsec(t-1)

We:
- Build these lags from filtered signals (strictly causal).
- Drop rows with NaNs (edges).
- Z-normalize target + features.
- Save normalized ARX dataset + meta with μ, σ to undo later.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# --------------------------- CONFIG ---------------------------
IN_CSV   = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_filtered.csv")
OUT_CSV  = Path("meta_arx/arx/arx_prep/model_arx_siso_norm.csv")
META_PATH = Path("meta_arx/arx/arx_prep/model_arx_siso_norm.meta.joblib")

# Target (filtered plant). Change this to electrode-1 resistance when available.
Y_FILT_COL = "Tot_Resistance_mOhm_filt"

# Filtered input/disturbance signals produced by script 2
U_FILT_COL    = "El1_dpos_mps_filt"       # manipulated variable
I1_FILT_COL   = "El1_kA_filt"             # disturbance current
VSEC_FILT_COL = "RMS_V_transformer_filt"  # disturbance voltage

# Lags matching the chosen ARX structure
MAX_AR_LAG  = 5   # y(t-1..2)
MAX_U_LAG   = 3   # u(t-1..3)
MAX_I1_LAG  = 2   # I1(t-1..2)
MAX_VSEC_LAG = 1  # Vsec(t-1)

# Forecast horizon (H = 0 → predict y(t))
H = 0

# Whether to write timestamp index
SAVE_INDEX_TIMESTAMP = True


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    ensure_parent(OUT_CSV)
    ensure_parent(META_PATH)

    # ---------- Load & basic prep ----------
    df = pd.read_csv(IN_CSV, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the input file.")
    df = df.sort_values("timestamp").set_index("timestamp")

    # ---------- Sanity checks for required columns ----------
    for c in [Y_FILT_COL, U_FILT_COL, I1_FILT_COL, VSEC_FILT_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required filtered column '{c}' in {IN_CSV}")

    # ---------- Target & target_time ----------
    # With H=0, target is simply y_filt(t).
    if H == 0:
        target_col = Y_FILT_COL
        df["target_time"] = df.index
    else:
        target_col = "y_target"
        df[target_col] = df[Y_FILT_COL].shift(-H)
        df["target_time"] = df.index + pd.to_timedelta(H, unit="s")

    # ---------- Build lags (strictly causal) ----------
    lag_cols = {}

    # y-lags: y(t-1), y(t-2)
    y_series = df[Y_FILT_COL].astype("float64")
    for L in range(1, MAX_AR_LAG + 1):
        lag_cols[f"y_filt_lag{L}"] = y_series.shift(L)

    # u-lags: u(t-1), u(t-2), u(t-3)
    u_series = df[U_FILT_COL].astype("float64")
    for L in range(1, MAX_U_LAG + 1):
        lag_cols[f"{U_FILT_COL}_lag{L}"] = u_series.shift(L)

    # I1-lags: I1(t-1), I1(t-2)
    i1_series = df[I1_FILT_COL].astype("float64")
    for L in range(1, MAX_I1_LAG + 1):
        lag_cols[f"{I1_FILT_COL}_lag{L}"] = i1_series.shift(L)

    # Vsec-lags: Vsec(t-1)
    vsec_series = df[VSEC_FILT_COL].astype("float64")
    for L in range(1, MAX_VSEC_LAG + 1):
        lag_cols[f"{VSEC_FILT_COL}_lag{L}"] = vsec_series.shift(L)

    lag_df = pd.DataFrame(lag_cols, index=df.index)

    # ---------- Assemble & drop NaNs ----------
    df_all = pd.concat([df[[target_col, "target_time"]], lag_df], axis=1).dropna().copy()

    # ---------- Feature column list ----------
    feat_cols = sorted([c for c in lag_df.columns if c in df_all.columns])
    keep_cols = ["target_time", target_col] + feat_cols
    df_arx = df_all[keep_cols].copy()

    # ---------- Z-normalization ----------
    # We normalize both the target and all feature columns.
    norm_cols = [target_col] + feat_cols

    mu = {}
    sigma = {}
    for c in norm_cols:
        m = df_arx[c].mean()
        s = df_arx[c].std(ddof=0)
        if s == 0 or np.isnan(s):
            # avoid division by zero; if zero-variance, set s=1 (effectively no scaling)
            s = 1.0
        mu[c] = float(m)
        sigma[c] = float(s)
        df_arx[c] = (df_arx[c] - m) / s

    # ---------- Save normalized ARX dataset ----------
    if SAVE_INDEX_TIMESTAMP:
        df_arx = df_arx.rename_axis("timestamp")

    df_arx.to_csv(OUT_CSV, index=SAVE_INDEX_TIMESTAMP)

    # ---------- Save meta (for undoing normalization & simulation) ----------
    meta = {
        "y_col": target_col,
        "X_cols": feat_cols,
        "norm": {
            "mu": mu,
            "sigma": sigma,
            "columns": norm_cols,
        },
        "config": {
            "horizon": H,
            "lags": {
                "y": list(range(1, MAX_AR_LAG + 1)),
                "u": list(range(1, MAX_U_LAG + 1)),
                "I1": list(range(1, MAX_I1_LAG + 1)),
                "Vsec": list(range(1, MAX_VSEC_LAG + 1)),
            },
            "signals": {
                "y_filt": Y_FILT_COL,
                "u_filt": U_FILT_COL,
                "I1_filt": I1_FILT_COL,
                "Vsec_filt": VSEC_FILT_COL,
            },
        },
    }
    ensure_parent(META_PATH)
    joblib.dump(meta, META_PATH)

    print(f"[prep] Wrote ARX dataset -> {OUT_CSV}")
    print(f"[prep] Wrote meta -> {META_PATH}")
    print(f"[prep] Target (normalized): {target_col}")
    print(f"[prep] Features: {len(feat_cols)} | Samples: {len(df_arx)}")


if __name__ == "__main__":
    main()
