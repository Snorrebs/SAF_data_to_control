#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

# ---------- CONFIG ----------
IN_CSV  = Path("meta_arx/data/1s_data_from_plant/07_24.csv")
OUT_CSV = Path("meta_arx/data/filt_data/07_24_filt_varx_current.csv")

TS_COL = "timestamp"

# VARX-current targets (also used as AR lags later)
Y_COLS = ["El1_kA", "El2_kA", "El3_kA"]

# Inputs
POS_COLS = ["El1_pos_m", "El2_pos_m", "El3_pos_m"]
V_COLS   = ["UL1N_V", "UL2N_V", "UL3N_V"]

# Optional impedance proxies (recommended but you can omit later in X_cols)
R_COLS = ["El1_Resistance_mOhm", "El2_Resistance_mOhm", "El3_Resistance_mOhm"]
TOT_R_COL = "Tot_Resistance_mOhm"

# Butterworth params (1 Hz sampling)
FS = 1.0
FC_HZ = 0.1
ORDER_Y    = 2   # currents
ORDER_EXOG = 4   # positions/voltages/resistances

# Whether to automatically convert electrode positions from cm → m if needed
FORCE_ELECTRODE_CM_TO_M = True


def lp_butter(series: pd.Series, fs: float, fc: float, order: int) -> pd.Series:
    """Zero-phase Butterworth low-pass filter for 1D series."""
    series = series.astype(float)
    nyq = fs / 2.0
    wn = fc / nyq
    b, a = butter(order, wn, btype="low", analog=False)

    # Short gaps → interpolate to avoid edge artifacts
    s = series.interpolate(limit=5).ffill().bfill().values
    y = filtfilt(b, a, s, method="gust")
    return pd.Series(y, index=series.index)


def maybe_cm_to_m(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Missing electrode position column '{col}'")
    if not FORCE_ELECTRODE_CM_TO_M:
        return

    mean_pos = df[col].dropna().astype(float).mean()
    # Heuristic: if mean position is >10, it’s likely cm rather than m
    if pd.notna(mean_pos) and mean_pos > 10.0:
        df[col] = df[col].astype(float) / 100.0
        print(f"[unit] Converted {col} from cm to m (mean={mean_pos:.3f})")


def main() -> None:
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    df = pd.read_csv(IN_CSV)

    if TS_COL not in df.columns:
        raise ValueError(f"Expected a '{TS_COL}' column in {IN_CSV}")

    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True, errors="coerce")
    df = df.set_index(TS_COL).sort_index().asfreq("1s")

    # ----- Position unit sanity (cm -> m) -----
    for c in POS_COLS:
        maybe_cm_to_m(df, c)

    # ----- Optional RMS transformer voltage (handy, but you still keep ULxN too) -----
    if "RMS_V_transformer" not in df.columns:
        missing_v = [c for c in V_COLS if c not in df.columns]
        if missing_v:
            raise ValueError(f"Missing voltage columns needed for RMS: {missing_v}")
        df["RMS_V_transformer"] = df[V_COLS].astype(float).pow(2).mean(axis=1).pow(0.5)

    # ----- Filter targets: currents -----
    for c in Y_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing target current column '{c}'")
        df[f"{c}_filt"] = lp_butter(df[c], FS, FC_HZ, ORDER_Y)

    # ----- Filter exogenous: positions -----
    for c in POS_COLS:
        df[f"{c}_filt"] = lp_butter(df[c], FS, FC_HZ, ORDER_EXOG)

    # ----- Filter exogenous: voltages -----
    for c in V_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing voltage column '{c}'")
        df[f"{c}_filt"] = lp_butter(df[c], FS, FC_HZ, ORDER_EXOG)

    # Optional: RMS filtered too (often nice for diagnostics)
    df["RMS_V_transformer_filt"] = lp_butter(df["RMS_V_transformer"], FS, FC_HZ, ORDER_EXOG)

    # ----- Optional: resistances (if present) -----
    for c in R_COLS:
        if c in df.columns:
            df[f"{c}_filt"] = lp_butter(df[c], FS, FC_HZ, ORDER_EXOG)

    if TOT_R_COL in df.columns:
        df[f"{TOT_R_COL}_filt"] = lp_butter(df[TOT_R_COL], FS, FC_HZ, ORDER_EXOG)

    # ----- Save filtered dataset -----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()