#!/usr/bin/env python3
# data_filtering.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path

# ---------- CONFIG ----------
IN_CSV  = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_residual.csv")
OUT_CSV = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_filtered.csv")

# Columns to filter for ARX inputs/diagnostics
exog_cols = [
    "El1_kA","El2_kA","El3_kA",
    "Tap_A","Tap_B","Tap_C",
    "RMS_V_transformer",
    "El1_pos_m","El2_pos_m","El3_pos_m",
]
target_col = "residual"
also_filter_raw_res = True  

# Butterworth params (1 Hz sampling)
FS = 2.0
ORDER_EXOG = 4
ORDER_TGT  = 2
FC_HZ = 0.05  #  ~ 1/(2π*0.01) ≈ 16 s;


def lp_butter(series: pd.Series, fs: float, fc: float, order: int) -> pd.Series:
    series = series.astype(float)
    nyq = fs / 2.0
    wn = fc / nyq
    print(wn)
    b, a = butter(order, wn, btype="low", analog=False)
    # Short gaps → interpolate to avoid edge artifacts
    s = series.interpolate(limit=5).ffill().bfill().values
    y = filtfilt(b, a, s, method="gust")
    return pd.Series(y, index=series.index)

def main():
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    df = pd.read_csv(IN_CSV)
    ts_col = "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index().asfreq("1s")

    # Filter target
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")
    df[f"{target_col}_filt"] = lp_butter(df[target_col], FS, FC_HZ, ORDER_TGT)


    if also_filter_raw_res and "Tot_Resistance_mOhm" in df.columns:
        df["Tot_Resistance_mOhm_filt"] = lp_butter(df["Tot_Resistance_mOhm"], FS, FC_HZ, ORDER_EXOG)

    # Filter exog
    for c in exog_cols:
        if c in df.columns:
            df[f"{c}_filt"] = lp_butter(df[c], FS, FC_HZ, ORDER_EXOG)
        else:
            print(f"[warn] missing exog column: {c}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
