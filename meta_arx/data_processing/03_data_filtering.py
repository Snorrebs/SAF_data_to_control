#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

# ---------- CONFIG ----------
IN_CSV  = Path("meta_arx/data/1s_data_from_plant/07_24.csv")
OUT_CSV = Path("meta_arx/data/filt_data/07_24_filt_pos_01.csv")

# Target signal for ARX: electrode 1 resistance (mΩ)
TARGET_COL = "El1_Resistance_mOhm"

# Electrode holder position column (El1, currently in cm from PI)
POS_COL = "El1_pos_m"

# Name for movement speed signal (Δ holder position per second, in m/s)
SPEED_COL = "El1_dpos_mps"

# Butterworth params (1 Hz sampling)
FS = 1.0          # sampling rate [Hz] (1 s data → 1 Hz)
ORDER_TGT  = 2    # order for target
ORDER_EXOG = 4    # order for exogenous
FC_HZ = 0.1      # low-pass cutoff frequency [Hz] (≈ 20 s time scale)

# Whether to automatically convert electrode positions from cm → m if needed
FORCE_ELECTRODE_CM_TO_M = True


def lp_butter(series: pd.Series, fs: float, fc: float, order: int) -> pd.Series:
    """
    Apply zero-phase Butterworth low-pass filter to a 1D series.
    """
    series = series.astype(float)
    nyq = fs / 2.0
    wn = fc / nyq
    b, a = butter(order, wn, btype="low", analog=False)
    # Short gaps → interpolate to avoid edge artifacts
    s = series.interpolate(limit=5).ffill().bfill().values
    y = filtfilt(b, a, s, method="gust")
    return pd.Series(y, index=series.index)


def main():
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    df = pd.read_csv(IN_CSV)

    ts_col = "timestamp"
    if ts_col not in df.columns:
        raise ValueError(f"Expected a '{ts_col}' column in {IN_CSV}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index().asfreq("1s")

    # ----- Electrode unit sanity (cm -> m) -----
    if FORCE_ELECTRODE_CM_TO_M and POS_COL in df.columns:
        mean_pos = df[POS_COL].dropna().mean()
        if pd.notna(mean_pos) and mean_pos > 10.0:
            df[POS_COL] = df[POS_COL] / 100.0
            print(f"[unit] Converted {POS_COL} from cm to m (mean={mean_pos:.3f})")

    # ----- Compute RMS_V_transformer if missing -----
    if "RMS_V_transformer" not in df.columns:
        volts = [c for c in ["UL1N_V", "UL2N_V", "UL3N_V"] if c in df.columns]
        if len(volts) != 3:
            raise ValueError(
                "Need UL1N_V, UL2N_V, UL3N_V to compute RMS_V_transformer, "
                f"found: {volts}"
            )
        df["RMS_V_transformer"] = (
            df[volts].astype(float).pow(2).mean(axis=1).pow(0.5)
        )

    # ----- Compute electrode 1 holder movement speed Δpos / 1 s (m/s) -----
    if POS_COL not in df.columns:
        raise ValueError(f"Missing electrode position column '{POS_COL}'")
    df[SPEED_COL] = df[POS_COL].astype(float).diff()
    # For first sample, just set speed = 0
    if len(df) > 0:
        df.loc[df.index[0], SPEED_COL] = 0.0


    # ----- Filter ARX target: El1 resistance -----
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")
    df[f"{TARGET_COL}_filt"] = lp_butter(df[TARGET_COL], FS, FC_HZ, ORDER_TGT)

    # (Optional: also filter furnace total resistance for diagnostics if present)
    if "Tot_Resistance_mOhm" in df.columns:
        df["Tot_Resistance_mOhm_filt"] = lp_butter(
            df["Tot_Resistance_mOhm"], FS, FC_HZ, ORDER_EXOG
        )

    # ----- Filter exogenous SISO signals (speed, current, RMS voltage) -----
    exog_to_filter = [POS_COL, "El1_kA", "RMS_V_transformer"]
    for c in exog_to_filter:
        if c not in df.columns:
            raise ValueError(f"Missing exogenous column '{c}' needed for ARX.")
        df[f"{c}_filt"] = lp_butter(df[c], FS, FC_HZ, ORDER_EXOG)

    # ----- Save filtered dataset -----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
