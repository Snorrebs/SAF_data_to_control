#!/usr/bin/env python3
# normalize_data.py
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
IN_CSV  = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_filtered.csv")
OUT_CSV = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_filtered_norm.csv")
STATS_JSON = Path("meta_arx/data/1s_data_from_plant/0702_0703_norm_stats.json")

# Which filtered columns to z-score (typical ARX inputs)
to_norm = [
    "El1_kA_filt","El2_kA_filt","El3_kA_filt",
    "Tap_A_filt","Tap_B_filt","Tap_C_filt",
    "RMS_V_transformer_filt",
    "El1_pos_m_filt","El2_pos_m_filt","El3_pos_m_filt",
]
# Optional: also z-score target; keep the physical one too
NORM_TARGET = False
target_col = "residual_filt"
target_norm_name = "residual_filt_zn"

# If you have a train/val split, put train range here to fit μ/σ on train only
TRAIN_START = None  # e.g., "2022-07-10T00:00:00Z"
TRAIN_END   = None  # e.g., "2022-07-20T00:00:00Z"

# ---------- SCRIPT ----------
def zscore(s: pd.Series, mu: float, std: float) -> pd.Series:
    std = std if std > 0 else 1.0
    return (s - mu) / std

def main():
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    df = pd.read_csv(IN_CSV)
    ts_col = "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index()

    # Choose window for fitting μ/σ
    if TRAIN_START and TRAIN_END:
        train_df = df.loc[pd.to_datetime(TRAIN_START):pd.to_datetime(TRAIN_END)]
    else:
        train_df = df  # fallback: fit on all data (OK if you’re just prototyping)

    stats = {}
    # Inputs
    for c in to_norm:
        if c not in df.columns:
            print(f"[warn] missing column to normalize: {c}")
            continue
        mu = float(train_df[c].mean())
        sd = float(train_df[c].std(ddof=0))
        stats[c] = {"mu": mu, "sd": sd}
        df[f"{c}_zn"] = zscore(df[c], mu, sd)

    # Target (optional)
    if NORM_TARGET:
        if target_col not in df.columns:
            raise ValueError(f"Missing target column '{target_col}'")
        mu = float(train_df[target_col].mean())
        sd = float(train_df[target_col].std(ddof=0))
        stats[target_col] = {"mu": mu, "sd": sd}
        df[target_norm_name] = zscore(df[target_col], mu, sd)

    # Save artifacts
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(OUT_CSV, index=False)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] wrote {OUT_CSV}")
    print(f"[OK] wrote {STATS_JSON}")

if __name__ == "__main__":
    main()
