#!/usr/bin/env python3
"""
Quick script to inspect raw vs filtered signals.

Edit the CONFIG section:
- CSV_PATH: which filtered file to inspect
- START / END: time window (or set to None to use full range)
- VARS: list of (raw_col, filt_col) pairs to plot
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
CSV_PATH = Path("meta_arx/data/filt_data/07_24_filt.csv")

# Time window (set to None to use full range)
START = "2022-07-06 22:01:00+00:00"   # or None
END   = "2022-07-06 23:00:00+00:00"   # or None

# Variables to plot: (raw_column_name, filtered_column_name)
VARS = [
    ("El1_Resistance_mOhm", "El1_Resistance_mOhm_filt"),
    # ("Tot_Resistance_mOhm", "Tot_Resistance_mOhm_filt"),
    # ("El1_dpos_mps", "El1_dpos_mps_filt"),
    # ("El1_kA", "El1_kA_filt"),
    # ("RMS_V_transformer", "RMS_V_transformer_filt"),
]
# Just comment/uncomment lines in VARS to choose which signals to see.
# -----------------------------------------


def main():
    # Load
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the CSV.")
    df = df.set_index("timestamp").sort_index()

    # Time slicing
    if START is not None:
        df = df[df.index >= pd.to_datetime(START)]
    if END is not None:
        df = df[df.index < pd.to_datetime(END)]

    if df.empty:
        raise ValueError("No data left after applying START/END window.")

    # Plot each raw/filtered pair
    for raw_col, filt_col in VARS:
        if raw_col not in df.columns:
            print(f"[WARN] raw column '{raw_col}' not in DataFrame, skipping.")
            continue
        if filt_col not in df.columns:
            print(f"[WARN] filtered column '{filt_col}' not in DataFrame, skipping.")
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[raw_col], label=f"{raw_col} (raw)", alpha=0.5)
        plt.plot(df.index, df[filt_col], label=f"{filt_col} (filtered)", linewidth=1.8)
        plt.title(f"{raw_col} – raw vs filtered")
        plt.xlabel("Time")
        plt.ylabel(raw_col)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
