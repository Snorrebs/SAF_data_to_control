#!/usr/bin/env python3
# filter_check.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
plt.tight_layout()

IN_CSV = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_filtered.csv")

START = "2022-07-07 02:00:30+02:00"  # example from your data
MINUTES = 20

def main():
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    df = pd.read_csv(IN_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    t0 = pd.to_datetime(START)
    t1 = t0 + pd.Timedelta(minutes=MINUTES)
    w = df.loc[t0:t1]

    cols = []
    if "Tot_Resistance_mOhm" in w.columns: cols.append("Tot_Resistance_mOhm")
    if "Tot_Resistance_mOhm_filt" in w.columns: cols.append("Tot_Resistance_mOhm_filt")
    if not cols:
        raise ValueError("Need Tot_Resistance_mOhm and/or Tot_Resistance_mOhm_filt in CSV.")

    ax = w[cols].plot(title=f"Resistance (raw vs filtered)\n{t0} → {t1}", lw=1.2)
    ax.set_ylabel("mΩ")
    ax.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
