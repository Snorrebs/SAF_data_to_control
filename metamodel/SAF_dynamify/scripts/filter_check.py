CSV = "data/interim/plant_1s_filtered_normalized.csv"  # or point to raw if you want to compare
TIMESTAMP = "timestamp"
RES_COL = "Tot_Resistance_mOhm"
RES_FILT_COL = "Tot_Resistance_mOhm_filt"
MINUTES = 10
START = None  # e.g., "2022-07-07 02:00:30+02"
# ===============

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

pdf = pd.read_csv(CSV, parse_dates=[TIMESTAMP])
if START:
    t0 = pd.Timestamp(START)
else:
    t0 = pdf[TIMESTAMP].iloc[0]
mask = (pdf[TIMESTAMP] >= t0) & (pdf[TIMESTAMP] <= t0 + timedelta(minutes=MINUTES))
win = pdf.loc[mask]

plt.figure(figsize=(12,6))
plt.plot(win[TIMESTAMP], win[RES_COL], label=RES_COL)
if RES_FILT_COL in win.columns:
    plt.plot(win[TIMESTAMP], win[RES_FILT_COL], label=RES_FILT_COL)
plt.legend(); plt.xlabel("time"); plt.ylabel("mΩ"); plt.title("Resistance vs filtered")
plt.tight_layout(); plt.show()