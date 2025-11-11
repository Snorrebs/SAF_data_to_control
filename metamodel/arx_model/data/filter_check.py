#!/usr/bin/env python3
"""
plot_filtered_vs_unfiltered.py

Quick check: plot raw vs. filtered resistance from the same dataset
over a short time window.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --- CONFIG ---

csv_path = "metamodel/arx_model/data/cleaned_data/filtered_data/Filtered_o5_01.csv"

# Columns to plot (both must exist in the same CSV)
res_cols = ["Tot_Resistance_mOhm", "Tot_Resistance_mOhm_filt"]

time_col = "timestamp"
start_time = "2022-07-07 12:00:00+02:00"
window_minutes = 10


# --- LOAD & SLICE ---

df = pd.read_csv(csv_path)
df[time_col] = pd.to_datetime(df[time_col])

start = pd.to_datetime(start_time)
end = start + timedelta(minutes=window_minutes)

df_window = df[(df[time_col] >= start) & (df[time_col] <= end)]

if df_window.empty:
    raise ValueError("No data in selected window. Check start_time and window_minutes.")


# --- PLOT ---
plt.figure(figsize=(12, 6))

for col in res_cols:
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found in CSV.")
        continue
    label = "filtered" if "filt" in col.lower() else "unfiltered"
    plt.plot(df_window[time_col], df_window[col], label=label, linewidth=2.0 if "filt" in col else 1.0, alpha=0.7)

plt.title(f"Filtered vs Unfiltered Resistance ({window_minutes} min window)")
plt.xlabel("Time")
plt.ylabel("Resistance [mΩ]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

print(f"[plot] {start} → {end}")
plt.show()
