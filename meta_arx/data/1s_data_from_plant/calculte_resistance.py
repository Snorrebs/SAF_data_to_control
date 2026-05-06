import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = Path("meta_arx/data/1s_data_from_plant/07_24.csv")
df = pd.read_csv(IN_CSV)

# =============================
# Basic resistance sanity check
# =============================
U = df["UL1N_V"].values
I = df["El1_kA"].values

average_U = U.mean()
average_I = I.mean()
R_cal = (average_U / average_I)
print(f"Calculated resistance R_cal (avg V / avg I): {R_cal:.4f} mOhm")

R = df["El1_Resistance_mOhm"].values
print(f"Average resistance from tag: {R.mean():.4f} mOhm")


# =============================
# Position & Current Analysis
# =============================
pos1 = df["El1_pos_m"].values
pos2 = df["El2_pos_m"].values
pos3 = df["El3_pos_m"].values

# Convert to cm for readability
pos1_cm = pos1 * 100

# --- 1) Static correlation ---
corr_level = np.corrcoef(pos1_cm, I)[0, 1]
print(f"Level correlation corr(pos, I): {corr_level:.4f}")

# --- 2) Delta correlation ---
dpos = np.diff(pos1_cm)
dI = np.diff(I)

corr_delta = np.corrcoef(dpos, dI)[0, 1]
print(f"Delta correlation corr(dpos, dI): {corr_delta:.4f}")

# --- 3) Lagged cross-correlation (Δpos -> ΔI) ---
max_lag = 30  # seconds
lags = np.arange(0, max_lag + 1)
xcorr = []

for lag in lags:
    if lag == 0:
        xcorr.append(np.corrcoef(dpos, dI)[0, 1])
    else:
        xcorr.append(np.corrcoef(dpos[:-lag], dI[lag:])[0, 1])

xcorr = np.array(xcorr)

best_lag = lags[np.argmax(np.abs(xcorr))]
print(f"Max |corr(dpos(t), dI(t+lag))| at lag = {best_lag} s")
print(f"Correlation at that lag: {xcorr[best_lag]:.4f}")

# =============================
# Plots
# =============================

# Histogram of positions
plt.figure(figsize=(10, 5))
plt.hist(pos1_cm, bins=30, alpha=0.6, label="El1")
plt.hist(pos2 * 100, bins=30, alpha=0.6, label="El2")
plt.hist(pos3 * 100, bins=30, alpha=0.6, label="El3")
plt.xlabel("Position (cm)")
plt.ylabel("Frequency")
plt.title("Distribution of Electrode Positions")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter Δpos vs ΔI
plt.figure(figsize=(6, 6))
plt.scatter(dpos, dI, alpha=0.3)
plt.xlabel("Δ Position (cm)")
plt.ylabel("Δ Current (kA)")
plt.title("Δpos vs ΔI")
plt.tight_layout()
plt.show()

# Cross-correlation plot
plt.figure(figsize=(8, 4))
plt.plot(lags, xcorr)
plt.axhline(0, linestyle="--")
plt.xlabel("Lag (seconds)")
plt.ylabel("Correlation")
plt.title("Cross-correlation: Δpos(t) vs ΔI(t+lag)")
plt.tight_layout()
plt.show()

