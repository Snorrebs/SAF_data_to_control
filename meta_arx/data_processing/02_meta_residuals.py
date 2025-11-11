#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

IN_CSV  = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s.csv")
OUT_CSV = Path("meta_arx/data/1s_data_from_plant/0702_0703_1s_residual.csv")
PLS_PATH = Path("meta_arx/metamodel/plsr_with_terms.joblib")

# --- 1) load plant day ---
df = pd.read_csv(IN_CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index().asfreq('1S')

# small NA guard to avoid filter / transform hiccups
df = df.interpolate(limit=5).ffill().bfill()

# true RMS (sqrt(mean of squares)) from phase-to-neutral voltages
df['RMS_V_transformer'] = (df[['UL1N_V','UL2N_V','UL3N_V']].astype(float).pow(2).mean(axis=1).pow(0.5))

# positions: if they look like centimeters, convert to meters once
for c in ['El1_pos_m','El2_pos_m','El3_pos_m']:
    if df[c].mean() > 10:
        df[c] = df[c] / 100.0

# Path B latents (constants for now)
CW_thk = (0.25, 0.25, 0.25)
rCW    = (2.3,  2.3,  2.3 )
rSiC   = (30.0, 30.0, 30.0)

BASE_X_COLS = [
    'RMS Voltage at Transformer (V), ge1',
    'El1 pos (m), ge2', 'El2 pos (m), ge3', 'El3 pos (m), ge4',
    'CW1 Thickness (m), ge62', 'CW2 Thickness (m), ge63', 'CW3 Thickness (m), ge64',
    'res. CW 1 (mΩ*m), ge6', 'res. CW 2 (mΩ*m), ge7', 'res. CW 3 (mΩ*m), ge8',
    'res. SiC12 (mΩ*m), ge10', 'res. SiC23 (mΩ*m), ge11', 'res. SiC31 (mΩ*m), ge12',
]
Y_COL = 'Tot Resistance (mΩ), ge28'   # just a label for bookkeeping

# --- 2) base features (13) ---
baseX = pd.DataFrame(index=df.index, data={
    'RMS Voltage at Transformer (V), ge1': df['RMS_V_transformer'],
    'El1 pos (m), ge2': df['El1_pos_m'],
    'El2 pos (m), ge3': df['El2_pos_m'],
    'El3 pos (m), ge4': df['El3_pos_m'],
    'CW1 Thickness (m), ge62': CW_thk[0],
    'CW2 Thickness (m), ge63': CW_thk[1],
    'CW3 Thickness (m), ge64': CW_thk[2],
    'res. CW 1 (mΩ*m), ge6': rCW[0],
    'res. CW 2 (mΩ*m), ge7': rCW[1],
    'res. CW 3 (mΩ*m), ge8': rCW[2],
    'res. SiC12 (mΩ*m), ge10': rSiC[0],
    'res. SiC23 (mΩ*m), ge11': rSiC[1],
    'res. SiC31 (mΩ*m), ge12': rSiC[2],
})[BASE_X_COLS].astype(float)

# --- 3) expand to XIS=104 exactly like training (deg=2, no bias) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(baseX.values[:2])  # lock deterministic order
X_poly = poly.transform(baseX.values)

# --- 4) predict with saved PLSR ---
art = joblib.load(PLS_PATH)
pls=art['pls']
print(art['n_components'], len(art['x_terms']))
# quick sanity: columns match expected count
n_expected = len(art['x_terms'])
if X_poly.shape[1] != n_expected:
    raise ValueError(f"Feature count mismatch: got {X_poly.shape[1]}, PLS expects {n_expected}")

y_meta = pls.predict(X_poly).ravel()

# NOTE: adjust scaling only if your stored model outputs Ω not mΩ.
# If your PLS outputs Ω, convert to mΩ:
# y_meta_mOhm = y_meta * 1e3
# If it already outputs mΩ, keep as-is:
y_meta_mOhm = y_meta/1000  # or: y_meta/1000 if your pkl is in µΩ, etc.

df['Tot_Resistance_meta'] = y_meta_mOhm

# --- 5) residuals ---
df[Y_COL] = df['Tot_Resistance_mOhm'].astype(float)
df['residual'] = df[Y_COL] - df['Tot_Resistance_meta']

# write a separate file so you keep the original intact
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV)

# handy summary
print(df[['Tot_Resistance_mOhm','Tot_Resistance_meta','residual']].describe())
print(f"[OK] wrote {OUT_CSV}")
