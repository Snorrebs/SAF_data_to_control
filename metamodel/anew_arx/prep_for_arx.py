import pandas as pd
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import PolynomialFeatures

RAW_CSV = "metamodel/arx_model/data/cleaned_data/07_08_1s.csv"
OUT_CSV = "metamodel/arx_model/data/cleaned_data/filtered_data/Filtered_o5_01.csv"
PLS_MODEL_PATH = "metamodel/models/plsr_ge28.pkl"
POLY_PATH = "plsr_poly_transformer.joblib"  # if missing, we recreate

FS = 1.0     # 1 Hz sampling
FC = 0.1    # low-pass cutoff (Hz) ≈ 6.7 s; tweak 0.10–0.20 if needed
ORDER = 5

Y_COL = 'Tot Resistance (mΩ), ge28'
BASE_X_COLS = [
    'RMS Voltage at Transformer (V), ge1',
    'El1 pos (m), ge2', 'El2 pos (m), ge3', 'El3 pos (m), ge4',
    'CW1 Thickness (m), ge62', 'CW2 Thickness (m), ge63', 'CW3 Thickness (m), ge64',
    'res. CW 1 (mΩ*m), ge6', 'res. CW 2 (mΩ*m), ge7', 'res. CW 3 (mΩ*m), ge8',
    'res. SiC12 (mΩ*m), ge10', 'res. SiC23 (mΩ*m), ge11', 'res. SiC31 (mΩ*m), ge12',
]
# Path B latents for now
CW_thk = (0.25, 0.25, 0.25)
rCW    = (2.3,  2.3,  2.3 )
rSiC   = (30.0, 30.0, 30.0)

EXOG = ['El1_kA','El2_kA','El3_kA','Tap_A','Tap_B','Tap_C',
        'RMS_V_transformer','El1_pos_m','El2_pos_m','El3_pos_m']

def butter_lp(s: pd.Series, fc: float, fs: float, order=2):
    nyq = 0.5*fs
    Wn = fc/nyq
    b,a = butter(order, Wn, btype='low', analog=False)
    x = s.astype(float).interpolate(limit_direction='both')
    y = filtfilt(b, a, x.values, padlen=min(3*max(len(a),len(b)), max(5, len(x)//4)))
    return pd.Series(y, index=s.index, name=s.name)

def zscore(s: pd.Series):
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd if sd and sd>0 else 1.0)

# 1) Load raw 1s data
df = pd.read_csv(RAW_CSV, parse_dates=['timestamp']).set_index('timestamp')
df = df.sort_index().asfreq('1s')

# Derived RMS voltage (unfiltered base signal)
df['RMS_V_transformer'] = df[['UL1N_V','UL2N_V','UL3N_V']].abs().mean(axis=1)

# Ensure positions in meters (unfiltered base signal)
for c in ['El1_pos_m','El2_pos_m','El3_pos_m']:
    if c in df and df[c].mean() > 10:
        df[c] = df[c] / 100.0

# 2) FILTER plant + exogenous (but NOT the metamodel input we’ll use)
to_filter = [c for c in (EXOG + ['Tot_Resistance_mOhm']) if c in df.columns]
for c in to_filter:
    df[c + '_filt'] = butter_lp(df[c], FC, FS, ORDER)

# 3) Build UNFILTERED base-X for metamodel and predict meta (static, unfiltered)
baseX_unf = pd.DataFrame(index=df.index, data={
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
})

try:
    poly = joblib.load(POLY_PATH)
except:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(baseX_unf.values[:3])  # init structure

X_poly = poly.transform(baseX_unf.values)
pls = joblib.load(PLS_MODEL_PATH)
y_meta = pls.predict(X_poly).ravel()  # Ω → mΩ
df['Tot_Resistance_meta'] = y_meta  # UNFILTERED meta

# 4) Filtered residual target for ARX (filtered plant – unfiltered meta)
df['Tot_Resistance_mOhm_filt'] = df['Tot_Resistance_mOhm_filt']  # explicit
df['residual_filt'] = df['Tot_Resistance_mOhm_filt'] - df['Tot_Resistance_meta']

# 5) Normalize (z) the FILTERED exogenous for ARX
exog_filt_cols = [c+'_filt' for c in EXOG if c+'_filt' in df.columns]
for c in exog_filt_cols:
    df[c + '_zn'] = zscore(df[c])

# 6) Save prepared CSV
df.to_csv(OUT_CSV)
print(f"[prepared] → {OUT_CSV}")
print("Columns for ARX target/exog:",
      "target = residual_filt ; exog =", [c+'_zn' for c in exog_filt_cols])
