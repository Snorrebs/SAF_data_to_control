import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures  # only if you didn't save `poly`

# 1) load plant day
df = pd.read_csv("SAF_data_to_control/metamodel/arx_model/data/cleaned_data/07_08_1s.csv", parse_dates=['timestamp']).set_index('timestamp')
df = df.sort_index().asfreq('1S')  # explicit 1 s grid
df['RMS_V_transformer'] = df[['UL1N_V','UL2N_V','UL3N_V']].abs().mean(axis=1)
for c in ['El1_pos_m','El2_pos_m','El3_pos_m']:
    if df[c].mean() > 10: df[c] /= 100.0

# Path B latents (for now)
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
Y_COL = 'Tot Resistance (mΩ), ge28'

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
})

# 2) expand to XIS=104 the same way you trained
poly = PolynomialFeatures(degree=2, include_bias=False)  # must match training
poly.fit(baseX.values[:2])  # dummy fit just to init structure; order is deterministic

X_poly = poly.transform(baseX.values)

# 3) predict with your saved PLSR
pls = joblib.load("SAF_data_to_control/metamodel/models/plsr_ge28.pkl")
y_meta = pls.predict(X_poly).ravel()
df['Tot_Resistance_meta'] = y_meta/1000

# 4) residuals

df[Y_COL] = df['Tot_Resistance_mOhm']
df['residual'] = df[Y_COL] - df['Tot_Resistance_meta']
df.to_csv("SAF_data_to_control/metamodel/arx_model/data/cleaned_data/07_08_1s_residuals.csv")

print(df[['Tot_Resistance_mOhm','Tot_Resistance_meta','residual']].describe())


# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,5))
# plt.plot(df.index, df['Tot_Resistance_mOhm'], label='Measured (plant)')
# plt.plot(df.index, df['Tot_Resistance_meta'], label='Metamodel (scaled)')
# plt.legend(); plt.ylabel('Resistance [mΩ]'); plt.show()
