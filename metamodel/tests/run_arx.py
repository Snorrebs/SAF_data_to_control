import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX  # ARX via SARIMAX
import matplotlib.pyplot as plt


df = pd.read_csv("metamodel/arx_model/data/cleaned_data/filtered_data/Filtered_o5_01.csv", parse_dates=['timestamp']).set_index('timestamp')
# --- choose exogenous drivers for the residual dynamics ---
# Ensure fixed 30s frequency (matches your resampling)
df = df.asfreq('1s')  # or: df = df.asfreq(pd.infer_freq(df.index) or '30S')
df = df.sort_index()

# Lag and drop NaNs again (after asfreq)
y = df['residual_filt']
exog_cols = [c for c in ['El1_kA','El2_kA','El3_kA','Tap_A','Tap_B','Tap_C',
                         'El1_pos_m','El2_pos_m','El3_pos_m','RMS_V_transformer'] if c in df.columns]
X = df[exog_cols].shift(1)
Z = pd.concat([y, X], axis=1).dropna()


# align
Z = pd.concat([y, X], axis=1).dropna()
y_aligned = Z['residual_filt']
X_aligned = Z[exog_cols]

# split without shuffling (block split)
n = len(Z); n_train = int(0.7*n)
y_tr, y_te = y_aligned.iloc[:n_train], y_aligned.iloc[n_train:]
X_tr, X_te = X_aligned.iloc[:n_train], X_aligned.iloc[n_train:]

# z-score EXOG only (train fit → test transform)
scaler = StandardScaler().fit(X_tr.values)
X_tr_s = scaler.transform(X_tr.values)
X_te_s = scaler.transform(X_te.values)

# Try a slightly higher AR order and robust optimizer
model = SARIMAX(y_tr, order=(2,0,0), exog=X_tr_s, trend='c',
                enforce_stationarity=True, enforce_invertibility=True)
res_arx = model.fit(method='lbfgs', maxiter=300, disp=False)

# --- in-sample fit + out-of-sample forecast on residuals ---
yhat_tr = res_arx.predict(start=y_tr.index[0], end=y_tr.index[-1], exog=X_tr_s)
yhat_te = res_arx.predict(start=y_te.index[0], end=y_te.index[-1], exog=X_te_s)

# stitch
yhat_all = pd.concat([yhat_tr, yhat_te])

# --- dynamified resistance = meta + residual_hat ---
df['residual_hat'] = yhat_all.reindex(df.index)
df['Tot_Resistance_dyn'] = df['Tot_Resistance_meta'] + df['residual_hat']

# --- evaluate ---
def eval_block(y_true, y_meta, y_dyn, label):
    r2_m  = r2_score(y_true, y_meta)
    r2_d  = r2_score(y_true, y_dyn)
    rmse_m = root_mean_squared_error(y_true, y_meta)
    rmse_d = root_mean_squared_error(y_true, y_dyn)
    print(f"[{label}]  R2 meta={r2_m:.3f} → dyn={r2_d:.3f} | RMSE meta={rmse_m:.4f} → dyn={rmse_d:.4f} mΩ")

# align eval frames
mask_tr = df.index.isin(y_tr.index)
mask_te = df.index.isin(y_te.index)

eval_block(df.loc[mask_tr,'Tot_Resistance_mOhm'],
           df.loc[mask_tr,'Tot_Resistance_meta'],
           df.loc[mask_tr,'Tot_Resistance_dyn'], "Train")

eval_block(df.loc[mask_te,'Tot_Resistance_mOhm'],
           df.loc[mask_te,'Tot_Resistance_meta'],
           df.loc[mask_te,'Tot_Resistance_dyn'], "Test")

# Mask for test indices
mask_te = df.index.isin(y_te.index)
df_test = df.loc[mask_te]

plt.figure(figsize=(12,5))
plt.plot(df_test.index, df_test['Tot_Resistance_mOhm'], label='Plant (measured)', lw=2)
plt.plot(df_test.index, df_test['Tot_Resistance_meta'], label='Metamodel (steady-state)', alpha=0.7)
plt.plot(df_test.index, df_test['Tot_Resistance_dyn'], label='Metamodel + ARX (dynamic)', alpha=0.9)
plt.xlabel("Time")
plt.ylabel("Total resistance [mΩ]")
plt.title("ARX performance – Test data only (out-of-sample)")
plt.legend()
plt.tight_layout()
plt.show()

