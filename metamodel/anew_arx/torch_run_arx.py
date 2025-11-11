#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

# ---------- Config ----------
CSV = "metamodel/arx_model/data/cleaned_data/filtered_data/Filtered_o5_01.csv"
AR_ORDER = 5          # ARX(2)
EXOG_LAG = 3          # exogenous at t-1 (matches your code)
RIDGE_LAMBDA = 1e-3   # 0 for OLS; small >0 for stability
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load & prep ----------
df = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp')
df = df.asfreq('1s').sort_index()

y = df['residual_filt']  # target residuals
exog_cols = [c for c in ['El1_kA','El2_kA','El3_kA','Tap_A','Tap_B','Tap_C',
                         'El1_pos_m','El2_pos_m','El3_pos_m','RMS_V_transformer']
             if c in df.columns]

# Build lagged variables
def lag(series: pd.Series, k: int) -> pd.Series:
    return series.shift(k)

ar_lags = {f'y_lag{i}': lag(y, i) for i in range(1, AR_ORDER + 1)}
X_exog = df[exog_cols].shift(EXOG_LAG)

Z = pd.concat([y.rename('y'), pd.DataFrame(ar_lags), X_exog], axis=1).dropna()

# Align
y_aligned = Z['y']
X_ar = Z[[f'y_lag{i}' for i in range(1, AR_ORDER + 1)]]
X_ex = Z[exog_cols]

# Train/test split (no shuffling)
n = len(Z); n_train = int(0.7 * n)
y_tr, y_te = y_aligned.iloc[:n_train], y_aligned.iloc[n_train:]
X_ar_tr, X_ar_te = X_ar.iloc[:n_train], X_ar.iloc[n_train:]
X_ex_tr, X_ex_te = X_ex.iloc[:n_train], X_ex.iloc[n_train:]

# Z-score exogenous using train fit → transform test
scaler = StandardScaler().fit(X_ex_tr.values)
X_ex_tr_s = scaler.transform(X_ex_tr.values)
X_ex_te_s = scaler.transform(X_ex_te.values)

# Design matrices: [1, y_{t-1}, y_{t-2}, exog_{t-1}...]
def build_design(X_ar_np, X_ex_np):
    ones = np.ones((X_ar_np.shape[0], 1), dtype=np.float32)
    return np.hstack([ones, X_ar_np.astype(np.float32), X_ex_np.astype(np.float32)])

X_tr_np = build_design(X_ar_tr.values, X_ex_tr_s)
X_te_np = build_design(X_ar_te.values, X_ex_te_s)
y_tr_np = y_tr.values.astype(np.float32).reshape(-1, 1)
y_te_np = y_te.values.astype(np.float32).reshape(-1, 1)

# ---------- Standardize AR lags too (fit on train only) ----------
from sklearn.preprocessing import StandardScaler as _SS

ar_scaler = _SS().fit(X_ar_tr.values)
X_ar_tr_s = ar_scaler.transform(X_ar_tr.values)
X_ar_te_s = ar_scaler.transform(X_ar_te.values)

# Design matrices now use standardized AR + standardized exog
X_tr_np = build_design(X_ar_tr_s, X_ex_tr_s)
X_te_np = build_design(X_ar_te_s, X_ex_te_s)

# Use float64 for the solve to improve conditioning
DTYPE_SOLVE = torch.float64

X_tr_t = torch.from_numpy(X_tr_np).to(DEVICE, dtype=DTYPE_SOLVE)  # [N,p]
y_tr_t = torch.from_numpy(y_tr_np).to(DEVICE, dtype=DTYPE_SOLVE)  # [N,1]
X_te_t = torch.from_numpy(X_te_np).to(DEVICE, dtype=DTYPE_SOLVE)

# Ridge: (XᵀX + λI)β = Xᵀy, but DON'T regularize the bias term
XtX = X_tr_t.T @ X_tr_t
Xty = X_tr_t.T @ y_tr_t

if RIDGE_LAMBDA > 0:
    lam = torch.full((XtX.shape[0],), RIDGE_LAMBDA, dtype=DTYPE_SOLVE, device=DEVICE)
    lam[0] = 0.0  # no penalty on intercept
    XtX = XtX + torch.diag(lam)

# Robust solve with safe fallback
try:
    beta = torch.linalg.solve(XtX, Xty)                 # [p,1]
except RuntimeError:
    # Fall back to least-squares (SVD-based), more forgiving if XtX is near-singular
    beta, *_ = torch.linalg.lstsq(X_tr_t, y_tr_t, rcond=None)

# Predictions
yhat_tr_t = X_tr_t @ beta                   # [Ntr,1]
yhat_te_t = X_te_t @ beta                   # [Nte,1]

# Back to CPU/NumPy
yhat_tr = pd.Series(yhat_tr_t.squeeze(1).detach().cpu().numpy(), index=y_tr.index, name='residual_hat')
yhat_te = pd.Series(yhat_te_t.squeeze(1).detach().cpu().numpy(), index=y_te.index, name='residual_hat')
yhat_all = pd.concat([yhat_tr, yhat_te])

# ---------- Reconstruct dynamified resistance ----------
df['residual_hat'] = yhat_all.reindex(df.index)
df['Tot_Resistance_dyn'] = df['Tot_Resistance_meta'] + df['residual_hat']

# ---------- Evaluation ----------
def eval_block(y_true, y_meta, y_dyn, label):
    # drop NaNs consistently
    eval_df = pd.concat([y_true, y_meta, y_dyn], axis=1).dropna()
    yt = eval_df.iloc[:,0].values
    ym = eval_df.iloc[:,1].values
    yd = eval_df.iloc[:,2].values
    r2_m  = r2_score(yt, ym)
    r2_d  = r2_score(yt, yd)
    rmse_m = root_mean_squared_error(yt, ym)
    rmse_d = root_mean_squared_error(yt, yd)
    print(f"[{label}]  R2 meta={r2_m:.3f} → dyn={r2_d:.3f} | RMSE meta={rmse_m:.4f} → dyn={rmse_d:.4f} mΩ")

mask_tr = df.index.isin(y_tr.index)
mask_te = df.index.isin(y_te.index)

eval_block(df.loc[mask_tr,'Tot_Resistance_mOhm'],
           df.loc[mask_tr,'Tot_Resistance_meta'],
           df.loc[mask_tr,'Tot_Resistance_dyn'], "Train")

eval_block(df.loc[mask_te,'Tot_Resistance_mOhm'],
           df.loc[mask_te,'Tot_Resistance_meta'],
           df.loc[mask_te,'Tot_Resistance_dyn'], "Test")

# ---------- Prepare aligned, standardized exogenous for all rows in Z ----------
# Z already has y_t and exog at t-EXOG_LAG (you built it with .shift(EXOG_LAG))
# We'll rebuild the standardized exog matrix aligned to Z rows.
X_ex_all = Z[exog_cols].values
X_ex_all_s = scaler.transform(X_ex_all)  # use the *train-fit* exog scaler

# Indices to work in Z-time (not full df)
Z_idx = Z.index

def arx_step(y_hist, exog_s_row, beta):
    """
    Single prediction step:
      y_hist: list/array of recent predicted residuals [y_{t-1}, y_{t-2}, ...] length=AR_ORDER
      exog_s_row: standardized exog vector for this step (shape [n_exog])
      beta: [p,1] coefficients (bias, AR_ORDER lags, exog)
    returns: float prediction for y_t
    """
    # Build row = [1, y_{t-1}, y_{t-2}, exog_s...]
    row = np.concatenate(([1.0], np.array(y_hist, dtype=np.float64), exog_s_row.astype(np.float64)))
    return float(row @ beta.squeeze(1).detach().cpu().numpy())

def forecast_horizon_OL(t0, H=30, mode="true_exog"):
    """
    Open-loop forecast for H seconds ahead starting *after* time t0.
    t0 must be one of Z.index (where lags are available).
    mode:
      - "true_exog": use actual future exog (from Z rows t0+1...t0+H)
      - "hold_last": keep exog fixed at its value at t0
    Returns: pandas.Series of length H indexed by the next H timestamps.
    """
    # Locate integer position in Z
    if t0 not in Z_idx:
        raise ValueError("t0 must be an index present in Z (after lag drop).")
    pos = Z_idx.get_loc(t0)

    # Need AR_ORDER past residuals at t0: y_{t0}, y_{t0-1}, ...
    # Z['y'] is y_t; for prediction at t0+1 we need [y_{t0}, y_{t0-1}]
    y_hist = [Z['y'].iloc[pos - i] for i in range(1, AR_ORDER + 1)]  # [y_{t0-1}, y_{t0-2}]
    # But we want order [y_{t-1}, y_{t-2}], already correct

    # Exog handling
    if mode == "true_exog":
        # we will use Z exog rows at positions pos+1 ... pos+H
        if pos + H >= len(Z_idx):
            raise ValueError("Not enough future samples in Z for the requested horizon.")
    elif mode == "hold_last":
        exog_hold = X_ex_all_s[pos].copy()
    else:
        raise ValueError("mode must be 'true_exog' or 'hold_last'.")

    preds = []
    times = []
    for k in range(1, H + 1):
        # pick standardized exog for this step
        if mode == "true_exog":
            exog_s_row = X_ex_all_s[pos + k]
        else:  # hold_last
            exog_s_row = exog_hold

        # predict next residual using current history
        y_next = arx_step(y_hist, exog_s_row, beta)

        preds.append(y_next)
        times.append(Z_idx[pos + k])  # forecast corresponds to y_{t0+k}

        # update AR history: prepend new prediction and drop the oldest
        y_hist = [y_next] + y_hist[:AR_ORDER - 1]

    return pd.Series(preds, index=pd.Index(times, name=Z['y'].index.name), name='residual_hat_h')

# ---------- Example: single 30 s forecast on test set ----------
H = 30  # seconds
# choose a start time inside the test window with enough future samples
t0 = y_te.index[int(0.5 * len(y_te))]  # mid-test as an example

res_30s = forecast_horizon_OL(t0, H=H, mode="true_exog")

# Reconstruct dyn resistance for the horizon
meta_30s = df.loc[res_30s.index, 'Tot_Resistance_meta']
dyn_30s  = meta_30s + res_30s
plant_30s = df.loc[res_30s.index, 'Tot_Resistance_mOhm']

# Quick plot
plt.figure(figsize=(12,5))
plt.plot(plant_30s.index, plant_30s.values, label='Plant', lw=2)
plt.plot(meta_30s.index,  meta_30s.values,  label='Metamodel', alpha=0.7)
plt.plot(dyn_30s.index,   dyn_30s.values,   label='Dyn forecast (+30s)', alpha=0.9)
plt.title(f"Open-loop ARX forecast: {H} s horizon starting after {t0}")
plt.xlabel("Time"); plt.ylabel("Total resistance [mΩ]")
# plt.legend(); plt.tight_layout(); plt.show()

# ---------- Offline evaluation: RMSE/R² at exactly 30 s ahead across the test ----------
def horizon_metrics_fast(H=30, mode="true_exog", stride=5, max_cases=None):
    """
    Fast RMSE/R² for H-second-ahead open-loop forecasts over the test window.
    Works for any AR_ORDER >= 1 (vectorized over many start times).

    mode:
      - "true_exog": uses actual future exog (offline evaluation)
      - "hold_last": holds exog fixed at start (no exog forecast needed)

    stride: evaluate every 'stride' seconds to reduce workload.
    max_cases: cap number of rollouts (optional).
    """
    assert mode in ("true_exog", "hold_last")

    # Ensure we have standardized exog aligned to Z
    # (If you didn't create X_ex_all_s earlier, do it here:)
    # X_ex_all_s = scaler.transform(Z[exog_cols].values)
    Xex = X_ex_all_s

    # Map timestamps to Z positions
    Z_pos = pd.Series(np.arange(len(Z.index)), index=Z.index)

    # Candidate starts: test timestamps present in Z
    starts = y_te.index
    starts = starts[starts.isin(Z_pos.index)]
    pos_all = Z_pos.loc[starts].to_numpy()

    # Validity filters:
    # 1) Need AR_ORDER past outputs available at start:
    pos_all = pos_all[pos_all - (AR_ORDER - 1) >= 0]
    # 2) Need H future samples available:
    pos_all = pos_all[pos_all + H < len(Z.index)]

    # Apply stride and max_cases
    if stride > 1:
        pos_all = pos_all[::stride]
    if max_cases is not None:
        pos_all = pos_all[:max_cases]

    if pos_all.size == 0:
        raise ValueError("No valid start positions for the requested horizon.")

    Nstarts = pos_all.shape[0]

    # Build initial AR history for all starts: [y_{t0}, y_{t0-1}, ..., y_{t0-(p-1)}]
    y_full = Z["y"].to_numpy(dtype=np.float64)
    p = AR_ORDER
    y_hist = np.empty((Nstarts, p), dtype=np.float64)
    for i in range(p):
        # i=0 -> y_{t0}, i=1 -> y_{t0-1}, ..., i=p-1 -> y_{t0-(p-1)}
        y_hist[:, i] = y_full[pos_all - i]

    # Coefficients
    beta_np = beta.detach().cpu().numpy().reshape(-1)  # [1 + p + n_exog]
    b0 = beta_np[0]
    b_ar = beta_np[1:1 + p]
    b_ex = beta_np[1 + p:]
    n_exog = Xex.shape[1]
    if b_ex.shape[0] != n_exog:
        raise ValueError("Beta/exog dimension mismatch. Did you change exog columns without refitting?")

    # Cache exog if holding constant
    if mode == "hold_last":
        exog_fixed = Xex[pos_all, :].copy()

    mu_ar   = ar_scaler.mean_.astype(np.float64)   # shape [p]
    std_ar  = ar_scaler.scale_.astype(np.float64)  # shape [p]

    # Roll H steps
    for k in range(1, H + 1):
        # standardized exog for this step (already correct)
        ex_k = Xex[pos_all + k, :] if mode == "true_exog" else exog_fixed

        # ---> standardize the AR vector to match training! <---
        # y_hist columns are [y_{t-1}, y_{t-2}, ..., y_{t-p}]
        y_hist_std = (y_hist - mu_ar[np.newaxis, :]) / std_ar[np.newaxis, :]

        # vectorized next-step prediction in the *standardized* design space
        ar_part = (y_hist_std * b_ar[np.newaxis, :]).sum(axis=1)  # β_AR · zscore(y_lags)
        ex_part = ex_k @ b_ex
        y_next = b0 + ar_part + ex_part

        # shift AR history in RAW space for next step
        if p > 1:
            y_hist[:, 1:] = y_hist[:, :-1]
        y_hist[:, 0] = y_next

    # After H steps, y_hist[:,0] == y_{t0+H} predictions
    y_pred_res_H = y_hist[:, 0]

    # Build total resistance predictions and metrics
    idx_H = Z.index[pos_all + H]
    meta_H = df.loc[idx_H, "Tot_Resistance_meta"].to_numpy(dtype=np.float64)
    plant_H = df.loc[idx_H, "Tot_Resistance_mOhm"].to_numpy(dtype=np.float64)
    y_pred_tot_H = meta_H + y_pred_res_H

    err = plant_H - y_pred_tot_H
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.sum((plant_H - plant_H.mean())**2)
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else np.nan

    return rmse, r2, int(Nstarts), pd.Index(idx_H, name="timestamp")

rmse30, r230, Ncases, used_idx = horizon_metrics_fast(H=30, mode="true_exog", stride=5)
print(f"[FAST H=30s] Test RMSE={rmse30:.4f} mΩ, R²={r230:.3f} over {Ncases} rollouts (stride=5, true exog).")


def horizon_curve(H_list=(1,5,10,15,20,25,30), mode="true_exog", stride=5):
    out = []
    for H in H_list:
        rmse, r2, N, _ = horizon_metrics_fast(H=H, mode=mode, stride=stride)
        out.append((H, rmse, r2, N))
    print("H, RMSE[mΩ], R2, Ncases")
    for H, rmse, r2, N in out:
        print(f"{H:2d}, {rmse:.4f}, {r2:.3f}, {N}")
    return out

_ = horizon_curve()


# ---------- Plot ----------
df_test = df.loc[mask_te].copy()
plt.figure(figsize=(12,5))
plt.plot(df_test.index, df_test['Tot_Resistance_mOhm'], label='Plant (measured)', lw=2)
plt.plot(df_test.index, df_test['Tot_Resistance_meta'], label='Metamodel (steady-state)', alpha=0.7)
plt.plot(df_test.index, df_test['Tot_Resistance_dyn'], label='Metamodel + ARX (Torch, GPU)', alpha=0.9)
plt.xlabel("Time")
plt.ylabel("Total resistance [mΩ]")
plt.title("ARX(2) via Torch Ridge — Test data (out-of-sample)")
plt.legend()
plt.tight_layout()
plt.show()

import os, joblib, json

# 1) Save coefficients and scalers (joblib handles NumPy and sklearn objects)
joblib.dump({
    "beta": beta.detach().cpu().numpy(),     # [1 + p + n_exog, 1]
    "AR_ORDER": AR_ORDER,
    "EXOG_LAG": EXOG_LAG,
    "exog_cols": exog_cols,
    "scaler": scaler,
    "ar_scaler": ar_scaler
}, "metamodel/anew_arx/models.joblib")

# 2) Optional human-readable summary
spec = {
    "type": "ARX",
    "mode": "residual",
    "AR_ORDER": AR_ORDER,
    "EXOG_LAG": EXOG_LAG,
    "exog_columns": exog_cols,
    "target": "residual_filt"
}
with open("metamodel/anew_arx/models/arx_spec.json", "w") as f:
    json.dump(spec, f, indent=2)

print("✅ Saved ARX model to models/arx_model.joblib and spec.json")
