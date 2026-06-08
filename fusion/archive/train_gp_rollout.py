"""
train_gp_rollout.py
-------------------
Train the "rollout" SVGP correction for the SAF codebase.

The VARX from Fusion_multielectrode (varx_pi_retrained.joblib, step9 lower-
triangular structure) is used to generate training residuals — same model that
the Fusion_multielectrode simulation uses.  However the GP features are built
using the SAF simulator's naming convention (El{i}_y_filt_lag, kA_filt_lag,
dpos_mps_filt_lag, etc.) so the bundle can be used directly in run_closed_loop.py
without any feature translation.

Key design decisions (same as V11):
  - H=1000 rollout windows — GP sees long-horizon cumulative errors
  - No step_in_window — avoids rollout-depth bias in sigma
  - No y_sim_sq — not needed
  - Debiasing — zero mean over training set (essential for stable simulation)
  - Lag-feedback separation — VARX lags updated with VARX predictions

Output:  fusion/models/gp_el{1,2,3}_rollout.pt

Run from SAF_data_to_control/:
  python fusion/train_gp_rollout.py
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, TensorDataset

_HERE         = Path(__file__).resolve().parent          # fusion/
_PROJECT_ROOT = _HERE.parent                             # SAF_data_to_control/
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from fusion.archive.train_models.svgp_model import SVGPModel, KernelSpec  # noqa

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

PI_DATA_PATH  = _HERE / "archive" / "data" / "raw" / "PI-data.csv"
VARX_PATH     = _HERE / "models" / "varx_pi_retrained.joblib"
MODELS_OUT    = _HERE / "models"

PI_ROW_START  = 276_480     # middle 80% of PI historian
PI_ROW_END    = 2_488_320

ELECTRODES    = [1, 2, 3]
NUM_INDUCING  = 512
EPOCHS        = 60
LR            = 3e-3
BATCH_SIZE    = 2048
PATIENCE      = 10
VAL_FRACTION  = 0.20
RANDOM_STATE  = 42
H             = 1000
STRIDE        = 1000

DETREND_WINDOW = 1800
FS, FC = 1.0, 0.1

PI_GI  = {1: r"\\ZMUCPI01\V1903T830EG104.1 GI",
           2: r"\\ZMUCPI01\V1903T830EG204.1 GI",
           3: r"\\ZMUCPI01\V1903T830EG304.1 GI"}
PI_KA  = {1: "\\\\ZMUCPI01\\V1903T830EU100.1 Strøm",
           2: "\\\\ZMUCPI01\\V1903T830EU200.1 Strøm",
           3: "\\\\ZMUCPI01\\V1903T830EU300.1 Strøm"}
PI_RES = {1: r"\\ZMUCPI01\V1903T830EU100.1 Resistans",
           2: r"\\ZMUCPI01\V1903T830EU200.1 Resistans",
           3: r"\\ZMUCPI01\V1903T830EU300.1 Resistans"}
PI_UL  = {1: r"\\ZMUCPI01\V1903T870EE976.1 UL1N",
           2: r"\\ZMUCPI01\V1903T870EE976.1 UL2N",
           3: r"\\ZMUCPI01\V1903T870EE976.1 UL3N"}
PI_TC  = {1: r"\\ZMUCPI01\V1903T830EU102.1 TCA",
           2: r"\\ZMUCPI01\V1903T830EU102.1 TCB",
           3: r"\\ZMUCPI01\V1903T830EU102.1 TCC"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_rollout] Device: {DEVICE}")


# =============================================================================
# Helpers
# =============================================================================

def _lp(arr, order=4):
    b, a = butter(order, FC / (FS / 2), btype="low", analog=False)
    s = pd.Series(arr).interpolate(limit=5).ffill().bfill().values
    return filtfilt(b, a, s, method="gust")

def _lag(arr, k):
    return np.concatenate([np.full(k, arr[0]), arr[:-k]])

def _rolling_std(arr, w=30):
    return pd.Series(arr).rolling(w, min_periods=1).std().fillna(0.0).values

def _rtilde(r_filt):
    trend = pd.Series(r_filt).rolling(DETREND_WINDOW, min_periods=1).median().values
    return (r_filt - trend).astype(np.float64)


# =============================================================================
# Load and preprocess PI data
# -- VARX state columns (step9 naming) for rollout state tracking
# -- SAF columns (y_filt, kA_filt, dpos_mps_filt) for GP features
# =============================================================================

def load_pi_data() -> pd.DataFrame:
    all_cols = list({c for sub in [
        list(PI_GI.values()), list(PI_KA.values()),
        list(PI_RES.values()), list(PI_UL.values()), list(PI_TC.values())
    ] for c in sub})

    print(f"  Loading PI rows {PI_ROW_START:,} to {PI_ROW_END:,} ...")
    raw = pd.read_csv(PI_DATA_PATH,
                      skiprows=range(1, PI_ROW_START),
                      nrows=PI_ROW_END - PI_ROW_START,
                      usecols=all_cols, low_memory=False).reset_index(drop=True)
    raw = raw.apply(pd.to_numeric, errors="coerce").interpolate(limit=10).ffill().bfill()

    ul = {i: raw[PI_UL[i]].values for i in (1, 2, 3)}
    rms_v = np.sqrt((ul[1]**2 + ul[2]**2 + ul[3]**2) / 3.0)
    v_f   = np.clip(_lp(rms_v), 0.0, None)
    tca   = pd.Series(raw[PI_TC[1]].values).ffill().bfill().values.astype(float)
    tcb   = pd.Series(raw[PI_TC[2]].values).ffill().bfill().values.astype(float)
    tcc   = pd.Series(raw[PI_TC[3]].values).ffill().bfill().values.astype(float)

    sigs: dict[str, np.ndarray] = {
        "RMS_V_transformer_filt_lag1": _lag(v_f, 1),
        "TCA": tca, "TCB": tcb, "TCC": tcc,
        "TCA_diff": np.concatenate([[0.0], np.diff(tca)]),
    }

    for i in (1, 2, 3):
        pos    = raw[PI_GI[i]].values / 100.0
        dpos   = np.concatenate([[0.0], np.diff(pos)])
        dpos_f = _lp(dpos)
        y_f    = _lp(raw[PI_RES[i]].values, order=2)
        ka_f   = np.clip(_lp(raw[PI_KA[i]].values), 0.0, None)
        z_i    = ul[i] / (ka_f + 1e-6)
        reac_f = np.clip(_lp(np.clip(np.sqrt(np.maximum(z_i**2 - y_f**2, 0.0)),
                                      0.0, 3.0 * 5.0)), 0.0, 3.0)
        r_tilde = _rtilde(y_f)

        # VARX state columns (step9 naming — for running the VARX rollout)
        for k in range(1, 11):
            sigs[f"El{i}_Resistance_mOhm_filt_lag{k}"] = _lag(r_tilde, k)
            sigs[f"El{i}_pos_m_filt_lag{k}"]           = _lag(pos, k)
            sigs[f"kA{i}_lag{k}"]                      = _lag(ka_f, k)
        for j in range(i + 1, 4):
            for k in range(1, 11):
                sigs[f"El{i}_Resistance_mOhm_filt_lag{k}->El{j}"] = _lag(r_tilde, k)

        # SAF GP feature columns (used to build the GP training features)
        for k in range(1, 6):
            sigs[f"El{i}_dpos_mps_filt_lag{k}"] = _lag(dpos_f, k)
        for k in range(1, 3):
            sigs[f"El{i}_kA_filt_lag{k}"]       = _lag(ka_f, k)
            sigs[f"El{i}_CalcReac_filt_lag{k}"] = _lag(reac_f, k)
        sigs[f"El{i}_y_filt_lag1"]             = _lag(y_f, 1)
        sigs[f"El{i}_rolling_std_R_30s"]        = _rolling_std(y_f, 30)
        sigs[f"El{i}_rolling_std_CalcReac_30s"] = _rolling_std(reac_f, 30)

        # Targets
        sigs[f"El{i}_R_true"]   = y_f
        sigs[f"El{i}_R_tilde"]  = r_tilde

    df = pd.DataFrame(sigs).iloc[10:].reset_index(drop=True)
    print(f"  {len(df):,} usable rows")
    return df


# =============================================================================
# GP feature columns — SAF naming, no step_in_window, no y_sim_sq
# =============================================================================

def gp_feature_cols(el: int) -> list[str]:
    """
    Feature set matching V8 GP minus step_in_window and y_sim_sq.
    All names are directly available in the SAF simulator state (sim._row).
    """
    other = [j for j in (1, 2, 3) if j != el]
    cols = ["y_sim"]                           # filled with VARX prediction at training time

    # Own electrode
    for k in range(1, 6):
        cols.append(f"El{el}_dpos_mps_filt_lag{k}")
    cols += [f"El{el}_kA_filt_lag1", f"El{el}_kA_filt_lag2"]
    cols += [f"El{el}_CalcReac_filt_lag1", f"El{el}_CalcReac_filt_lag2"]
    cols.append(f"TC{'ABC'[el-1]}")
    cols.append("RMS_V_transformer_filt_lag1")

    # Cross-electrode
    for j in other:
        for k in range(1, 4):
            cols.append(f"El{j}_dpos_mps_filt_lag{k}")
        cols.append(f"El{j}_y_filt_lag1")
        cols += [f"El{j}_kA_filt_lag1", f"El{j}_kA_filt_lag2"]
        cols += [f"El{j}_CalcReac_filt_lag1", f"El{j}_CalcReac_filt_lag2"]
        cols.append(f"TC{'ABC'[j-1]}")

    # Rolling variability
    for j in (1, 2, 3):
        cols.append(f"El{j}_rolling_std_CalcReac_30s")
    for j in (1, 2, 3):
        cols.append(f"El{j}_rolling_std_R_30s")

    cols.append(f"El{el}_R_imbalance")
    cols.append("TCA_diff")
    return cols


# =============================================================================
# VARX prediction (step9 lower-triangular)
# =============================================================================

def varx_step(bundle: dict, state: dict) -> np.ndarray:
    xcols_per_eq = bundle["X_cols_per_eq"]
    models       = bundle["models"]
    x_scalers    = bundle.get("X_scalers")
    preds        = np.zeros(3, dtype=float)
    for eq_i, (xcols_eq, model) in enumerate(zip(xcols_per_eq, models)):
        x = np.array([state.get(c, 0.0) for c in xcols_eq], dtype=float).reshape(1, -1)
        if x_scalers:
            x = x_scalers[eq_i].transform(x)
        preds[eq_i] = float(model.predict(x)[0])
    return preds


# =============================================================================
# Build H=1000 rollout dataset
# =============================================================================

def build_rollout_dataset(df: pd.DataFrame, bundle: dict, el: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    xcols_flat = bundle.get("X_cols_flat", bundle.get("X_cols", []))
    feat_cols  = gp_feature_cols(el)
    n = len(df)
    y_real = {i: df[f"El{i}_R_tilde"].values for i in (1, 2, 3)}

    all_starts  = list(range(0, n - H - 1, STRIDE))
    total_win   = len(all_starts)
    print_every = max(1, total_win // 10)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    t0_build = time.time()

    for wid, t0 in enumerate(all_starts):
        if wid % print_every == 0:
            print(f"  El{el} window {wid:>5}/{total_win}  "
                  f"elapsed={( time.time()-t0_build)/60:.1f}min", flush=True)

        state = {c: float(df[c].iloc[t0]) if c in df.columns else 0.0
                 for c in xcols_flat}

        # rolling std buffers seeded from real data
        r_buf    = {i: list(df[f"El{i}_R_true"].values[max(0,t0-29):t0+1]) for i in (1,2,3)}
        reac_buf = {i: list(df[f"El{i}_CalcReac_filt_lag1"].values[max(0,t0-29):t0+1]) for i in (1,2,3)}

        X_win = np.empty((H, len(feat_cols)), dtype=np.float32)
        y_win = np.empty(H, dtype=np.float32)
        n_rows = 0

        for step in range(H):
            t = t0 + step
            if t + 1 >= n:
                break

            r_varx   = varx_step(bundle, state)
            y_sim_el = float(r_varx[el - 1])

            # R imbalance in R_tilde space
            r_mean = np.mean(r_varx)

            # Build GP feature vector using SAF naming
            feats: dict[str, float] = {}
            feats["y_sim"] = y_sim_el
            feats[f"El{el}_R_imbalance"] = y_sim_el - r_mean
            feats["TCA_diff"] = state.get("TCA_diff", 0.0)
            feats["RMS_V_transformer_filt_lag1"] = state.get("RMS_V_transformer_filt_lag1", 0.0)
            for j in (1, 2, 3):
                feats[f"TC{'ABC'[j-1]}"] = state.get(f"TC{'ABC'[j-1]}", 0.0)
                feats[f"El{j}_rolling_std_R_30s"]        = float(np.std(r_buf[j]))    if len(r_buf[j])>1    else 0.0
                feats[f"El{j}_rolling_std_CalcReac_30s"] = float(np.std(reac_buf[j])) if len(reac_buf[j])>1 else 0.0
                feats[f"El{j}_y_filt_lag1"] = float(df[f"El{j}_y_filt_lag1"].values[min(t,n-1)])
                for k in range(1, 6):
                    feats[f"El{j}_dpos_mps_filt_lag{k}"] = float(df[f"El{j}_dpos_mps_filt_lag{k}"].values[min(t,n-1)])
                for k in range(1, 3):
                    feats[f"El{j}_kA_filt_lag{k}"]       = float(df[f"El{j}_kA_filt_lag{k}"].values[min(t,n-1)])
                    feats[f"El{j}_CalcReac_filt_lag{k}"] = float(df[f"El{j}_CalcReac_filt_lag{k}"].values[min(t,n-1)])

            X_win[n_rows] = np.array([feats.get(f, 0.0) for f in feat_cols], dtype=np.float32)
            y_win[n_rows] = float(y_real[el][t + 1]) - y_sim_el
            n_rows += 1

            # Advance VARX state (lag-feedback separation)
            for i in (1, 2, 3):
                rp = float(r_varx[i - 1]); t_next = min(t + 1, n - 1)
                for k in range(10, 1, -1):
                    state[f"El{i}_Resistance_mOhm_filt_lag{k}"] = \
                        state.get(f"El{i}_Resistance_mOhm_filt_lag{k-1}", 0.0)
                    for j in range(i+1, 4):
                        key = f"El{i}_Resistance_mOhm_filt_lag{k}->El{j}"
                        if key in state:
                            state[key] = state.get(f"El{i}_Resistance_mOhm_filt_lag{k-1}->El{j}", 0.0)
                state[f"El{i}_Resistance_mOhm_filt_lag1"] = rp
                for j in range(i+1, 4):
                    k1 = f"El{i}_Resistance_mOhm_filt_lag1->El{j}"
                    if k1 in state: state[k1] = rp
                pos_new = float(df[f"El{i}_pos_m_filt_lag1"].values[t_next])
                ka_new  = float(df[f"kA{i}_lag1"].values[t_next])
                for k in range(10, 1, -1):
                    state[f"El{i}_pos_m_filt_lag{k}"] = state.get(f"El{i}_pos_m_filt_lag{k-1}", 0.0)
                    state[f"kA{i}_lag{k}"]            = state.get(f"kA{i}_lag{k-1}", 0.0)
                state[f"El{i}_pos_m_filt_lag1"] = pos_new
                state[f"kA{i}_lag1"]            = ka_new
                r_buf[i].append(rp); reac_buf[i].append(float(df[f"El{i}_CalcReac_filt_lag1"].values[t_next]))
                if len(r_buf[i])>30: r_buf[i] = r_buf[i][-30:]
                if len(reac_buf[i])>30: reac_buf[i] = reac_buf[i][-30:]

        if n_rows > 0:
            X_list.append(X_win[:n_rows].copy())
            y_list.append(y_win[:n_rows].copy())

    X_out = np.concatenate(X_list); y_out = np.concatenate(y_list)
    print(f"  El{el}: {len(y_out):,} samples  delta [{y_out.min():.4f}, {y_out.max():.4f}] mOhm")
    return X_out, y_out


# =============================================================================
# Train SVGP
# =============================================================================

def train_svgp(X_all: np.ndarray, y_all: np.ndarray, el: int,
               feat_cols: list[str]) -> dict:
    n = len(X_all)
    n_val = int(n * VAL_FRACTION)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(n)
    X_tr, y_tr = X_all[idx[n_val:]], y_all[idx[n_val:]]
    X_va, y_va = X_all[idx[:n_val]], y_all[idx[:n_val]]

    debias = float(y_tr.mean())
    y_tr_d = y_tr - debias; y_va_d = y_va - debias
    print(f"  El{el}: {len(X_tr):,} train / {len(X_va):,} val  debias={debias:.6f} mOhm")

    x_mean = X_tr.mean(0).astype(np.float32)
    x_std  = X_tr.std(0).astype(np.float32); x_std[x_std < 1e-8] = 1.0
    y_mean = float(y_tr_d.mean()); y_std = float(y_tr_d.std()) or 1.0

    X_tr_s = (X_tr.astype(np.float32) - x_mean) / x_std
    X_va_s = (X_va.astype(np.float32) - x_mean) / x_std
    y_tr_s = (y_tr_d.astype(np.float32) - y_mean) / y_std
    y_va_s = (y_va_d.astype(np.float32) - y_mean) / y_std

    Xt = torch.tensor(X_tr_s).to(DEVICE); yt = torch.tensor(y_tr_s).to(DEVICE)
    Xv = torch.tensor(X_va_s).to(DEVICE); yv = torch.tensor(y_va_s).to(DEVICE)

    Z_init = torch.tensor(X_tr_s[rng.choice(len(X_tr_s), NUM_INDUCING, replace=False)])
    spec   = KernelSpec(name="matern52", ard=True)
    model  = SVGPModel(Z_init, spec).to(DEVICE)
    lik    = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    opt    = torch.optim.Adam([{"params": model.parameters()}, {"params": lik.parameters()}], lr=LR)
    mll    = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(Xt))
    ldr    = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    best_val = float("inf"); best_m = best_l = None; patience_ctr = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); lik.train()
        ep_loss = 0.0
        for xb, yb in ldr:
            opt.zero_grad(); loss = -mll(lik(model(xb)), yb); loss.backward()
            opt.step(); ep_loss += float(loss.item())
        ep_loss /= len(ldr)
        if epoch % 5 == 0 or epoch == 1:
            model.eval(); lik.eval()
            with torch.no_grad(): vl = float(-mll(lik(model(Xv)), yv).item())
            print(f"  El{el} epoch {epoch:3d}/{EPOCHS}  train={ep_loss:.4f}  val={vl:.4f}")
            if vl < best_val:
                best_val = vl
                best_m = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_l = {k: v.cpu().clone() for k, v in lik.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  El{el} early stop at epoch {epoch}"); break

    model.load_state_dict(best_m); lik.load_state_dict(best_l)
    model.eval(); lik.eval()
    sv = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(Xt), batch_size=4096):
            sv.append(lik(model(xb)).variance.sqrt().cpu().numpy())
    sigma_ref = float(np.percentile(np.concatenate(sv), 90)) * y_std

    return {
        "model": model.cpu(), "likelihood": lik.cpu(),
        "feature_names": feat_cols,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": np.array([y_mean + debias], dtype=np.float32),
        "y_std":  np.array([y_std], dtype=np.float32),
        "sigma_ref": sigma_ref, "debias_offset": debias,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== Rollout SVGP training ===")
    print("VARX: step9 (varx_pi_retrained)  |  GP features: SAF naming  |  H=1000\n")

    df     = load_pi_data()
    bundle = joblib.load(VARX_PATH)
    print(f"VARX loaded: {bundle['model_name']}\n")

    for el in ELECTRODES:
        print(f"\n{'='*50}\n  Electrode {el}\n{'='*50}")
        feat_cols = gp_feature_cols(el)
        print(f"  GP features: {len(feat_cols)}")
        print(f"  First: {feat_cols[:3]}  Last: {feat_cols[-3:]}")

        X, y = build_rollout_dataset(df, bundle, el)
        gp   = train_svgp(X, y, el, feat_cols)

        out = MODELS_OUT / f"gp_el{el}_rollout.pt"
        torch.save(gp, out)
        print(f"\n  Saved {out.name}  sigma_ref={gp['sigma_ref']:.5f}  "
              f"features={len(feat_cols)}  debias={gp['debias_offset']:.6f}")

    print("\n\nDone. Models in:", MODELS_OUT)
