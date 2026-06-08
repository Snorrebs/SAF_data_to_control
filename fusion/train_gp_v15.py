"""
train_gp_v15.py
Delta-R ARX + SVGP correction model (version 15).

Per-electrode Ridge regression on dR = R(t) - R(t-1) with 5 own-R lags,
5 own-dpos lags, and cross-electrode dpos lags. No cross-R lags.
DeltaARXWrapper holds kA/reac/V at lag-1 values between steps.

Outputs: fusion/models/arx_joint_v15.joblib and gp_el{1,2,3}_v15.pt
Run with: python fusion/train_gp_v15.py
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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.archive.train_models.svgp_model import SVGPModel, KernelSpec
from fusion.training.delta_arx import DeltaARXWrapper

warnings.filterwarnings("ignore")

# CONFIGURATION
PI_DATA_PATH = (Path(r"C:\Users\wighu\OneDrive\Skrivebord\SAPPHIRE\Master-SAFECI\KODE\SAF_data_to_control")
                / "fusion" / "archive" / "data" / "raw" / "PI-data.csv")

ARX_MODEL   = _HERE / "models" / "arx_joint_v15.joblib"
MODELS_OUT  = _HERE / "models"

PI_ROW_START = 276_480
PI_ROW_END   = 2_488_320
ELECTRODES   = [1, 2, 3]

NUM_INDUCING  = 512
EPOCHS        = 60
LR            = 3e-3
BATCH_SIZE    = 4096
PATIENCE      = 10
VAL_FRACTION  = 0.20
RANDOM_STATE  = 42

ROLLOUT_H      = 1000
ROLLOUT_STRIDE = 300

SEM_MIN_DPOS   = 0.005
SEM_FLOOR      = 0.10
SEM_EPS        = 1e-6
REAC_CLIP_MOHM = 3.0
FS, FC         = 1.0, 0.1

# Number of own-electrode lags built in load_pi_data and used in the ARX.
# V14 used 3. Matching the reference model (step9_n10) more closely.
N_R_LAGS    = 5
N_DPOS_LAGS = 5

PI_GI  = {1: r"\\ZMUCPI01\V1903T830EG104.1 GI",
           2: r"\\ZMUCPI01\V1903T830EG204.1 GI",
           3: r"\\ZMUCPI01\V1903T830EG304.1 GI"}
PI_KA  = {1: "\\\\ZMUCPI01\\V1903T830EU100.1 Strøm",
           2: "\\\\ZMUCPI01\\V1903T830EU200.1 Strøm",
           3: "\\\\ZMUCPI01\\V1903T830EU300.1 Strøm"}
PI_RES = {1: r"\\ZMUCPI01\V1903T830EU100.1 Resistans",
           2: r"\\ZMUCPI01\V1903T830EU200.1 Resistans",
           3: r"\\ZMUCPI01\V1903T830EU300.1 Resistans"}
PI_TC  = {1: r"\\ZMUCPI01\V1903T830EU102.1 TCA",
           2: r"\\ZMUCPI01\V1903T830EU102.1 TCB",
           3: r"\\ZMUCPI01\V1903T830EU102.1 TCC"}
PI_UL  = {1: r"\\ZMUCPI01\V1903T870EE976.1 UL1N",
           2: r"\\ZMUCPI01\V1903T870EE976.1 UL2N",
           3: r"\\ZMUCPI01\V1903T870EE976.1 UL3N"}

_Y_COLS_JOINT = (
    [f"El{i}_R_target"    for i in (1, 2, 3)]
    + [f"El{i}_kA_target"   for i in (1, 2, 3)]
    + [f"El{i}_Reac_target" for i in (1, 2, 3)]
    + ["V_target"]
)
_Y_INDEX = {
    "R":    {1: 0, 2: 1, 3: 2},
    "kA":   {1: 3, 2: 4, 3: 5},
    "reac": {1: 6, 2: 7, 3: 8},
    "v":    9,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_v15] Device: {DEVICE}")


def _lp(arr: np.ndarray, order: int = 4) -> np.ndarray:
    b, a = butter(order, FC / (FS / 2), btype="low", analog=False)
    return filtfilt(b, a, pd.Series(arr).interpolate(limit=5).ffill().bfill().values,
                    method="gust")

def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k, arr[0]), arr[:-k]])

def _rolling_std(arr: np.ndarray, w: int = 30) -> np.ndarray:
    return pd.Series(arr).rolling(w, min_periods=1).std().fillna(0).values


def load_pi_data() -> pd.DataFrame:
    all_pi_cols = (list(PI_GI.values()) + list(PI_KA.values())
                   + list(PI_UL.values()) + list(PI_RES.values())
                   + list(PI_TC.values()))
    seen = set()
    all_pi_cols = [c for c in all_pi_cols if not (c in seen or seen.add(c))]

    print(f"  Loading rows {PI_ROW_START:,} to {PI_ROW_END:,} ...")
    raw = pd.read_csv(PI_DATA_PATH,
                      skiprows=range(1, PI_ROW_START),
                      nrows=PI_ROW_END - PI_ROW_START,
                      usecols=all_pi_cols, low_memory=False).reset_index(drop=True)
    raw = raw.apply(pd.to_numeric, errors="coerce").interpolate(limit=10).ffill().bfill()
    print(f"  Loaded {len(raw):,} rows")

    ul    = {i: raw[PI_UL[i]].values for i in (1, 2, 3)}
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
        reac_f = np.clip(
            _lp(np.clip(np.sqrt(np.maximum(z_i**2 - y_f**2, 0.0)), 0.0, REAC_CLIP_MOHM * 5.0)),
            0.0, REAC_CLIP_MOHM)
        p = f"El{i}"
        sigs[f"{p}_dpos_raw"]   = dpos
        sigs[f"{p}_pos_m_lag1"] = _lag(pos, 1)
        # Build N_R_LAGS and N_DPOS_LAGS lags (more history than V14's 3)
        for k in range(1, max(N_R_LAGS, N_DPOS_LAGS) + 1):
            if k <= N_DPOS_LAGS:
                sigs[f"{p}_dpos_mps_filt_lag{k}"] = _lag(dpos_f, k)
            if k <= N_R_LAGS:
                sigs[f"{p}_y_filt_lag{k}"]        = _lag(y_f,    k)
            if k <= 3:
                sigs[f"{p}_kA_filt_lag{k}"]       = _lag(ka_f,   k)
                sigs[f"{p}_CalcReac_filt_lag{k}"] = _lag(reac_f, k)
        sigs[f"{p}_R_true"]                   = y_f
        sigs[f"{p}_rolling_std_R_30s"]        = _rolling_std(y_f, 30)
        sigs[f"{p}_rolling_std_CalcReac_30s"] = _rolling_std(reac_f, 30)

    df = pd.DataFrame(sigs).iloc[max(N_R_LAGS, N_DPOS_LAGS):].reset_index(drop=True)
    print(f"  {len(df):,} usable rows after lag warm-up")
    return df


def arx_xcols_v15(el: int) -> list[str]:
    """Per-electrode ARX features for V15.

    Own-electrode: R lags 1-5, dpos lags 1-5, kA lags 1-2, CalcReac lags 1-2, pos lag1.
    Cross-electrode: dpos lags 1-2 only (physical coupling without R coupling).
    Global: TCA/TCB/TCC, RMS_V.
    """
    other = [j for j in (1, 2, 3) if j != el]
    cols  = []
    for k in range(1, N_R_LAGS + 1):
        cols.append(f"El{el}_y_filt_lag{k}")
    for k in range(1, N_DPOS_LAGS + 1):
        cols.append(f"El{el}_dpos_mps_filt_lag{k}")
    cols += [f"El{el}_kA_filt_lag1", f"El{el}_kA_filt_lag2"]
    cols += [f"El{el}_CalcReac_filt_lag1", f"El{el}_CalcReac_filt_lag2"]
    cols.append(f"El{el}_pos_m_lag1")
    for j in other:
        cols += [f"El{j}_dpos_mps_filt_lag1", f"El{j}_dpos_mps_filt_lag2"]
    cols += ["TCA", "TCB", "TCC", "RMS_V_transformer_filt_lag1"]
    return cols


def train_arx_v15(df: pd.DataFrame) -> dict:
    """Train per-electrode Ridge on dR = R(t) - R(t-1).

    Infrastructure fixes vs V14:
    - Identity Y_scaler: inverse_transform(x) = x (no scaling corruption).
    - Explicit clip_overrides: kA clips [0, 400], R clips [0.5, 2.5].
    - DeltaARXWrapper.predict() holds kA/reac/V at lag-1 (not zero).
    All three were bugs in V14 that caused divergence.
    """
    xcols_per_el: dict[int, list[str]] = {el: arx_xcols_v15(el) for el in (1, 2, 3)}
    joint_xcols   = sorted(set(c for cols in xcols_per_el.values() for c in cols))

    df_arx = df.copy()
    for i in (1, 2, 3):
        df_arx[f"El{i}_dR_target"] = df_arx[f"El{i}_R_true"] - df_arx[f"El{i}_y_filt_lag1"]
    df_arx = df_arx.dropna()

    n     = len(df_arx)
    split = int(n * 0.80)
    print(f"  ARX train={split:,}  holdout={n-split:,}")

    models:    dict[int, RidgeCV]        = {}
    x_scalers: dict[int, StandardScaler] = {}

    for el in (1, 2, 3):
        xcols  = xcols_per_el[el]
        X_tr   = df_arx[xcols].iloc[:split].to_numpy(dtype=np.float64)
        y_tr   = df_arx[f"El{el}_dR_target"].iloc[:split].to_numpy(dtype=np.float64)
        X_te   = df_arx[xcols].iloc[split:].to_numpy(dtype=np.float64)
        y_te   = df_arx[f"El{el}_dR_target"].iloc[split:].to_numpy(dtype=np.float64)
        R_lag1_te = df_arx[f"El{el}_y_filt_lag1"].iloc[split:].to_numpy()
        R_true_te = df_arx[f"El{el}_R_true"].iloc[split:].to_numpy()

        sc = StandardScaler().fit(X_tr)
        x_scalers[el] = sc

        ridge = RidgeCV(alphas=np.logspace(-3, 4, 20), cv=5)
        ridge.fit(sc.transform(X_tr), y_tr)
        models[el] = ridge

        R_pred_te = R_lag1_te + ridge.predict(sc.transform(X_te))
        mae_dR = float(np.mean(np.abs(ridge.predict(sc.transform(X_te)) - y_te)))
        mae_R  = float(np.mean(np.abs(R_pred_te - R_true_te)))

        own_r_idx = [i for i, c in enumerate(xcols) if f"El{el}_y_filt_lag" in c]
        ar_coefs  = np.array(ridge.coef_)[own_r_idx]
        print(f"  El{el}: dR MAE={mae_dR:.5f}  R MAE={mae_R:.5f}  "
              f"AR coefs={ar_coefs.round(4).tolist()}  alpha={ridge.alpha_:.2e}")

    wrapper = DeltaARXWrapper.build(models, x_scalers, xcols_per_el,
                                    joint_xcols, _Y_COLS_JOINT, _Y_INDEX)

    # Identity Y_scaler: inverse_transform(x) = x*1 + 0 = x.
    # Fitting [-1, +1] per column gives mean_=0, scale_=1 exactly.
    _n_out    = len(_Y_COLS_JOINT)
    _y_sc_id  = StandardScaler().fit(np.vstack([-np.ones(_n_out), np.ones(_n_out)]))

    # Explicit physical clip bounds so the simulator does not use the Y_scaler-
    # derived clips (which would be [0, 6] with the identity scaler).
    _clips = {
        "r_clip":    {i: (0.5, 2.5)   for i in (1, 2, 3)},
        "ka_clip":   {i: (0.0, 400.0) for i in (1, 2, 3)},
        "reac_clip": {i: (0.0, 5.0)   for i in (1, 2, 3)},
    }

    bundle = dict(
        model_name     = "arx_v15_delta_per_electrode_5lags",
        X_cols         = joint_xcols,
        y_cols         = _Y_COLS_JOINT,
        y_index        = _Y_INDEX,
        model          = wrapper,
        Y_scaler       = _y_sc_id,
        X_scaler       = StandardScaler(with_mean=False, with_std=False).fit(
                             np.ones((2, len(joint_xcols)))),
        pi_rows        = f"{PI_ROW_START}-{PI_ROW_END}",
        delta_target   = True,
        clip_overrides = _clips,
    )
    out_path = MODELS_OUT / "arx_joint_v15.joblib"
    joblib.dump(bundle, out_path)
    print(f"  Saved ARX -> {out_path}")
    return bundle


def gp_feature_cols_v15(i: int) -> list[str]:
    """V9-style GP features (identical to V14, 36 features)."""
    other = [j for j in (1, 2, 3) if j != i]
    tc    = ["TCA", "TCB", "TCC"]
    cols  = [
        "step_in_window", "y_sim", "y_sim_sq",
        f"El{i}_dpos_mps_filt_lag1", f"El{i}_dpos_mps_filt_lag2",
        f"El{i}_dpos_mps_filt_lag3",
        f"El{i}_kA_filt_lag1",       f"El{i}_kA_filt_lag2",
        f"El{i}_CalcReac_filt_lag1", f"El{i}_CalcReac_filt_lag2",
        tc[i - 1], "RMS_V_transformer_filt_lag1",
    ]
    for j in other:
        cols += [f"El{j}_dpos_mps_filt_lag1", f"El{j}_dpos_mps_filt_lag2",
                 f"El{j}_y_filt_lag1",
                 f"El{j}_kA_filt_lag1", f"El{j}_kA_filt_lag2",
                 f"El{j}_CalcReac_filt_lag1", f"El{j}_CalcReac_filt_lag2",
                 tc[j - 1]]
    for j in (1, 2, 3):
        cols.append(f"El{j}_rolling_std_CalcReac_30s")
    for j in (1, 2, 3):
        cols.append(f"El{j}_rolling_std_R_30s")
    cols.append(f"El{i}_R_imbalance")
    cols.append("TCA_diff")
    return cols


def _sem_weight(d_raw: dict[int, np.ndarray], el: int, t: int) -> float:
    d_el = float(d_raw[el][t])
    total = d_el + sum(float(d_raw[j][t]) for j in (1, 2, 3) if j != el) + SEM_EPS
    return SEM_FLOOR + (1.0 - SEM_FLOOR) * d_el / total


def _arx_predict_v15(state: dict, wrapper: DeltaARXWrapper) -> dict[int, float]:
    all_cols = list(dict.fromkeys(c for el in (1, 2, 3) for c in wrapper.xcols_per_el_[el]))
    x = np.array([state.get(c, 0.0) for c in all_cols], dtype=np.float64).reshape(1, -1)
    out = {}
    for el in (1, 2, 3):
        col_idx = [all_cols.index(c) for c in wrapper.xcols_per_el_[el]]
        Xz  = wrapper.x_scalers_[el].transform(x[:, col_idx])
        dR  = float(wrapper.models_[el].predict(Xz)[0])
        out[el] = state.get(f"El{el}_y_filt_lag1", 0.0) + dR
    return out


def _advance_rollout_v15(state: dict, r_pred: dict[int, float],
                          df_arrays: dict, t_next: int) -> None:
    for i in (1, 2, 3):
        for k in range(N_R_LAGS, 1, -1):
            state[f"El{i}_y_filt_lag{k}"] = state.get(f"El{i}_y_filt_lag{k-1}", r_pred[i])
        state[f"El{i}_y_filt_lag1"] = r_pred[i]

    for i in (1, 2, 3):
        for k in range(N_DPOS_LAGS, 1, -1):
            state[f"El{i}_dpos_mps_filt_lag{k}"] = state.get(
                f"El{i}_dpos_mps_filt_lag{k-1}", 0.0)
        c1 = f"El{i}_dpos_mps_filt_lag1"
        state[c1] = float(df_arrays[c1][t_next]) if c1 in df_arrays else 0.0

        for sig in ("kA_filt", "CalcReac_filt"):
            for k in range(3, 1, -1):
                state[f"El{i}_{sig}_lag{k}"] = state.get(
                    f"El{i}_{sig}_lag{k-1}", state.get(f"El{i}_{sig}_lag1", 0.0))
            c1 = f"El{i}_{sig}_lag1"
            state[c1] = float(df_arrays[c1][t_next]) if c1 in df_arrays else state.get(c1, 0.0)


def build_gp_dataset_v15(df: pd.DataFrame, wrapper: DeltaARXWrapper,
                          el: int, el_idx: int = 0, n_els: int = 3,
                          t0_session: float = 0.0,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    feat_cols     = gp_feature_cols_v15(el)
    n             = len(df)
    window_starts = list(range(0, n - ROLLOUT_H - 1, ROLLOUT_STRIDE))
    n_windows     = len(window_starts)
    print(f"  El{el}: {n_windows:,} windows  (H={ROLLOUT_H}, stride={ROLLOUT_STRIDE})")

    all_feat_cols = sorted(set(c for el2 in (1, 2, 3) for c in wrapper.xcols_per_el_[el2]))
    all_needed    = set(feat_cols) | set(all_feat_cols) | {f"El{el}_R_true", "TCA_diff"}
    df_arrays     = {col: df[col].values for col in all_needed if col in df.columns}

    r_mean_arr = (df["El1_R_true"].values + df["El2_R_true"].values
                  + df["El3_R_true"].values) / 3.0
    r_true_el  = df[f"El{el}_R_true"].values
    d_raw      = {i: np.abs(df[f"El{i}_dpos_raw"].values) for i in (1, 2, 3)}

    X_list, y_list, w_list = [], [], []
    _PRINT_EVERY = max(1, n_windows // 20)
    t_el_start   = time.time()

    for wi, t0 in enumerate(window_starts):
        state = {c: float(df_arrays[c][t0]) if c in df_arrays else 0.0
                 for c in all_feat_cols}

        for h in range(ROLLOUT_H):
            t = t0 + h
            if t + 1 >= n:
                break

            r_pred    = _arx_predict_v15(state, wrapper)
            y_sim_val = r_pred[el]
            delta_val = float(r_true_el[t + 1]) - y_sim_val

            if np.isnan(delta_val):
                break

            sem_w = _sem_weight(d_raw, el, t)
            x     = np.empty(len(feat_cols), dtype=np.float32)
            bad   = False
            for fi, f in enumerate(feat_cols):
                if   f == "step_in_window":       x[fi] = float(h)
                elif f == "y_sim":                x[fi] = y_sim_val
                elif f == "y_sim_sq":             x[fi] = y_sim_val * y_sim_val
                elif f == f"El{el}_R_imbalance":  x[fi] = float(r_true_el[t] - r_mean_arr[t])
                elif f == "TCA_diff":             x[fi] = float(df_arrays["TCA_diff"][t])
                else:
                    x[fi] = float(state[f]) if f in state else (
                        float(df_arrays[f][t]) if f in df_arrays else 0.0)
                if np.isnan(x[fi]):
                    bad = True; break

            if bad:
                break

            X_list.append(x)
            y_list.append(np.float32(delta_val))
            w_list.append(np.float32(sem_w))

            if t + 1 < n:
                _advance_rollout_v15(state, r_pred, df_arrays, t + 1)

        if (wi + 1) % _PRINT_EVERY == 0 or wi == n_windows - 1:
            elapsed   = time.time() - t_el_start
            rate      = (wi + 1) / max(elapsed, 1e-6)
            eta_el    = (n_windows - wi - 1) / rate
            els_left  = n_els - el_idx - 1
            secs_per  = elapsed / max(wi + 1, 1) * n_windows
            eta_total = eta_el + els_left * secs_per

            def _fmt(s):
                h, r = divmod(int(s), 3600); m, sc = divmod(r, 60)
                return f"{h}h{m:02d}m" if h else f"{m}m{sc:02d}s"

            print(f"  El{el}  {wi+1:>5}/{n_windows}  "
                  f"elapsed={_fmt(elapsed)}  ETA this={_fmt(eta_el)}  "
                  f"ETA total={_fmt(eta_total)}  samples={len(X_list):,}", flush=True)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    w = np.array(w_list, dtype=np.float32)

    delta_mean = float(y.mean())
    hi         = int((w > 0.9).sum())
    pct        = np.percentile(w, [10, 50, 90])
    print(f"  El{el}: {len(X):,} samples  "
          f"SEM p10/50/90=[{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}]  "
          f"high-purity={hi:,} ({100*hi/max(len(w),1):.1f}%)  "
          f"delta_mean={delta_mean:.4f}  y_std={float(y.std()):.4f}")
    return X, y, w, delta_mean


def train_svgp_v15(X_all: np.ndarray, y_all: np.ndarray,
                   w_all: np.ndarray, delta_mean: float, el: int) -> dict:
    n     = len(X_all)
    n_val = int(n * VAL_FRACTION)
    rng   = np.random.default_rng(RANDOM_STATE)
    idx   = rng.permutation(n)
    X_tr, y_tr, w_tr = X_all[idx[n_val:]], y_all[idx[n_val:]], w_all[idx[n_val:]]
    X_va, y_va       = X_all[idx[:n_val]],  y_all[idx[:n_val]]
    print(f"  El{el}: train={len(X_tr):,}  val={len(X_va):,}  eff_SEM={w_tr.mean():.3f}")

    x_mean = X_tr.mean(0).astype(np.float32)
    x_std  = X_tr.std(0).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0

    y_tr_db = y_tr - delta_mean;  y_std = float(y_tr_db.std()) or 1.0
    y_va_db = y_va - delta_mean

    X_tr_s = (X_tr - x_mean) / x_std;  X_va_s = (X_va - x_mean) / x_std
    y_tr_s = y_tr_db / y_std;           y_va_s = y_va_db / y_std

    Xt = torch.tensor(X_tr_s, dtype=torch.float32)
    yt = torch.tensor(y_tr_s, dtype=torch.float32)
    Xv = torch.tensor(X_va_s, dtype=torch.float32)
    yv = torch.tensor(y_va_s, dtype=torch.float32)

    hp_idx  = np.where(w_tr > 0.7)[0]
    pool    = hp_idx if len(hp_idx) >= NUM_INDUCING else np.arange(len(X_tr_s))
    Z_init  = torch.tensor(X_tr_s[rng.choice(pool, NUM_INDUCING, replace=False)],
                            dtype=torch.float32)

    model = SVGPModel(Z_init, KernelSpec(name="matern52", ard=True)).to(DEVICE)
    lik   = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    model.train(); lik.train()
    opt = torch.optim.Adam([{"params": model.parameters()},
                            {"params": lik.parameters()}], lr=LR)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(Xt))

    wt      = torch.tensor(w_tr, dtype=torch.float32)
    sampler = WeightedRandomSampler(wt, num_samples=len(wt), replacement=True)
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                         sampler=sampler, pin_memory=(DEVICE.type == "cuda"))

    best_val = float("inf"); best_sd = None; best_lsd = None; pat = 0

    for epoch in range(1, EPOCHS + 1):
        model.train(); lik.train()
        ep_loss = sum(float((-mll(lik(model(xb.to(DEVICE))), yb.to(DEVICE))).item())
                      for xb, yb in loader) / len(loader)
        if epoch % 3 == 0 or epoch == 1:
            model.eval(); lik.eval()
            vl = []
            with torch.no_grad():
                for xvb, yvb in DataLoader(TensorDataset(Xv, yv), batch_size=BATCH_SIZE):
                    vl.append(float(-mll(lik(model(xvb.to(DEVICE))), yvb.to(DEVICE)).item()))
            val_loss = float(np.mean(vl))
            print(f"  El{el} epoch {epoch:3d}/{EPOCHS}  train={ep_loss:.4f}  val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_lsd = {k: v.cpu().clone() for k, v in lik.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE // 3:
                    print(f"  El{el} early stop at epoch {epoch}"); break
        else:
            for xb, yb in loader:
                opt.zero_grad()
                (-mll(lik(model(xb.to(DEVICE))), yb.to(DEVICE))).backward()
                opt.step()

    model.load_state_dict(best_sd); lik.load_state_dict(best_lsd)
    model.eval(); lik.eval()
    sigma_vals = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(Xt), batch_size=4096):
            sigma_vals.append(lik(model(xb.to(DEVICE))).variance.sqrt().cpu().numpy())
    sigma_ref = float(np.percentile(np.concatenate(sigma_vals), 90)) * y_std

    return {
        "model": model.cpu(), "likelihood": lik.cpu(),
        "feature_names": gp_feature_cols_v15(el),
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": np.array([0.0], dtype=np.float32),
        "y_std":  np.array([y_std], dtype=np.float32),
        "sigma_ref": sigma_ref, "r_op_offset": {},
        "metadata": {
            "variant": "v15", "delta_mean": delta_mean,
            "rollout_H": ROLLOUT_H, "rollout_stride": ROLLOUT_STRIDE,
            "arx_model": ARX_MODEL.name, "pi_rows": f"{PI_ROW_START}-{PI_ROW_END}",
            "n_r_lags": N_R_LAGS, "n_dpos_lags": N_DPOS_LAGS,
        },
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=== V15 GP Training (delta-R, 5 lags, fixed infrastructure) ===")
    print(f"R_lags={N_R_LAGS}  dpos_lags={N_DPOS_LAGS}  "
          f"H={ROLLOUT_H}  stride={ROLLOUT_STRIDE}\n")

    df = load_pi_data()

    arx_path = MODELS_OUT / "arx_joint_v15.joblib"
    if arx_path.exists():
        print("ARX SKIPPING (already exists)")
        bundle  = joblib.load(arx_path)
        wrapper = bundle["model"]
    else:
        print("ARX training")
        bundle  = train_arx_v15(df)
        wrapper = bundle["model"]

    els_to_train = [el for el in ELECTRODES
                    if not (MODELS_OUT / f"gp_el{el}_v15.pt").exists()]
    t_session = time.time()

    for el_i, el in enumerate(els_to_train):
        print(f"\nElectrode {el}")
        t_el = time.time()
        X, y, w, dm = build_gp_dataset_v15(df, wrapper, el,
                                            el_idx=el_i, n_els=len(els_to_train),
                                            t0_session=t_session)
        bun      = train_svgp_v15(X, y, w, dm, el)
        out_path = MODELS_OUT / f"gp_el{el}_v15.pt"
        torch.save(bun, out_path)
        print(f"  Saved: {out_path}  sigma_ref={bun['sigma_ref']:.6f}  "
              f"features={len(bun['feature_names'])}  ({time.time()-t_el:.0f}s)")

    print(f"\nV15 complete ({time.time()-t0:.0f}s).")
