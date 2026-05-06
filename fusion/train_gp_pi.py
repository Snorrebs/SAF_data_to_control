"""
train_gp_pi.py
--------------
Train per-electrode SVGP GP correction models using the PI plant data.

This produces the gp_el{i}_pi_512.pt files in fusion/models/.
It uses the same 512-inducing SVGP architecture as the 2026 txt-trained GPs,
but is trained on the middle 80% of the PI dataset (rows 276,480 to 2,488,320).

Run AFTER train_arx.py (or keep the included arx_joint_txt2026.joblib).

HOW TO RUN (from Spyder)
------------------------
1. Set your Spyder working directory to SAF_data_to_control/
2. Make sure PI_DATA_PATH below points to your PI-data.csv
3. Open this file in Spyder and press F5 (or Run)

PI DATA FORMAT
--------------
PI-data.csv has raw PI tag columns with backslash-separated paths, e.g.:
  \\ZMUCPI01\\V1903T830EG104.1 GI       (electrode position, cm)
  \\ZMUCPI01\\V1903T830EU100.1 Strøm    (arc current, kA)
  etc.

The mapping from PI tags to model signals is defined in the PI_* dicts below.
If your PI export uses different tag names, update those dicts.

EXPECTED RUNTIME (CPU, no GPU)
-------------------------------
  ~2.2M rows of PI data, STRIDE=360 -> ~6,100 windows per electrode
  ~20-40 minutes per electrode on CPU (total ~1-2 hours for all 3)
  GPU (if available) will be used automatically and is much faster.

WHAT IT SAVES
-------------
  fusion/models/gp_el1_pi_512.pt
  fusion/models/gp_el2_pi_512.pt
  fusion/models/gp_el3_pi_512.pt

To use the PI-trained GP in the simulator, set GP_VARIANT = "pi_512" in
run_closed_loop.py (and gp_variant = "pi_512" in example_rule_controller.py).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# =============================================================================
# CONFIGURE THESE
# =============================================================================

# Path to your PI-data.csv (absolute or relative to SAF_data_to_control/)
PI_DATA_PATH = "fusion/archive/data/raw/PI-data.csv"

# ARX model to use (the included txt2026 ARX works fine for PI data too)
ARX_MODEL = "fusion/models/arx_joint_txt2026.joblib"

# Which electrodes to train GP corrections for
ELECTRODES = [1, 2, 3]

# Output folder
MODELS_OUT = "fusion/models"

# PI data row range -- middle 80% of the 2,764,800-row dataset
PI_ROW_START = 276_480    # int(0.10 * 2_764_800)
PI_ROW_END   = 2_488_320  # int(0.90 * 2_764_800)

# ------- GP training hyperparameters -------
NUM_INDUCING = 512   # inducing points (more = more accurate, slower)
EPOCHS       = 100
LR           = 3e-3
BATCH_SIZE   = 1024
PATIENCE     = 12

# Window stride for the ARX rollout dataset.
# 360 = every 6 minutes -> ~6,100 windows over 2.2M rows.
# Increase to speed up dataset building at the cost of less coverage.
H      = 180
STRIDE = 360

VAL_FRACTION = 0.20
RANDOM_STATE = 42

# PI tag names -- update these if your export uses different names.
# GI = electrode position (cm, will be converted to m)
PI_GI  = {
    1: r"\\ZMUCPI01\V1903T830EG104.1 GI",
    2: r"\\ZMUCPI01\V1903T830EG204.1 GI",
    3: r"\\ZMUCPI01\V1903T830EG304.1 GI",
}
# Arc current (kA)
PI_KA  = {
    1: "\\\\ZMUCPI01\\V1903T830EU100.1 Strøm",
    2: "\\\\ZMUCPI01\\V1903T830EU200.1 Strøm",
    3: "\\\\ZMUCPI01\\V1903T830EU300.1 Strøm",
}
# Arc resistance (mOhm)
PI_RES = {
    1: r"\\ZMUCPI01\V1903T830EU100.1 Resistans",
    2: r"\\ZMUCPI01\V1903T830EU200.1 Resistans",
    3: r"\\ZMUCPI01\V1903T830EU300.1 Resistans",
}
# Tap changer positions (discrete, no filtering needed)
PI_TC  = {
    1: r"\\ZMUCPI01\V1903T830EU102.1 TCA",
    2: r"\\ZMUCPI01\V1903T830EU102.1 TCB",
    3: r"\\ZMUCPI01\V1903T830EU102.1 TCC",
}
# Per-phase voltages (used to compute shared transformer RMS voltage)
PI_UL  = {
    1: r"\\ZMUCPI01\V1903T870EE976.1 UL1N",
    2: r"\\ZMUCPI01\V1903T870EE976.1 UL2N",
    3: r"\\ZMUCPI01\V1903T870EE976.1 UL3N",
}

# Reactance clip: normal operating range is 0.8-1.1 mOhm.
# Pre-clip raw reactance before filtering to prevent furnace-off spikes
# (kA ~ 0 -> Z -> inf) from spreading into operating regions via filtfilt.
REAC_CLIP_MOHM = 3.0

# =============================================================================
# (nothing to change below this line)
# =============================================================================

import gc
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import SVGPModel from the package so torch.save stores the correct class path
# (fusion.archive.train_models.svgp_model.SVGPModel), enabling gp_loader to
# find it when loading the saved bundle.
from fusion.archive.train_models.svgp_model import SVGPModel, KernelSpec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_gp_pi] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

_FS = 1.0; _FC = 0.1   # 1 Hz sampling, 0.1 Hz cutoff


def _lp(arr: np.ndarray, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (same as ARX training pipeline)."""
    b, a = butter(order, _FC / (_FS / 2), btype="low", analog=False)
    s = pd.Series(arr).interpolate(limit=5).ffill().bfill().values
    return filtfilt(b, a, s, method="gust")


def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    """k-step lag: first k values are back-filled with arr[0]."""
    return np.concatenate([np.full(k, arr[0]), arr[:-k]])


# ---------------------------------------------------------------------------
# PI data loading (same logic as fusion/training/train_multi_arx_v2.py)
# ---------------------------------------------------------------------------

def load_pi_slice(path: Path, row_start: int, row_end: int) -> pd.DataFrame:
    """
    Load a row slice from PI-data.csv and return a preprocessed DataFrame.

    Applies the same Butterworth LP filters used in ARX training.
    Reactance is derived from the impedance triangle: X = sqrt(Z^2 - R^2)
    where Z = V_phase / kA.
    """
    all_pi_cols = (
        list(PI_GI.values()) + list(PI_KA.values())
        + list(PI_UL.values()) + list(PI_RES.values())
        + list(PI_TC.values())
    )
    # Deduplicate while preserving order (TC columns are shared across electrodes)
    seen = set(); all_pi_cols = [c for c in all_pi_cols if not (c in seen or seen.add(c))]

    print(f"  Loading PI rows {row_start:,} – {row_end:,}  "
          f"({row_end - row_start:,} rows) ...")
    raw = pd.read_csv(
        path,
        skiprows=range(1, row_start),   # skip header row + rows before row_start
        nrows=row_end - row_start,
        usecols=all_pi_cols,
        low_memory=False,
    ).reset_index(drop=True)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.interpolate(limit=10).ffill().bfill()

    # Per-phase RMS voltages for per-electrode impedance calculation
    ul = {i: raw[PI_UL[i]].values for i in (1, 2, 3)}

    # Shared transformer RMS (average of 3 phases)
    rms_v = np.sqrt((ul[1]**2 + ul[2]**2 + ul[3]**2) / 3.0)
    v_f   = np.clip(_lp(rms_v), 0.0, None)

    tca = pd.Series(raw[PI_TC[1]].values).ffill().bfill().values
    tcb = pd.Series(raw[PI_TC[2]].values).ffill().bfill().values
    tcc = pd.Series(raw[PI_TC[3]].values).ffill().bfill().values

    df = pd.DataFrame({
        "RMS_V_transformer_filt_lag1": _lag(v_f, 1),
        "V_target": v_f,
        "TCA": tca, "TCB": tcb, "TCC": tcc,
    })

    for i in (1, 2, 3):
        # Position: PI stores in cm, convert to m
        pos   = raw[PI_GI[i]].values / 100.0
        dpos  = np.concatenate([[0.0], np.diff(pos)])
        dpos_f = _lp(dpos)

        y_f  = _lp(raw[PI_RES[i]].values, order=2)
        ka_f = np.clip(_lp(raw[PI_KA[i]].values), 0.0, None)

        # Reactance from impedance triangle: X = sqrt(Z^2 - R^2)
        # Z = V_phase / kA  [V / (1000 A) = mOhm]
        # Pre-clip before LPF: kA~0 during furnace-off makes Z very large.
        z_i      = ul[i] / (ka_f + 1e-6)
        reac_raw = np.sqrt(np.maximum(z_i**2 - y_f**2, 0.0))
        reac_f   = np.clip(
            _lp(np.clip(reac_raw, 0.0, REAC_CLIP_MOHM * 5.0)),
            0.0, REAC_CLIP_MOHM,
        )

        p = f"El{i}"
        df[f"{p}_dpos_mps_filt_lag1"] = _lag(dpos_f, 1)
        df[f"{p}_dpos_mps_filt_lag2"] = _lag(dpos_f, 2)
        df[f"{p}_dpos_mps_filt_lag3"] = _lag(dpos_f, 3)
        df[f"{p}_pos_m_lag1"]         = _lag(pos, 1)
        df[f"{p}_y_filt_lag1"]        = _lag(y_f, 1)
        df[f"{p}_y_filt_lag2"]        = _lag(y_f, 2)
        df[f"{p}_y_filt_lag3"]        = _lag(y_f, 3)
        df[f"{p}_R_target"]           = y_f
        df[f"{p}_kA_filt_lag1"]       = _lag(ka_f, 1)
        df[f"{p}_kA_filt_lag2"]       = _lag(ka_f, 2)
        df[f"{p}_kA_filt_lag3"]       = _lag(ka_f, 3)
        df[f"{p}_CalcReac_filt_lag1"] = _lag(reac_f, 1)
        df[f"{p}_CalcReac_filt_lag2"] = _lag(reac_f, 2)
        df[f"{p}_CalcReac_filt_lag3"] = _lag(reac_f, 3)

        print(f"  El{i}  R: [{y_f.min():.3f}, {y_f.max():.3f}]  "
              f"kA: [{ka_f.min():.1f}, {ka_f.max():.1f}]  "
              f"Reac: [{reac_f.min():.3f}, {reac_f.max():.3f}] mOhm")

    df = df.iloc[3:].reset_index(drop=True)
    print(f"  {len(df):,} usable rows after lag warm-up drop")
    return df


# ---------------------------------------------------------------------------
# GP feature names (38 features per electrode -- same as txt2026 GP)
# ---------------------------------------------------------------------------

def _gp_feature_cols(i: int) -> list[str]:
    other = [j for j in (1, 2, 3) if j != i]
    tc = ["TCA", "TCB", "TCC"]
    cols = [
        "step_in_window",
        "y_sim",
        f"El{i}_dpos_mps_filt_lag1", f"El{i}_dpos_mps_filt_lag2",
        f"El{i}_dpos_mps_filt_lag3",
        f"El{i}_pos_m_lag1",
        f"El{i}_kA_filt_lag1", f"El{i}_kA_filt_lag2",
        f"El{i}_CalcReac_filt_lag1", f"El{i}_CalcReac_filt_lag2",
        tc[i - 1],
        "RMS_V_transformer_filt_lag1",
    ]
    for j in other:
        cols += [
            f"El{j}_dpos_mps_filt_lag1", f"El{j}_dpos_mps_filt_lag2",
            f"El{j}_pos_m_lag1",
            f"El{j}_y_filt_lag1",
            f"El{j}_kA_filt_lag1", f"El{j}_kA_filt_lag2",
            f"El{j}_CalcReac_filt_lag1", f"El{j}_CalcReac_filt_lag2",
            tc[j - 1],
        ]
    for j in (1, 2, 3):
        cols.append(f"El{j}_rolling_std_CalcReac_30s")
    for j in (1, 2, 3):
        cols.append(f"El{j}_rolling_std_R_30s")
    cols.append(f"El{i}_R_imbalance")
    cols.append("TCA_diff")
    return cols


# ---------------------------------------------------------------------------
# ARX fast inference
# ---------------------------------------------------------------------------

def _make_predictor(bundle: dict):
    model   = bundle["model"]
    xcols   = bundle["X_cols"]
    x_mean  = bundle["X_scaler"].mean_
    x_scale = bundle["X_scaler"].scale_
    y_mean  = bundle["Y_scaler"].mean_
    y_scale = bundle["Y_scaler"].scale_
    yidx    = bundle["y_index"]

    if hasattr(model, "estimators_"):
        coefs = np.array([np.asarray(e.coef_).ravel() for e in model.estimators_])
        ints  = np.array([float(np.asarray(e.intercept_).ravel()[0])
                          for e in model.estimators_])
    else:
        coefs = np.asarray(model.coef_).T
        ints  = np.asarray(model.intercept_).ravel()

    def predict_all(state: dict) -> np.ndarray:
        x   = np.array([state.get(c, 0.0) for c in xcols], dtype=np.float64)
        x_z = (x - x_mean) / x_scale
        y_z = coefs @ x_z + ints
        return y_z * y_scale + y_mean

    return predict_all, yidx


# ---------------------------------------------------------------------------
# Dataset builder: H-step free-running rollout
# ---------------------------------------------------------------------------

def build_gp_dataset(df: pd.DataFrame, bundle: dict, electrodes: list[int]) -> dict:
    """
    Build the GP training dataset using H-step free-running ARX rollout.

    Seeds each window from real data, runs the ARX freely for H steps using
    real electrode positions, and records (features, delta) where:
      delta = real_R_next - ARX_R_next
    The GP learns to predict delta to correct ARX errors.
    """
    predict_all, yidx = _make_predictor(bundle)

    n        = len(df)
    tc_arr   = {1: df["TCA"].values, 2: df["TCB"].values, 3: df["TCC"].values}
    y_real   = {i: df[f"El{i}_R_target"].values for i in electrodes}
    fcols    = {i: _gp_feature_cols(i) for i in electrodes}

    all_starts  = list(range(3, n - H - 1, STRIDE))
    total_win   = len(all_starts)
    print_every = max(1, min(500, total_win // 20))
    t0_build    = time.time()

    max_rows = total_win * H
    arr      = {i: np.empty((max_rows, 1 + len(fcols[i]) + 1), dtype=np.float32)
                for i in electrodes}
    row_idx  = {i: 0 for i in electrodes}
    delta_buf = {i: [] for i in electrodes}

    print(f"  {total_win:,} rollout windows  (H={H} steps, stride={STRIDE})")

    for wid, t0 in enumerate(all_starts):
        if wid > 0 and wid % print_every == 0:
            elapsed = time.time() - t0_build
            rate    = wid / elapsed if elapsed > 0 else 1e-9
            eta     = (total_win - wid) / rate
            pct     = 100 * wid / total_win
            parts   = [f"el{i} delta={float(np.mean(delta_buf[i])):+.4f}"
                       for i in electrodes if delta_buf[i]]
            print(f"  {wid:>6,}/{total_win:,} ({pct:4.1f}%)  "
                  f"elapsed={elapsed/60:4.1f}min  ETA={eta/60:4.1f}min  " +
                  "  ".join(parts))
            for i in electrodes:
                delta_buf[i].clear()

        _tca_prev = float(df["TCA"].iloc[max(0, t0 - 1)])
        state: dict[str, float] = {
            "TCA": float(df["TCA"].iloc[t0]),
            "TCB": float(df["TCB"].iloc[t0]),
            "TCC": float(df["TCC"].iloc[t0]),
            "TCA_diff": float(df["TCA"].iloc[t0]) - _tca_prev,
            "RMS_V_transformer_filt_lag1":
                float(df["RMS_V_transformer_filt_lag1"].iloc[t0]),
        }
        for j in (1, 2, 3):
            for col in [
                f"El{j}_dpos_mps_filt_lag1", f"El{j}_dpos_mps_filt_lag2",
                f"El{j}_dpos_mps_filt_lag3", f"El{j}_pos_m_lag1",
                f"El{j}_y_filt_lag1", f"El{j}_y_filt_lag2", f"El{j}_y_filt_lag3",
                f"El{j}_kA_filt_lag1", f"El{j}_kA_filt_lag2", f"El{j}_kA_filt_lag3",
                f"El{j}_CalcReac_filt_lag1", f"El{j}_CalcReac_filt_lag2",
                f"El{j}_CalcReac_filt_lag3",
            ]:
                state[col] = float(df[col].iloc[t0]) if col in df.columns else 0.0

        # Seed rolling variability buffers from real data before this window
        rolling_reac: dict[int, deque] = {}
        rolling_r:    dict[int, deque] = {}
        for j in (1, 2, 3):
            r_start = max(0, t0 - 30 + 1)
            rolling_reac[j] = deque(
                df[f"El{j}_CalcReac_filt_lag1"].iloc[r_start:t0+1].values, maxlen=30)
            rolling_r[j] = deque(
                df[f"El{j}_y_filt_lag1"].iloc[r_start:t0+1].values, maxlen=30)
        for j in (1, 2, 3):
            state[f"El{j}_rolling_std_CalcReac_30s"] = float(np.std(rolling_reac[j]))
            state[f"El{j}_rolling_std_R_30s"]        = float(np.std(rolling_r[j]))

        # H-step free-running ARX rollout using real electrode positions
        for k in range(H):
            t = t0 + k
            if t + 1 >= n:
                break

            prev_tca = state.get("TCA", float(tc_arr[1][t]))
            state["TCA"]            = float(tc_arr[1][t])
            state["TCB"]            = float(tc_arr[2][t])
            state["TCC"]            = float(tc_arr[3][t])
            state["TCA_diff"]       = state["TCA"] - prev_tca
            state["step_in_window"] = float(k)

            y_all = predict_all(state)

            new_r    = {j: float(max(y_all[yidx["R"][j]],    0.0)) for j in (1, 2, 3)}
            new_ka   = {j: float(max(y_all[yidx["kA"][j]],   0.0)) for j in (1, 2, 3)}
            new_reac = {j: float(np.clip(max(y_all[yidx["reac"][j]], 0.0), 0.0, 3.0))
                        for j in (1, 2, 3)}
            new_v    = float(max(y_all[yidx["v"]], 0.0))

            r_mean_all = sum(new_r[j] for j in (1, 2, 3)) / 3.0
            for i in electrodes:
                y_sim_i = new_r[i]
                state["y_sim"]              = y_sim_i
                state[f"El{i}_R_imbalance"] = y_sim_i - r_mean_all
                delta_i = float(y_real[i][t + 1]) - y_sim_i
                delta_buf[i].append(delta_i)

                ri = row_idx[i]
                arr[i][ri, 0] = wid
                for fi, fc in enumerate(fcols[i]):
                    arr[i][ri, 1 + fi] = state.get(fc, 0.0)
                arr[i][ri, -1] = delta_i
                row_idx[i] += 1

            for j in (1, 2, 3):
                dpos_t1 = (float(df[f"El{j}_dpos_mps_filt_lag1"].iloc[t + 1])
                           if t + 1 < n else 0.0)
                pos_t1  = (float(df[f"El{j}_pos_m_lag1"].iloc[t + 1])
                           if t + 1 < n else state[f"El{j}_pos_m_lag1"])
                state[f"El{j}_dpos_mps_filt_lag3"] = state[f"El{j}_dpos_mps_filt_lag2"]
                state[f"El{j}_dpos_mps_filt_lag2"] = state[f"El{j}_dpos_mps_filt_lag1"]
                state[f"El{j}_dpos_mps_filt_lag1"] = dpos_t1
                state[f"El{j}_pos_m_lag1"]         = pos_t1
                state[f"El{j}_y_filt_lag3"]        = state[f"El{j}_y_filt_lag2"]
                state[f"El{j}_y_filt_lag2"]        = state[f"El{j}_y_filt_lag1"]
                state[f"El{j}_y_filt_lag1"]        = new_r[j]
                state[f"El{j}_kA_filt_lag3"]       = state[f"El{j}_kA_filt_lag2"]
                state[f"El{j}_kA_filt_lag2"]       = state[f"El{j}_kA_filt_lag1"]
                state[f"El{j}_kA_filt_lag1"]       = new_ka[j]
                state[f"El{j}_CalcReac_filt_lag3"] = state[f"El{j}_CalcReac_filt_lag2"]
                state[f"El{j}_CalcReac_filt_lag2"] = state[f"El{j}_CalcReac_filt_lag1"]
                state[f"El{j}_CalcReac_filt_lag1"] = new_reac[j]
                rolling_reac[j].append(new_reac[j])
                rolling_r[j].append(new_r[j])
            state["RMS_V_transformer_filt_lag1"] = new_v
            for j in (1, 2, 3):
                state[f"El{j}_rolling_std_CalcReac_30s"] = float(np.std(rolling_reac[j]))
                state[f"El{j}_rolling_std_R_30s"]        = float(np.std(rolling_r[j]))

    gc.collect()
    datasets: dict[int, pd.DataFrame] = {}
    for i in electrodes:
        cols = ["window_id"] + fcols[i] + ["delta"]
        df_i = pd.DataFrame(arr[i][:row_idx[i]], columns=cols)
        df_i["window_id"] = df_i["window_id"].astype(np.int32)
        print(f"  El{i}: {len(df_i):,} rows  "
              f"delta mean={df_i['delta'].mean():+.5f}  std={df_i['delta'].std():.5f}")
        datasets[i] = df_i
    return datasets


# ---------------------------------------------------------------------------
# SVGP training
# ---------------------------------------------------------------------------

def _chrono_split(df: pd.DataFrame) -> tuple:
    wids    = df["window_id"].unique()
    rng     = np.random.default_rng(RANDOM_STATE)
    n_val   = max(1, int(len(wids) * VAL_FRACTION))
    val_ids = set(rng.choice(wids, size=n_val, replace=False).tolist())
    return df[~df["window_id"].isin(val_ids)], df[df["window_id"].isin(val_ids)]


def train_gp_electrode(electrode: int, dataset: pd.DataFrame, out_path: Path) -> None:
    i     = electrode
    fcols = _gp_feature_cols(i)

    print(f"\n{'='*60}")
    print(f"  Training GP (PI data) for electrode {i}  ({len(fcols)} features)")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    train_df, val_df = _chrono_split(dataset)
    print(f"  Train windows: {train_df['window_id'].nunique():,}  "
          f"rows: {len(train_df):,}")
    print(f"  Val   windows: {val_df['window_id'].nunique():,}  "
          f"rows: {len(val_df):,}")

    X_train = train_df[fcols].to_numpy(dtype=np.float64)
    y_train = train_df["delta"].to_numpy(dtype=np.float64)
    X_val   = val_df[fcols].to_numpy(dtype=np.float64)
    y_val   = val_df["delta"].to_numpy(dtype=np.float64)

    x_mean = X_train.mean(axis=0).astype(np.float32)
    x_std  = np.where(X_train.std(axis=0) < 1e-8, 1.0,
                      X_train.std(axis=0)).astype(np.float32)
    y_mean = np.array([float(y_train.mean())], dtype=np.float32)
    y_std  = np.array([max(float(y_train.std()), 1e-8)], dtype=np.float32)

    X_tr_s = ((X_train - x_mean) / x_std).astype(np.float32)
    X_vl_s = ((X_val   - x_mean) / x_std).astype(np.float32)
    y_tr_s = ((y_train - float(y_mean[0])) / float(y_std[0])).astype(np.float32)
    y_vl_s = ((y_val   - float(y_mean[0])) / float(y_std[0])).astype(np.float32)

    print(f"  Target std (train): {y_train.std():.6f} mOhm")

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32).to(DEVICE)
    rng     = np.random.default_rng(RANDOM_STATE)
    idx     = rng.choice(len(X_tr_t), size=min(NUM_INDUCING, len(X_tr_t)), replace=False)
    Z       = X_tr_t[idx].clone()

    kspec      = KernelSpec(name="matern32", ard=True)
    model      = SVGPModel(Z, kspec).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    X_vl_t = torch.tensor(X_vl_s, dtype=torch.float32).to(DEVICE)
    y_tr_t  = torch.tensor(y_tr_s, dtype=torch.float32).to(DEVICE)
    y_vl_t  = torch.tensor(y_vl_s, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=BATCH_SIZE, shuffle=True)
    mll    = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_tr_t))
    opt    = torch.optim.Adam([{"params": model.parameters()},
                               {"params": likelihood.parameters()}], lr=LR)

    best_mae   = float("inf")
    best_state = None
    patience   = PATIENCE
    ep_times: list[float] = []

    print(f"  Training SVGP ({NUM_INDUCING} inducing pts, max {EPOCHS} epochs) ...")
    with gpytorch.settings.cholesky_jitter(1e-3):
        for ep in range(1, EPOCHS + 1):
            t_ep = time.time()
            model.train(); likelihood.train()
            total_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = -mll(model(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(xb)
            train_loss = total_loss / len(X_tr_t)

            model.eval(); likelihood.eval()
            with torch.no_grad():
                pred   = likelihood(model(X_vl_t))
                val_mu = pred.mean.cpu().numpy()
            mae = float(np.mean(np.abs(y_vl_s - val_mu)))

            ep_times.append(time.time() - t_ep)
            avg_ep = sum(ep_times) / len(ep_times)
            eta    = avg_ep * (EPOCHS - ep)
            print(f"  ep {ep:03d}/{EPOCHS}  loss={train_loss:.4f}  "
                  f"val_MAE={mae:.4f}  ep={avg_ep:.0f}s  ETA={eta/60:.0f}min",
                  flush=True)

            if mae < best_mae:
                best_mae   = mae
                best_state = {
                    "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "lik":   {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
                    "epoch": ep,
                }
                patience = PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    print(f"  Early stopping — restoring epoch {best_state['epoch']} "
                          f"(val_MAE={best_mae:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["lik"])
    model.eval(); likelihood.eval()
    model = model.cpu(); likelihood = likelihood.cpu()

    # sigma_ref: 90th-percentile of predictive std on training data.
    # Used for GP uncertainty blending in plant.py.
    X_tr_cpu = X_tr_t.cpu()
    with torch.no_grad():
        preds_s = likelihood(model(X_tr_cpu))
        vars_s  = preds_s.variance.numpy()
    sigma_ref = float(np.percentile(
        np.sqrt(np.maximum(vars_s, 0.0)) * float(y_std[0]), 90))
    print(f"  sigma_ref (90th-pct train std) = {sigma_ref:.6f} mOhm")

    feed_feature_means = {f: float(x_mean[fi]) for fi, f in enumerate(fcols)}

    bundle = {
        "model":              model,
        "likelihood":         likelihood,
        "feature_names":      fcols,
        "target_name":        "delta",
        "model_type":         "svgp",
        "x_mean":             x_mean,
        "x_std":              x_std,
        "y_mean":             y_mean,
        "y_std":              y_std,
        "metadata":           {
            "electrode": i, "H": H, "stride": STRIDE,
            "num_inducing": NUM_INDUCING,
            "data_source": "PI middle-80%",
            "pi_rows": f"{PI_ROW_START}-{PI_ROW_END}",
        },
        "sigma_ref":          sigma_ref,
        "feed_feature_means": feed_feature_means,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"  Saved -> {out_path}")

    # Quick holdout evaluation in physical units
    with torch.no_grad():
        pred_vl = likelihood(model(X_vl_t.cpu()))
        mu_vl   = pred_vl.mean.numpy() * float(y_std[0]) + float(y_mean[0])
    mae_phys = float(np.mean(np.abs(y_val - mu_vl)))
    print(f"  Val MAE (physical) = {mae_phys:.5f} mOhm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GP correction training -- PI data (middle 80%)")
    print("=" * 60)
    print(f"PI data  : {PI_DATA_PATH}")
    print(f"Rows     : {PI_ROW_START:,} – {PI_ROW_END:,}  "
          f"({PI_ROW_END - PI_ROW_START:,} rows = middle 80%)")
    print(f"ARX model: {ARX_MODEL}")
    print(f"Electrodes: {ELECTRODES}")
    print(f"Models out: {MODELS_OUT}")

    pi_path = _PROJECT_ROOT / PI_DATA_PATH
    if not pi_path.exists():
        raise FileNotFoundError(
            f"PI data not found: {pi_path}\n"
            f"Set PI_DATA_PATH at the top of this script."
        )

    arx_path = _PROJECT_ROOT / ARX_MODEL
    if not arx_path.exists():
        raise FileNotFoundError(
            f"ARX model not found: {arx_path}\n"
            "Run train_arx.py first, or use the included arx_joint_txt2026.joblib."
        )

    # Inject __main__.ReducedRankRidge so joblib can find the class when loading
    # the included arx_joint_txt2026.joblib (which was pickled as __main__.ReducedRankRidge).
    import __main__
    from fusion.training.arx_model import ReducedRankRidge as _RRR
    if not hasattr(__main__, "ReducedRankRidge"):
        __main__.ReducedRankRidge = _RRR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arx_bundle = joblib.load(arx_path)
    print(f"Loaded ARX: {len(arx_bundle['X_cols'])} features -> "
          f"{len(arx_bundle['y_cols'])} outputs")

    # Load and preprocess PI data
    print("\n--- Loading PI data ---")
    df = load_pi_slice(_PROJECT_ROOT / PI_DATA_PATH, PI_ROW_START, PI_ROW_END)

    # Use first 80% for training, last 20% as holdout (never seen during training)
    n_train  = int(len(df) * 0.80)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_hold  = df.iloc[n_train:].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} rows   Holdout: {len(df_hold):,} rows")
    del df

    # Build GP training dataset via H-step ARX rollout
    print("\n--- Building GP training dataset (ARX rollout) ---")
    datasets = build_gp_dataset(df_train, arx_bundle, ELECTRODES)
    del df_train

    # Train one GP per electrode
    out_dir = _PROJECT_ROOT / MODELS_OUT
    for i in ELECTRODES:
        out_path = out_dir / f"gp_el{i}_pi_512.pt"
        train_gp_electrode(i, datasets[i], out_path)

    print("\n" + "=" * 60)
    print("GP PI training complete.")
    print("New model files:")
    for i in ELECTRODES:
        print(f"  fusion/models/gp_el{i}_pi_512.pt")
    print("\nTo use: set GP_VARIANT = \"pi_512\" in run_closed_loop.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
else:
    main()   # also run when exec()'d from Spyder
