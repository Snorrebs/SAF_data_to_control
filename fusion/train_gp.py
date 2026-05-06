"""
train_gp.py
-----------
Train per-electrode SVGP GP correction models on top of the joint ARX.

Run AFTER train_arx.py. Saves one .pt file per electrode into fusion/models/.

HOW TO RUN (from Spyder)
------------------------
1. Set your Spyder working directory to SAF_data_to_control/
2. Make sure train_arx.py has already been run (fusion/models/arx_joint_txt2026.joblib exists)
3. Use the same DATA_FOLDER as in train_arx.py
4. Open this file in Spyder and press F5 (or Run)

EXPECTED RUNTIME (CPU, no GPU)
-------------------------------
  ~140k rows: 20-40 minutes per electrode
  ~500k rows: 1-2 hours per electrode

Training all three electrodes runs sequentially; the total time is 3x the above.
If you have a GPU, PyTorch will use it automatically and training will be much faster.

HOW THE GP WORKS
----------------
At each time step, the joint ARX model predicts all 10 signals one step ahead.
The GP learns to predict the RESIDUAL (real R - ARX R) from the simulator state.
At inference, the GP correction is added to the ARX prediction:
  y_fused = y_arx + GP.predict(features)

This is why the GP training needs the ARX model: it runs the ARX in a free-running
simulation over your data to generate the ARX errors for the GP to learn from.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# =============================================================================
# CONFIGURE THESE
# =============================================================================

# Same data folder as in train_arx.py
DATA_FOLDER = "fusion/data"

# ARX model to build GP corrections on top of
ARX_MODEL = "fusion/models/arx_joint_txt2026.joblib"

# Which electrodes to train corrections for (train all three if you have enough data)
ELECTRODES = [1, 2, 3]

# Output folder for GP models
MODELS_OUT = "fusion/models"

# Column name mapping (must match train_arx.py)
COLUMN_MAP = {
    "r":    "El{i}_Resistance_mOhm_filt",
    "pos":  "El{i}_pos_m",
    "ka":   "El{i}_kA_filt",
    "reac": "El{i}_CalcReac_filt",
    "v":    "RMS_V_transformer_filt",
    "tca":  "TCA",
    "tcb":  "TCB",
    "tcc":  "TCC",
}

# ------- GP training hyperparameters -------
# Reduce NUM_INDUCING or EPOCHS if training is too slow; increase for better accuracy.
NUM_INDUCING = 256   # number of sparse GP inducing points (more = more accurate, slower)
EPOCHS       = 100   # max training epochs (early stopping may terminate sooner)
LR           = 3e-3  # learning rate
BATCH_SIZE   = 1024  # mini-batch size
PATIENCE     = 12    # early stopping patience (epochs without improvement)

# Rollout window: each training window runs the ARX for H steps from a seed point.
# Longer H captures longer-range dynamics but takes more time to build the dataset.
H      = 180   # window length in seconds (= steps at 1s sampling)
STRIDE = 90    # spacing between window starts (overlap = H - STRIDE steps)

HOLDOUT_FRAC   = 0.10   # last fraction of data withheld for evaluation
VAL_FRACTION   = 0.20   # fraction of windows used for validation during training
RANDOM_STATE   = 42

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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import SVGPModel from the package so that torch.save stores the correct
# class path (fusion.archive.train_models.svgp_model.SVGPModel), which lets
# gp_loader.py find it when loading the saved bundle.
from fusion.archive.train_models.svgp_model import SVGPModel, KernelSpec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_gp] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Signal preprocessing (same as train_arx.py)
# ---------------------------------------------------------------------------

_FS = 1.0; _FC = 0.1


def _lp(arr: np.ndarray, order: int = 4) -> np.ndarray:
    b, a = butter(order, _FC / (_FS / 2), btype="low", analog=False)
    s = pd.Series(arr).interpolate(limit=5).ffill().bfill().values
    return filtfilt(b, a, s, method="gust")


def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k, arr[0]), arr[:-k]])


def _col(template: str, i: int) -> str:
    return template.replace("{i}", str(i))


def load_and_preprocess(data_folder: str | Path) -> pd.DataFrame:
    folder = Path(data_folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(raw):,} rows from {len(csv_files)} file(s)")

    n   = len(raw)
    out: dict[str, np.ndarray] = {}

    v_f = np.clip(_lp(raw[COLUMN_MAP["v"]].values.astype(np.float64)), 0.0, None)
    out["RMS_V_transformer_filt_lag1"] = _lag(v_f, 1)
    out["V_target"] = v_f

    for name, key in [("TCA", "tca"), ("TCB", "tcb"), ("TCC", "tcc")]:
        out[name] = raw[COLUMN_MAP[key]].values.astype(np.float64)

    for i in (1, 2, 3):
        p    = f"El{i}"
        pos  = raw[_col(COLUMN_MAP["pos"], i)].values.astype(np.float64)
        dpos = np.concatenate([[0.0], np.diff(pos)])
        dpos_f = _lp(dpos)
        out[f"{p}_dpos_mps_filt_lag1"] = _lag(dpos_f, 1)
        out[f"{p}_dpos_mps_filt_lag2"] = _lag(dpos_f, 2)
        out[f"{p}_dpos_mps_filt_lag3"] = _lag(dpos_f, 3)
        out[f"{p}_pos_m_lag1"]         = _lag(pos, 1)

        y_f  = _lp(raw[_col(COLUMN_MAP["r"], i)].values.astype(np.float64), order=2)
        out[f"{p}_y_filt_lag1"] = _lag(y_f, 1)
        out[f"{p}_y_filt_lag2"] = _lag(y_f, 2)
        out[f"{p}_y_filt_lag3"] = _lag(y_f, 3)
        out[f"{p}_R_target"]    = y_f

        ka_f = np.clip(_lp(raw[_col(COLUMN_MAP["ka"], i)].values.astype(np.float64)), 0.0, None)
        out[f"{p}_kA_filt_lag1"] = _lag(ka_f, 1)
        out[f"{p}_kA_filt_lag2"] = _lag(ka_f, 2)
        out[f"{p}_kA_filt_lag3"] = _lag(ka_f, 3)

        reac_f = np.clip(_lp(np.clip(raw[_col(COLUMN_MAP["reac"], i)].values.astype(np.float64),
                                     0.0, 3.0)), 0.0, 3.0)
        out[f"{p}_CalcReac_filt_lag1"] = _lag(reac_f, 1)
        out[f"{p}_CalcReac_filt_lag2"] = _lag(reac_f, 2)
        out[f"{p}_CalcReac_filt_lag3"] = _lag(reac_f, 3)

    df = pd.DataFrame(out)
    df = df.iloc[3:].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# GP feature names (38 features per electrode)
# ---------------------------------------------------------------------------

def _gp_feature_cols(i: int) -> list[str]:
    other = [j for j in (1, 2, 3) if j != i]
    tc = ["TCA", "TCB", "TCC"]
    cols = [
        "step_in_window",
        "y_sim",
        f"El{i}_dpos_mps_filt_lag1", f"El{i}_dpos_mps_filt_lag2", f"El{i}_dpos_mps_filt_lag3",
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
# ARX fast inference (extracts weight arrays to avoid sklearn overhead in loop)
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
        ints  = np.array([float(np.asarray(e.intercept_).ravel()[0]) for e in model.estimators_])
    else:
        coefs = np.asarray(model.coef_).T
        ints  = np.asarray(model.intercept_).ravel()

    def predict_all(state: dict) -> np.ndarray:
        x   = np.array([state.get(c, 0.0) for c in xcols], dtype=np.float64)
        x_z = (x - x_mean) / x_scale
        y_z = coefs @ x_z + ints
        return y_z * y_scale + y_mean   # shape (10,)

    return predict_all, yidx


# ---------------------------------------------------------------------------
# Dataset builder: H-step free-running rollout
# ---------------------------------------------------------------------------

def build_gp_dataset(df: pd.DataFrame, bundle: dict, electrodes: list[int]) -> dict:
    """
    Build the GP training dataset using H-step free-running ARX rollout.

    For each window: seed the ARX from real data, run it freely for H steps
    using real electrode positions. Record (GP features, delta) at each step
    where delta = real_R_next - ARX_R_next. The GP learns to predict this delta.
    """
    predict_all, yidx = _make_predictor(bundle)

    n        = len(df)
    tc_arr   = {1: df["TCA"].values, 2: df["TCB"].values, 3: df["TCC"].values}
    y_real   = {i: df[f"El{i}_R_target"].values for i in electrodes}
    fcols    = {i: _gp_feature_cols(i) for i in electrodes}

    all_starts   = list(range(3, n - H - 1, STRIDE))
    total_win    = len(all_starts)
    print_every  = max(1, min(500, total_win // 20))
    t0_build     = time.time()

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

        # Seed state from real data
        _tca_prev = float(df["TCA"].iloc[max(0, t0 - 1)])
        state: dict[str, float] = {
            "TCA": float(df["TCA"].iloc[t0]),
            "TCB": float(df["TCB"].iloc[t0]),
            "TCC": float(df["TCC"].iloc[t0]),
            "TCA_diff": float(df["TCA"].iloc[t0]) - _tca_prev,
            "RMS_V_transformer_filt_lag1": float(df["RMS_V_transformer_filt_lag1"].iloc[t0]),
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

        # Seed rolling buffers from real data before the window
        rolling_reac: dict[int, deque] = {}
        rolling_r:    dict[int, deque] = {}
        for j in (1, 2, 3):
            r_start = max(0, t0 - 30 + 1)
            rolling_reac[j] = deque(df[f"El{j}_CalcReac_filt_lag1"].iloc[r_start:t0+1].values,
                                    maxlen=30)
            rolling_r[j]    = deque(df[f"El{j}_y_filt_lag1"].iloc[r_start:t0+1].values,
                                    maxlen=30)
        for j in (1, 2, 3):
            state[f"El{j}_rolling_std_CalcReac_30s"] = float(np.std(rolling_reac[j]))
            state[f"El{j}_rolling_std_R_30s"]        = float(np.std(rolling_r[j]))

        # H-step free-running rollout
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

            # Compute R imbalance for each electrode and write rows
            r_mean_all = sum(new_r[j] for j in (1, 2, 3)) / 3.0
            for i in electrodes:
                y_sim_i = new_r[i]
                state["y_sim"] = y_sim_i
                state[f"El{i}_R_imbalance"] = y_sim_i - r_mean_all
                delta_i = float(y_real[i][t + 1]) - y_sim_i
                delta_buf[i].append(delta_i)

                ri = row_idx[i]
                arr[i][ri, 0] = wid
                for fi, fc in enumerate(fcols[i]):
                    arr[i][ri, 1 + fi] = state.get(fc, 0.0)
                arr[i][ri, -1] = delta_i
                row_idx[i] += 1

            # Advance lag registers (use real position; ARX-predicted signals)
            for j in (1, 2, 3):
                dpos_t1 = float(df[f"El{j}_dpos_mps_filt_lag1"].iloc[t + 1]) if t + 1 < n else 0.0
                pos_t1  = float(df[f"El{j}_pos_m_lag1"].iloc[t + 1]) if t + 1 < n else state[f"El{j}_pos_m_lag1"]
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


def train_gp_electrode(
    electrode: int,
    dataset:   pd.DataFrame,
    out_path:  Path,
) -> None:
    i     = electrode
    fcols = _gp_feature_cols(i)

    print(f"\n{'='*60}")
    print(f"  Training GP for electrode {i}  ({len(fcols)} features)")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    train_df, val_df = _chrono_split(dataset)
    print(f"  Train windows: {train_df['window_id'].nunique():,}  rows: {len(train_df):,}")
    print(f"  Val   windows: {val_df['window_id'].nunique():,}  rows: {len(val_df):,}")

    X_train = train_df[fcols].to_numpy(dtype=np.float64)
    y_train = train_df["delta"].to_numpy(dtype=np.float64)
    X_val   = val_df[fcols].to_numpy(dtype=np.float64)
    y_val   = val_df["delta"].to_numpy(dtype=np.float64)

    # Standardise features
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

    # Build model
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
                pred = likelihood(model(X_vl_t))
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
                best_state = {"model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                              "lik":   {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
                              "epoch": ep}
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

    # Compute sigma_ref: 90th-percentile of predictive std on training data
    X_tr_cpu = X_tr_t.cpu()
    with torch.no_grad():
        preds_s = likelihood(model(X_tr_cpu))
        vars_s  = preds_s.variance.numpy()
    sigma_ref = float(np.percentile(np.sqrt(np.maximum(vars_s, 0.0)) * float(y_std[0]), 90))
    print(f"  sigma_ref (90th-pct train std) = {sigma_ref:.6f} mOhm")

    # Compute feature means (for fallback in plant.py when features are missing)
    feed_feature_means = {f: float(x_mean[fi]) for fi, f in enumerate(fcols)}

    # Save bundle in the same format as the original GP bundles
    bundle = {
        "model":             model,
        "likelihood":        likelihood,
        "feature_names":     fcols,
        "target_name":       "delta",
        "model_type":        "svgp",
        "x_mean":            x_mean,
        "x_std":             x_std,
        "y_mean":            y_mean,
        "y_std":             y_std,
        "metadata":          {"electrode": i, "H": H, "stride": STRIDE,
                              "num_inducing": NUM_INDUCING},
        "sigma_ref":         sigma_ref,
        "feed_feature_means": feed_feature_means,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"  Saved -> {out_path}")

    # Quick holdout evaluation on the val set (in physical units)
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
    print("GP correction training")
    print("=" * 60)
    print(f"Data folder : {DATA_FOLDER}")
    print(f"ARX model   : {ARX_MODEL}")
    print(f"Electrodes  : {ELECTRODES}")
    print(f"Models out  : {MODELS_OUT}")

    # Load ARX model
    arx_path = _PROJECT_ROOT / ARX_MODEL
    if not arx_path.exists():
        raise FileNotFoundError(
            f"ARX model not found: {arx_path}\n"
            "Run train_arx.py first."
        )
    import __main__
    from fusion.training.arx_model import ReducedRankRidge as _RRR
    if not hasattr(__main__, "ReducedRankRidge"):
        __main__.ReducedRankRidge = _RRR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arx_bundle = joblib.load(arx_path)
    print(f"Loaded ARX model  ({len(arx_bundle['X_cols'])} features -> {len(arx_bundle['y_cols'])} outputs)")

    # Load and preprocess data
    print("\n--- Loading data ---")
    df = load_and_preprocess(_PROJECT_ROOT / DATA_FOLDER)
    n_train = int(len(df) * (1.0 - HOLDOUT_FRAC))
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_hold  = df.iloc[n_train:].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} rows   Holdout: {len(df_hold):,} rows")
    del df

    # Build GP datasets (ARX rollout)
    print("\n--- Building GP training dataset (ARX rollout) ---")
    datasets = build_gp_dataset(df_train, arx_bundle, ELECTRODES)
    del df_train

    # Train one GP per electrode
    out_dir = _PROJECT_ROOT / MODELS_OUT
    for i in ELECTRODES:
        out_path = out_dir / f"gp_el{i}_txt2026_512.pt"
        train_gp_electrode(i, datasets[i], out_path)

    print("\n" + "=" * 60)
    print("GP training complete.")
    print("New model files are in:", out_dir)
    print("VRFT v5.py will use them automatically (via run_closed_loop.py).")
    print("=" * 60)


if __name__ == "__main__":
    main()
else:
    main()   # also run when exec()'d from Spyder
