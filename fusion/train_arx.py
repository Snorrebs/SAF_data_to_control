"""
train_arx.py
------------
Train the joint ARX model from CSV data files.

Saves the new model to fusion/models/arx_joint.joblib.

HOW TO RUN 
------------------------
1. Set your working directory to SAF_data_to_control/
2. Put your CSV data files in the folder defined by DATA_FOLDER below
3. Open this file press Run (F5 in  Spyder)

DATA FORMAT
-----------
Each CSV file must have these columns:

  El1_Resistance_mOhm_filt   arc resistance electrode 1 (mOhm)
  El2_Resistance_mOhm_filt   arc resistance electrode 2 (mOhm)
  El3_Resistance_mOhm_filt   arc resistance electrode 3 (mOhm)
  El1_pos_m                  electrode 1 position (m)
  El2_pos_m                  electrode 2 position (m)
  El3_pos_m                  electrode 3 position (m)
  El1_kA_filt                arc current electrode 1 (kA)
  El2_kA_filt                arc current electrode 2 (kA)
  El3_kA_filt                arc current electrode 3 (kA)
  El1_CalcReac_filt          arc reactance electrode 1 (mOhm)
  El2_CalcReac_filt          arc reactance electrode 2 (mOhm)
  El3_CalcReac_filt          arc reactance electrode 3 (mOhm)
  RMS_V_transformer_filt     transformer RMS voltage (V)
  TCA                        transformerstep position A
  TCB                        transformerstep position B
  TCC                        transformerstep position C

If you have data with different column names, update COLUMN_MAP below.
The model trains on all files in DATA_FOLDER and evaluates on the last 20%.

EXPECTED RUNTIME
----------------
  ~140k rows: 2-5 minutes
  ~500k rows: 5-15 minutes
"""
from __future__ import annotations

# NB! Configure this before you run the script
#============================================================================
# Folder containing your CSV data files.
# Put one or more .csv files here, they will be chained together.
# Relative to SAF_data_to_control.
DATA_FOLDER = "fusion/data"

# Where to save the trained model. Leave as-is to replace the default model.
MODEL_OUT = "fusion/models/arx_joint_txt2026.joblib"

# Fraction of data to hold out for evaluation (last HOLDOUT_FRAC of the data).
HOLDOUT_FRAC = 0.20

# Column name mapping. Change right-hand side if your CSV uses different names.
# Format: internal_name -> your_csv_column_name
COLUMN_MAP = {
    "r":    "El{i}_Resistance_mOhm_filt",   # {i} replaced by 1, 2, 3
    "pos":  "El{i}_pos_m",
    "ka":   "El{i}_kA_filt",
    "reac": "El{i}_CalcReac_filt",
    "v":    "RMS_V_transformer_filt",
    "tca":  "TCA",
    "tcb":  "TCB",
    "tcc":  "TCC",
}

# no need to change things below this line
# ==========================================================================

import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

_HERE         = Path(__file__).resolve().parent        # fusion/
_PROJECT_ROOT = _HERE.parent                           # SAF_data_to_control/
for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.training.arx_model import ReducedRankRidge


#                       Signal processing
#------------------------------------------------------------------
_FS = 1.0; _FC = 0.1


def _lp(arr: np.ndarray, order: int = 4) -> np.ndarray:
    b, a = butter(order, _FC / (_FS / 2), btype="low", analog=False)
    s = pd.Series(arr).interpolate(limit=5).ffill().bfill().values
    return filtfilt(b, a, s, method="gust")


def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k, arr[0]), arr[:-k]])


# Data loading
def _col(template: str, i: int) -> str:
    return template.replace("{i}", str(i))


def load_data(data_folder: str | Path) -> pd.DataFrame:
    folder = Path(data_folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}\n"
            "Put your data files there and re-run."
        )
    print(f"Loading {len(csv_files)} CSV file(s) from {folder}:")
    frames = []
    for f in csv_files:
        print(f"  {f.name} ...", end=" ", flush=True)
        df = pd.read_csv(f, low_memory=False)
        df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        frames.append(df)
        print(f"{len(df):,} rows")
    raw = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(raw):,} rows")
    return raw


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    n   = len(raw)
    out: dict[str, np.ndarray] = {}

    # Transformer voltage
    v_col = COLUMN_MAP["v"]
    v_f   = np.clip(_lp(raw[v_col].values.astype(np.float64)), 0.0, None)
    out["RMS_V_transformer_filt_lag1"] = _lag(v_f, 1)
    out["V_target"] = v_f

    # Transformer steps
    for name, key in [("TCA", "tca"), ("TCB", "tcb"), ("TCC", "tcc")]:
        out[name] = raw[COLUMN_MAP[key]].values.astype(np.float64)

    for i in (1, 2, 3):
        p = f"El{i}"

        pos    = raw[_col(COLUMN_MAP["pos"], i)].values.astype(np.float64)
        dpos   = np.concatenate([[0.0], np.diff(pos)])
        dpos_f = _lp(dpos)
        out[f"{p}_dpos_mps_filt_lag1"] = _lag(dpos_f, 1)
        out[f"{p}_dpos_mps_filt_lag2"] = _lag(dpos_f, 2)
        out[f"{p}_dpos_mps_filt_lag3"] = _lag(dpos_f, 3)
        out[f"{p}_pos_m_lag1"]         = _lag(pos, 1)

        r_raw = raw[_col(COLUMN_MAP["r"], i)].values.astype(np.float64)
        y_f   = _lp(r_raw, order=2)
        out[f"{p}_y_filt_lag1"] = _lag(y_f, 1)
        out[f"{p}_y_filt_lag2"] = _lag(y_f, 2)
        out[f"{p}_y_filt_lag3"] = _lag(y_f, 3)
        out[f"{p}_R_target"]    = y_f

        ka_f = np.clip(_lp(raw[_col(COLUMN_MAP["ka"], i)].values.astype(np.float64)), 0.0, None)
        out[f"{p}_kA_filt_lag1"] = _lag(ka_f, 1)
        out[f"{p}_kA_filt_lag2"] = _lag(ka_f, 2)
        out[f"{p}_kA_filt_lag3"] = _lag(ka_f, 3)
        out[f"{p}_kA_target"]    = ka_f

        reac_raw = np.clip(raw[_col(COLUMN_MAP["reac"], i)].values.astype(np.float64), 0.0, 3.0)
        reac_f   = np.clip(_lp(reac_raw), 0.0, 3.0)
        out[f"{p}_CalcReac_filt_lag1"] = _lag(reac_f, 1)
        out[f"{p}_CalcReac_filt_lag2"] = _lag(reac_f, 2)
        out[f"{p}_CalcReac_filt_lag3"] = _lag(reac_f, 3)
        out[f"{p}_Reac_target"]        = reac_f

        print(f"  El{i}  R:[{y_f.min():.3f},{y_f.max():.3f}]  "
              f"kA:[{ka_f.min():.1f},{ka_f.max():.1f}]  "
              f"Reac:[{reac_f.min():.3f},{reac_f.max():.3f}] mOhm")

    df = pd.DataFrame(out)
    df = df.iloc[3:].reset_index(drop=True)   # remove lag warm-up rows
    print(f"  {len(df):,} usable rows after lag warm-up")
    return df


def _r_xcols(i: int) -> list[str]:
    other = [j for j in (1, 2, 3) if j != i]
    return [
        f"El{i}_dpos_mps_filt_lag1", f"El{i}_dpos_mps_filt_lag2", f"El{i}_dpos_mps_filt_lag3",
        f"El{i}_pos_m_lag1",
        f"El{i}_y_filt_lag1", f"El{i}_y_filt_lag2", f"El{i}_y_filt_lag3",
        f"El{i}_kA_filt_lag1", f"El{i}_kA_filt_lag2", f"El{i}_kA_filt_lag3",
        "RMS_V_transformer_filt_lag1",
        ["TCA", "TCB", "TCC"][i - 1],
        f"El{i}_CalcReac_filt_lag1", f"El{i}_CalcReac_filt_lag2", f"El{i}_CalcReac_filt_lag3",
        f"El{other[0]}_kA_filt_lag1",
        f"El{other[1]}_kA_filt_lag1",
    ]


def _ka_xcols(i: int) -> list[str]:
    return [
        f"El{i}_dpos_mps_filt_lag1", f"El{i}_dpos_mps_filt_lag2", f"El{i}_dpos_mps_filt_lag3",
        f"El{i}_pos_m_lag1",
        f"El{i}_y_filt_lag1", f"El{i}_y_filt_lag2", f"El{i}_y_filt_lag3",
        f"El{i}_kA_filt_lag1", f"El{i}_kA_filt_lag2", f"El{i}_kA_filt_lag3",
        ["TCA", "TCB", "TCC"][i - 1],
        f"El{i}_CalcReac_filt_lag1", f"El{i}_CalcReac_filt_lag2",
    ]


def _reac_xcols(i: int) -> list[str]:
    return [
        f"El{i}_dpos_mps_filt_lag1", f"El{i}_dpos_mps_filt_lag2",
        f"El{i}_pos_m_lag1",
        f"El{i}_kA_filt_lag1", f"El{i}_kA_filt_lag2", f"El{i}_kA_filt_lag3",
        "RMS_V_transformer_filt_lag1",
        ["TCA", "TCB", "TCC"][i - 1],
        f"El{i}_CalcReac_filt_lag1", f"El{i}_CalcReac_filt_lag2", f"El{i}_CalcReac_filt_lag3",
    ]


_V_XCOLS = [
    "TCA", "TCB", "TCC",
    "El1_kA_filt_lag1", "El1_y_filt_lag1", "El1_pos_m_lag1",
    "RMS_V_transformer_filt_lag1",
]

_Y_COLS = (
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


def _joint_xcols() -> list[str]:
    cols: set[str] = set()
    for i in (1, 2, 3):
        cols.update(_r_xcols(i))
        cols.update(_ka_xcols(i))
        cols.update(_reac_xcols(i))
    cols.update(_V_XCOLS)
    return sorted(cols)



#                           Training
# ----------------------------------------------------------------------------

def train_arx(df: pd.DataFrame, x_cols: list[str], out_path: Path) -> dict:
    data  = df[x_cols + _Y_COLS].dropna()
    n     = len(data)
    split = int(n * (1.0 - HOLDOUT_FRAC))

    X_tr = data[x_cols].iloc[:split].to_numpy(dtype=np.float64)
    Y_tr = data[_Y_COLS].iloc[:split].to_numpy(dtype=np.float64)
    X_te = data[x_cols].iloc[split:].to_numpy(dtype=np.float64)
    Y_te = data[_Y_COLS].iloc[split:].to_numpy(dtype=np.float64)

    print(f"  Train: {len(X_tr):,} rows   Holdout: {len(X_te):,} rows")

    x_sc = StandardScaler().fit(X_tr)
    y_sc = StandardScaler().fit(Y_tr)

    X_tr_z = x_sc.transform(X_tr)
    X_te_z = x_sc.transform(X_te)
    Y_tr_z = y_sc.transform(Y_tr)

    model = ReducedRankRidge(alphas=np.logspace(-4, 4, 20),
                             ranks=(2, 3, 4, 5, 6, 8, 10), cv=5)
    print("  Fitting ReducedRankRidge (cross-validating alpha and rank) ...")
    model.fit(X_tr_z, Y_tr_z)

    Yhat_te = y_sc.inverse_transform(model.predict(X_te_z))
    print(f"  {'Output':<28}  {'MAE':>8}  {'RMSE':>8}")
    for j, col in enumerate(_Y_COLS):
        mae  = float(np.mean(np.abs(Y_te[:, j] - Yhat_te[:, j])))
        rmse = float(np.sqrt(np.mean((Y_te[:, j] - Yhat_te[:, j]) ** 2)))
        print(f"  {col:<28}  {mae:8.5f}  {rmse:8.5f}")

    bundle = dict(
        model_name = "arx_joint_v3_rrr",
        X_cols     = x_cols,
        y_cols     = _Y_COLS,
        y_index    = _Y_INDEX,
        model      = model,
        Y_scaler   = y_sc,
        X_scaler   = x_sc,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"\nSaved -> {out_path}")
    return bundle


#                              Main 
#  (Not sure if this works in Spyder, if it wont, just uncomment the else in __name__==__main__)
# ----------------------------------------------------------------------------------------------
def main():
    print("=" * 60+"\nJoint ARX training\n"+"=" * 60)
    print(f"Data folder : {DATA_FOLDER}")
    print(f"Model out   : {MODEL_OUT}")

    print("\n--- Loading data ---")
    raw = load_data(_PROJECT_ROOT / DATA_FOLDER)

    print("\n--- Building features ---")
    df = build_features(raw)
    del raw

    x_cols = _joint_xcols()
    print(f"\n--- Training ({len(x_cols)} features -> {len(_Y_COLS)} outputs) ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_arx(df, x_cols, _PROJECT_ROOT / MODEL_OUT)

    print("\nDone.")
    print("Next step: run train_gp.py to train the GP correction models.")

if __name__ == "__main__":
    main()
#else:
#    main()

