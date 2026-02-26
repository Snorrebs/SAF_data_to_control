#!/usr/bin/env python3
"""
03_prep_for_arx.py  (VARX-current)

Build a 3-output VARX regression table from filtered plant data.

Targets (filtered):
  y(t) = [El1_kA_filt, El2_kA_filt, El3_kA_filt]

Features X (filtered, strictly causal lags):
  - AR coupling: lags of all three currents
  - Positions: lags of all three positions
  - Voltages: lags of UL1N/UL2N/UL3N
  - (Optional) Resistances: lags of El1/2/3 resistance (+ optional Tot resistance)

We:
- Build lagged columns from filtered signals (strictly causal).
- Drop rows with NaNs (edges).
- Save ARX dataset + meta with column lists & config.
"""

from pathlib import Path
import pandas as pd
import joblib

# --------------------------- CONFIG ---------------------------
IN_CSV    = Path("meta_arx/data/filt_data/07_24_filt_varx_current.csv")
OUT_CSV   = Path("meta_arx/arx/arx_prep_data/varx_curr_2321_07.csv")
META_PATH = Path("meta_arx/arx/arx_prep_meta/varx_curr_2321_07.meta.joblib")

TS_COL = "timestamp"

# Targets (filtered currents)
Y_FILT_COLS = ["El1_kA_filt", "El2_kA_filt", "El3_kA_filt"]

# Inputs (filtered)
POS_FILT_COLS = ["El1_pos_m_filt", "El2_pos_m_filt", "El3_pos_m_filt"]
V_FILT_COLS   = ["UL1N_V_filt", "UL2N_V_filt", "UL3N_V_filt"]

# Optional exogenous (filtered)
USE_RESISTANCE = True
R_FILT_COLS = [
    "El1_Resistance_mOhm_filt",
    "El2_Resistance_mOhm_filt",
    "El3_Resistance_mOhm_filt",
]
USE_TOT_RESISTANCE = True
TOT_R_FILT_COL = "Tot_Resistance_mOhm_filt"

# Lags (safe defaults)
P_I = 2   # current AR lags
Q_P = 2   # position lags
R_V = 1   # voltage lags
S_R = 1   # resistance lags (if enabled)

# Forecast horizon (H = 0 → predict y(t))
H = 0

# Whether to write timestamp index
SAVE_INDEX_TIMESTAMP = True


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns in {IN_CSV}: {missing}")


def main() -> None:
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    ensure_parent(OUT_CSV)
    ensure_parent(META_PATH)

    # ---------- Load & basic prep ----------
    df = pd.read_csv(IN_CSV, parse_dates=[TS_COL])
    if TS_COL not in df.columns:
        raise ValueError(f"Expected a '{TS_COL}' column in the input file.")
    df = df.sort_values(TS_COL).set_index(TS_COL)

    # ---------- Sanity checks ----------
    _require_cols(df, Y_FILT_COLS, "target (current)")
    _require_cols(df, POS_FILT_COLS, "position")
    _require_cols(df, V_FILT_COLS, "voltage")

    if USE_RESISTANCE:
        _require_cols(df, R_FILT_COLS, "resistance")
        if USE_TOT_RESISTANCE and TOT_R_FILT_COL not in df.columns:
            print(f"[warn] {TOT_R_FILT_COL} not found; disabling Tot resistance.")
            use_tot_r = False
        else:
            use_tot_r = USE_TOT_RESISTANCE
    else:
        use_tot_r = False

    # ---------- Target_time ----------
    if H == 0:
        df["target_time"] = df.index
        y_target_cols = Y_FILT_COLS
    else:
        # shift targets -H (future)
        y_target_cols = [c.replace("_filt", "_target") for c in Y_FILT_COLS]
        for src, dst in zip(Y_FILT_COLS, y_target_cols):
            df[dst] = df[src].shift(-H)
        df["target_time"] = df.index + pd.to_timedelta(H, unit="s")

    # ---------- Build lags (strictly causal) ----------
    lag_cols: dict[str, pd.Series] = {}

    # 1) AR coupling: current lags for all electrodes
    for c in Y_FILT_COLS:
        s = df[c].astype("float64")
        for L in range(1, P_I + 1):
            lag_cols[f"{c}_lag{L}"] = s.shift(L)

    # 2) positions
    for c in POS_FILT_COLS:
        s = df[c].astype("float64")
        for L in range(1, Q_P + 1):
            lag_cols[f"{c}_lag{L}"] = s.shift(L)

    # 3) voltages
    for c in V_FILT_COLS:
        s = df[c].astype("float64")
        for L in range(1, R_V + 1):
            lag_cols[f"{c}_lag{L}"] = s.shift(L)

    # 4) resistances (optional)
    if USE_RESISTANCE:
        for c in R_FILT_COLS:
            s = df[c].astype("float64")
            for L in range(1, S_R + 1):
                lag_cols[f"{c}_lag{L}"] = s.shift(L)

        if use_tot_r:
            s = df[TOT_R_FILT_COL].astype("float64")
            for L in range(1, S_R + 1):
                lag_cols[f"{TOT_R_FILT_COL}_lag{L}"] = s.shift(L)

    lag_df = pd.DataFrame(lag_cols, index=df.index)

    # ---------- Assemble & drop NaNs ----------
    keep_y = y_target_cols
    df_all = pd.concat([df[["target_time"] + keep_y], lag_df], axis=1).dropna().copy()

    # ---------- Feature list ----------
    feat_cols = list(lag_df.columns)
    keep_cols = ["target_time"] + keep_y + feat_cols
    df_varx = df_all[keep_cols].copy()

    # ---------- Save dataset ----------
    if SAVE_INDEX_TIMESTAMP:
        df_varx = df_varx.rename_axis("timestamp")
    df_varx.to_csv(OUT_CSV, index=SAVE_INDEX_TIMESTAMP)

    # ---------- Save meta ----------
    meta = {
        "y_cols": keep_y,
        "X_cols": feat_cols,
        "config": {
            "horizon": H,
            "lags": {
                "currents_p": P_I,
                "positions_q": Q_P,
                "voltages_r": R_V,
                "resistances_s": (S_R if USE_RESISTANCE else 0),
            },
            "signals": {
                "y_filt_cols": Y_FILT_COLS,
                "pos_filt_cols": POS_FILT_COLS,
                "v_filt_cols": V_FILT_COLS,
                "r_filt_cols": (R_FILT_COLS if USE_RESISTANCE else []),
                "tot_r_filt_col": (TOT_R_FILT_COL if (USE_RESISTANCE and use_tot_r) else None),
            },
            "use_resistance": USE_RESISTANCE,
            "use_tot_resistance": (USE_RESISTANCE and use_tot_r),
        },
    }
    joblib.dump(meta, META_PATH)

    print(f"[prep] Wrote VARX dataset -> {OUT_CSV}")
    print(f"[prep] Wrote meta -> {META_PATH}")
    print(f"[prep] Targets: {keep_y}")
    print(f"[prep] Features: {len(feat_cols)} | Samples: {len(df_varx)}")


if __name__ == "__main__":
    main()
