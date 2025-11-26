from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# --------------------------- CONFIG ---------------------------
IN_CSV         = Path("data/1s_data_from_plant/0702_0703_1s_filtered.csv")
OUT_CSV        = Path("arx/arx_prep/model_arx_1_5_5.csv")
SCALERS_PATH   = Path("arx/arx_prep/model_arx_scalers_1_5_5.joblib")  # will write META next to this ...

# Target (filtered plant)
Y_FILT_COL     = "Tot_Resistance_mOhm_filt"

# Base exogenous signals (prefer *_filt if present)
EXOG_BASE = [
    "UL1N_V","UL2N_V","UL3N_V",
    "El1_pos_m","El2_pos_m","El3_pos_m",
    "El1_kA","El2_kA","El3_kA",
    "Tap_A","Tap_B","Tap_C",
    "RMS_V_transformer",
]

# Lags
MAX_AR_LAG     = 1
MAX_X_LAG      = 5

# Misc
FORCE_ELECTRODE_CM_TO_M = True
SAVE_INDEX_TIMESTAMP    = True

H = 1                # forecast horizon in seconds; 0 = predict y(t)
INCLUDE_U_T = False      # True = nowcast (use current inputs u(t)), False = strictly causal


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def main():
    assert IN_CSV.exists(), f"Missing input CSV: {IN_CSV}"
    ensure_parent(OUT_CSV)
    ensure_parent(SCALERS_PATH)

    # ---------- Load & basic prep ----------
    df = pd.read_csv(IN_CSV, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the input file.")
    df = df.sort_values("timestamp").set_index("timestamp")

    # ---------- Quick feature: RMS_V_transformer if missing ----------
    if "RMS_V_transformer" not in df.columns:
        volts = [c for c in ["UL1N_V","UL2N_V","UL3N_V"] if c in df.columns]
        if len(volts) == 3:
            df["RMS_V_transformer"] = df[volts].abs().mean(axis=1)

    # ---------- Electrode unit sanity (cm -> m) ----------
    if FORCE_ELECTRODE_CM_TO_M:
        for c in ["El1_pos_m","El2_pos_m","El3_pos_m"]:
            if c in df.columns:
                m = df[c].dropna().mean()
                if pd.notna(m) and m > 10.0:
                    df[c] = df[c] / 100.0

    # ---------- Check target column ----------
    if Y_FILT_COL not in df.columns:
        raise ValueError(f"Need '{Y_FILT_COL}' in the input data.")

    # ---------- Choose inputs (prefer *_filt if present, otherwise raw) ----------
    chosen_inputs = []
    for c in EXOG_BASE:
        if f"{c}_filt" in df.columns:
            chosen_inputs.append(f"{c}_filt")
        elif c in df.columns:
            chosen_inputs.append(c)
    if not chosen_inputs:
        raise ValueError("No exogenous inputs available (neither raw nor *_filt present).")

    # ---------- Target & target_time ----------
    # target(t) = y(t+H) if H>0 else y(t)
    target_col = "y_target" if H > 0 else Y_FILT_COL
    df[target_col] = df[Y_FILT_COL].shift(-H) if H > 0 else df[Y_FILT_COL]
    # target_time is when the label happens (useful for leakage-safe splits)
    df["target_time"] = (df.index + pd.to_timedelta(H, unit="s")) if H > 0 else df.index

    # ---------- Build lags (NO LEAKAGE) ----------
    lag_cols = {}

    # Output (AR) lags from UNshifted y (past only)
    y_raw = df[Y_FILT_COL].astype("float32")
    for L in range(1, MAX_AR_LAG + 1):
        lag_cols[f"y_raw_lag{L}"] = y_raw.shift(L)

    # Input (X) lags from their past only (+ optional contemporaneous u(t))
    for c in chosen_inputs:
        if INCLUDE_U_T:
            lag_cols[f"{c}_t0"] = df[c]
        for L in range(1, MAX_X_LAG + 1):
            lag_cols[f"{c}_lag{L}"] = df[c].shift(L)

    lag_df = pd.DataFrame(lag_cols, index=df.index)

    # ---------- Assemble & drop NaNs ----------
    # Note: this drops the first max lag rows and the last H rows
    df_all = pd.concat([df[[target_col, "target_time"]], lag_df], axis=1).dropna().copy()

    # ---------- Keep only what we need ----------
    feat_cols = sorted([c for c in lag_df.columns if c in df_all.columns])
    keep_cols = ["target_time", target_col] + feat_cols
    df_out = df_all[keep_cols]

    # ---------- Save ----------
    if SAVE_INDEX_TIMESTAMP:
        df_out = df_out.rename_axis("timestamp")

    df_out.to_csv(OUT_CSV, index=SAVE_INDEX_TIMESTAMP)

    # ---------- Save meta (no scalers here) ----------
    meta = {
        "y_col": target_col,
        "X_cols": feat_cols,
        "config": {
            "horizon": H,
            "max_ar_lag": MAX_AR_LAG,
            "max_x_lag": MAX_X_LAG,
            "included_base_inputs": chosen_inputs,
            "include_u_t": INCLUDE_U_T,
        },
    }
    meta_path = SCALERS_PATH.with_suffix(".meta.joblib")
    joblib.dump(meta, meta_path)

    print(f"[prep] Wrote {OUT_CSV}")
    print(f"[prep] Wrote meta -> {meta_path}")
    print(f"[prep] Target: {target_col} | Features: {len(feat_cols)} | Samples: {len(df_out)}")
    print(f"[prep] Inputs chosen: {chosen_inputs}")

if __name__ == "__main__":
    main()
