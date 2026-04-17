#!/usr/bin/env python3
"""
make_synthetic_plant.py

Generates a synthetic oscillatory VARX plant bundle for 3 electrodes that
plugs directly into the arx_state.py / closed_loop_sim.py pipeline.

Plant dynamics (discrete-time, Ts = 1 s) — cross-coupled across electrodes:

    R_i(t) = Y_OP[i]
            + A1[i] * (R_i(t-1) - Y_OP[i])
            + A2[i] * (R_i(t-2) - Y_OP[i])
            + sum_j  B1[i,j] * (u_j(t-2) - U_OP[j])
            + sum_j  B2[i,j] * (u_j(t-3) - U_OP[j])
            + noise_i

Cross-coupling:
  - Diagonal B terms: negative  (lower electrode i -> lower R_i, direct effect)
  - Off-diagonal B terms: positive (lower electrode j -> raises R_i, redistribution)

Output
------
  run_simulation/models/synthetic_varx_plant.meta.joblib
  run_simulation/init_data/synthetic_varx_plant_init.csv

Usage
-----
  cd meta_arx/
  python -m run_simulation.scripts.make_synthetic_plant
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from noise import pnoise1
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ #
#  OUTPUT PATHS
# ------------------------------------------------------------------ #
OUT_DIR     = Path("run_simulation/models")
BUNDLE_PATH = OUT_DIR / "synthetic_varx_plant.meta.joblib"
INIT_CSV    = Path("run_simulation/init_data") / "synthetic_varx_plant_init.csv"

# ------------------------------------------------------------------ #
#  SIGNAL NAMES
# ------------------------------------------------------------------ #
Y_COLS  = ["El1_Resistance_mOhm_filt", "El2_Resistance_mOhm_filt", "El3_Resistance_mOhm_filt"]
U_BASES = ["El1_pos_m_filt",           "El2_pos_m_filt",           "El3_pos_m_filt"]
V_BASE  = "RMS_V_transformer_filt"

# ------------------------------------------------------------------ #
#  PLANT PARAMETERS
# ------------------------------------------------------------------ #
R_POLES = [0.96, 0.95, 0.97]
OMEGA_D = [0.40, 0.50, 0.35]

A1 = [2 * R_POLES[i] * np.cos(OMEGA_D[i]) for i in range(3)]
A2 = [-(R_POLES[i] ** 2)                   for i in range(3)]

# B1[i,j] = effect of electrode j position on electrode i resistance at lag 2
# Diagonal negative: lower own electrode -> lower own resistance
# Off-diagonal positive: lower other electrode -> raise own resistance (redistribution)
B1 = np.array([
    [-8.0,  2.5,  2.5],
    [ 2.5, -9.0,  2.5],
    [ 2.5,  2.5, -7.0],
], dtype=float)

B2 = np.array([
    [-3.0,  1.0,  1.0],
    [ 1.0, -3.5,  1.0],
    [ 1.0,  1.0, -2.5],
], dtype=float)

# Operating points
Y_OP = [300.0, 305.0, 295.0]   # mΩ
U_OP = [  0.50,  0.52,  0.48]  # m
V_OP = 400.0                    # V

NOISE_STD  = 1.0   # mΩ — per-electrode process noise
WARMUP     = 300
INIT_ROWS  = 100   # how many warmup rows to save in the init CSV for VRFT history
TS         = 1.0

# Perlin disturbance on transformer voltage
PERLIN_AMP   = 20.0   # V   — peak deviation from V_OP
PERLIN_SCALE = 0.01   # controls frequency: smaller = slower drift
PERLIN_OCT   = 4      # octaves: more = rougher texture
C_V          = 0.05   # mΩ/V — how much voltage deviation affects resistance

# ------------------------------------------------------------------ #
#  FEATURE COLUMN ORDER
# ------------------------------------------------------------------ #
def _make_x_cols() -> list[str]:
    cols: list[str] = []
    for u_base in U_BASES:
        for lag in (1, 2, 3):
            cols.append(f"{u_base}_lag{lag}")
    for y_col in Y_COLS:
        for lag in (1, 2):
            cols.append(f"{y_col}_lag{lag}")
    cols.append(f"{V_BASE}_lag1")
    return cols

X_COLS = _make_x_cols()


# ------------------------------------------------------------------ #
#  WARM-UP SIMULATION
# ------------------------------------------------------------------ #
def simulate_warmup(n: int, rng: np.random.Generator, v_hist: np.ndarray):
    y = np.array([np.full(n, Y_OP[i]) for i in range(3)], dtype=float)
    u = np.array([np.full(n, U_OP[i]) for i in range(3)], dtype=float)

    for i in range(3):
        du = rng.normal(0, 0.002, size=n).cumsum()
        u[i] += du
        u[i] = np.clip(u[i], U_OP[i] - 0.15, U_OP[i] + 0.15)

    for t in range(2, n):
        u_dev_2 = u[:, t - 2] - np.array(U_OP)
        u_dev_3 = u[:, t - 3] - np.array(U_OP) if t >= 3 else np.zeros(3)
        v_dev   = v_hist[t - 1] - V_OP            # voltage deviation at t-1
        for i in range(3):
            y[i, t] = (
                Y_OP[i]
                + A1[i] * (y[i, t - 1] - Y_OP[i])
                + A2[i] * (y[i, t - 2] - Y_OP[i])
                + B1[i] @ u_dev_2
                + B2[i] @ u_dev_3
                + C_V * v_dev                      # voltage disturbance (same on all phases)
                + rng.normal(0, NOISE_STD)
            )

    return y, u


# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
def main() -> None:
    rng = np.random.default_rng(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Warm-up simulation ----
    print("[synth] Simulating warm-up ...")

    # Perlin noise voltage disturbance — slow correlated drift around V_OP
    v_hist = np.array([
        V_OP + PERLIN_AMP * pnoise1(t * PERLIN_SCALE, octaves=PERLIN_OCT)
        for t in range(WARMUP)
    ])

    y_hist, u_hist = simulate_warmup(WARMUP, rng, v_hist)

    print(f"[synth] y_hist range: {y_hist.min():.1f} – {y_hist.max():.1f} mΩ")
    print(f"[synth] u_hist range: {u_hist.min():.3f} – {u_hist.max():.3f} m")

    # ---- 2. Build dataset ----
    rows_x = []
    rows_y = []

    for t in range(3, WARMUP):
        x_row: dict = {}
        for i, (u_base, y_col) in enumerate(zip(U_BASES, Y_COLS)):
            x_row[f"{u_base}_lag1"] = u_hist[i, t - 1]
            x_row[f"{u_base}_lag2"] = u_hist[i, t - 2]
            x_row[f"{u_base}_lag3"] = u_hist[i, t - 3]
            x_row[f"{y_col}_lag1"]  = y_hist[i, t - 1]
            x_row[f"{y_col}_lag2"]  = y_hist[i, t - 2]
        x_row[f"{V_BASE}_lag1"] = v_hist[t - 1]

        rows_x.append(x_row)
        rows_y.append({y_col: y_hist[i, t] for i, y_col in enumerate(Y_COLS)})

    df_x = pd.DataFrame(rows_x)[X_COLS]
    df_y = pd.DataFrame(rows_y)[Y_COLS]

    # ---- 3. Fit scalers ----
    X_scaler = StandardScaler()
    X_scaler.fit(df_x.values)

    Y_scaler = StandardScaler()
    Y_scaler.fit(df_y.values)

    X_z = X_scaler.transform(df_x.values)
    Y_z = Y_scaler.transform(df_y.values)

    sigma_x = X_scaler.scale_
    sigma_y = Y_scaler.scale_

    print(f"[synth] Y means : {Y_scaler.mean_}")
    print(f"[synth] Y stds  : {sigma_y}")

    # ---- 4. Build z-space coefficients ----
    base_ridge = Ridge(alpha=1.0)
    base_ridge.fit(X_z, Y_z[:, 0])

    estimators = []
    for i in range(3):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_z, Y_z[:, i])

        coef = np.zeros(len(X_COLS))
        for j, col in enumerate(X_COLS):
            # Input gains: all electrodes contribute to all outputs
            for k, u_base in enumerate(U_BASES):
                if col == f"{u_base}_lag2":
                    coef[j] = B1[i, k] * sigma_x[j] / sigma_y[i]
                elif col == f"{u_base}_lag3":
                    coef[j] = B2[i, k] * sigma_x[j] / sigma_y[i]

            # AR lags: diagonal only (each resistance depends on its own history)
            y_col = Y_COLS[i]
            if col == f"{y_col}_lag1":
                coef[j] = A1[i] * sigma_x[j] / sigma_y[i]
            elif col == f"{y_col}_lag2":
                coef[j] = A2[i] * sigma_x[j] / sigma_y[i]

        ridge.coef_          = coef
        ridge.intercept_     = 0.0
        ridge.n_features_in_ = len(X_COLS)
        estimators.append(ridge)

    multi_model = MultiOutputRegressor(base_ridge)
    multi_model.estimators_ = estimators
    multi_model.n_jobs = None

    # ---- 5. Sanity checks ----
    print("\n[synth] AR poles per electrode:")
    for i in range(3):
        ar_poly = np.array([1.0, -A1[i], -A2[i]])
        poles   = np.roots(ar_poly)
        print(f"  El{i+1} poles: {np.round(poles, 3)}  |poles|={np.round(np.abs(poles), 3)}")

    print("\n[synth] B1 gain matrix (row=output electrode, col=input electrode):")
    print(np.round(B1, 2))

    # ---- 6. Save bundle ----
    bundle = {
        "model_name": "synthetic_varx_plant",
        "y_cols":     Y_COLS,
        "X_cols":     X_COLS,
        "model":      multi_model,
        "X_scaler":   X_scaler,
        "Y_scaler":   Y_scaler,
        "plant_params": {
            "a1": A1, "a2": A2,
            "B1": B1.tolist(), "B2": B2.tolist(),
            "y_op": Y_OP, "u_op": U_OP,
            "noise_std": NOISE_STD,
            "Ts": TS,
        },
    }
    dump(bundle, BUNDLE_PATH)
    print(f"\n[synth] Bundle saved → {BUNDLE_PATH}")

    # ---- 7. Save init CSV ----
    # Save the last INIT_ROWS timesteps of warmup history so that VRFT
    # has enough run-up to fit conservative reference-model gains.
    # load_initial_state() will use the last row as the starting state —
    # the extra rows are there purely for VRFT to read further back in time.
    init_rows = []
    for t in range(WARMUP - INIT_ROWS, WARMUP):
        row: dict = {}
        for i, (u_base, y_col) in enumerate(zip(U_BASES, Y_COLS)):
            row[f"{u_base}_lag1"] = float(u_hist[i, t])
            row[f"{u_base}_lag2"] = float(u_hist[i, t - 1])
            row[f"{u_base}_lag3"] = float(u_hist[i, t - 2])
            row[f"{y_col}_lag1"]  = float(y_hist[i, t])
            row[f"{y_col}_lag2"]  = float(y_hist[i, t - 1])
            row[f"{y_col}_lag3"]  = float(y_hist[i, t - 2])
            row[f"{y_col}_lag4"]  = float(y_hist[i, t - 3])
            row[y_col]            = float(y_hist[i, t])
        row[f"{V_BASE}_lag1"] = float(v_hist[t])
        init_rows.append(row)

    init_df = pd.DataFrame(init_rows)
    init_df.to_csv(INIT_CSV, index=False)
    print(f"[synth] Init CSV saved → {INIT_CSV}  ({len(init_df)} rows, last row = starting state)")

    # ---- 8. Step response check: step u1 only, observe cross-coupling ----
    print("\n[synth] Step response: u1 += 0.05 m from t=5, u2/u3 held at U_OP:")
    y_check = np.array([[Y_OP[i]] * 30 for i in range(3)], dtype=float)
    u_check = np.array([[U_OP[i]] * 30 for i in range(3)], dtype=float)
    u_check[0, 5:] = U_OP[0] + 0.05

    for t in range(2, 30):
        u_dev_2 = u_check[:, t - 2] - np.array(U_OP)
        u_dev_3 = u_check[:, t - 3] - np.array(U_OP) if t >= 3 else np.zeros(3)
        for i in range(3):
            y_check[i, t] = (
                Y_OP[i]
                + A1[i] * (y_check[i, t - 1] - Y_OP[i])
                + A2[i] * (y_check[i, t - 2] - Y_OP[i])
                + B1[i] @ u_dev_2
                + B2[i] @ u_dev_3
            )

    for t in range(0, 30, 5):
        vals = "  ".join(f"El{i+1}={y_check[i,t]:.2f} mΩ" for i in range(3))
        print(f"  t={t:3d}  {vals}")
    print("  → El1 should rise, El2 and El3 should fall slightly")


if __name__ == "__main__":
    main()