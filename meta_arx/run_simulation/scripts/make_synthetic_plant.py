#!/usr/bin/env python3
"""
make_synthetic_plant.py

Generates a synthetic oscillatory VARX plant bundle for 3 electrodes that
plugs directly into the arx_state.py / closed_loop_sim.py pipeline.

Plant dynamics (discrete-time, Ts = 1 s) — identical structure per electrode,
with slight parameter variation so the three channels are distinguishable:

    y_i(t) = a1_i * y_i(t-1) + a2_i * y_i(t-2)
            + b1_i * u_i(t-2) + b2_i * u_i(t-3)
            + noise_i

Electrodes are independent (no cross-coupling) to keep the example simple.

Output
------
  run_simulation_PID/models/synthetic_varx_plant.meta.joblib
  run_simulation_PID/init_data/synthetic_varx_plant_init.csv

Usage
-----
  cd meta_arx/
  python -m run_simulation_PID.scripts.make_synthetic_plant

Then in run_closed_loop.py point MODEL_PATH and HIST_CSV at the files above.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
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
#  SIGNAL NAMES  (must match arx_state.py naming conventions)
# ------------------------------------------------------------------ #
Y_COLS  = ["El1_kA_filt",    "El2_kA_filt",    "El3_kA_filt"]
U_BASES = ["El1_pos_m_filt", "El2_pos_m_filt", "El3_pos_m_filt"]
V_BASE  = "RMS_V_transformer_filt"

# ------------------------------------------------------------------ #
#  PLANT PARAMETERS  (slight variation per electrode)
# ------------------------------------------------------------------ #
#  Poles: r_i * exp(±j * omega_i)  — all stable, all oscillatory
R       = [0.96, 0.95, 0.97]
OMEGA_D = [0.40, 0.50, 0.35]

A1 = [2 * R[i] * np.cos(OMEGA_D[i]) for i in range(3)]
A2 = [-(R[i] ** 2)                   for i in range(3)]

B1 = [12.0, 14.0, 10.0]   # input gain lag-2
B2 = [ 5.0,  6.0,  4.0]   # input gain lag-3

# Operating points (physical units)
Y_OP  = [80.0,  82.0,  78.0]   # kA
U_OP  = [ 0.50,  0.52,  0.48]  # m
V_OP  = 400.0                   # V

NOISE_STD = 0.3   # kA
WARMUP    = 300
TS        = 1.0

# ------------------------------------------------------------------ #
#  FEATURE COLUMN ORDER
#  All lag columns for all three electrodes + shared voltage lag
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
def simulate_warmup(n: int, rng: np.random.Generator):
    y = np.array([np.full(n, Y_OP[i]) for i in range(3)])   # (3, n)
    u = np.array([np.full(n, U_OP[i]) for i in range(3)])   # (3, n)

    # Small random walk on each electrode
    for i in range(3):
        du = rng.normal(0, 0.002, size=n).cumsum()
        u[i] += du
        u[i] = np.clip(u[i], U_OP[i] - 0.15, U_OP[i] + 0.15)

    for t in range(2, n):
        for i in range(3):
            u3 = u[i, t - 3] if t >= 3 else U_OP[i]
            y[i, t] = (
                A1[i] * y[i, t - 1]
                + A2[i] * y[i, t - 2]
                + B1[i] * (u[i, t - 2] - U_OP[i])
                + B2[i] * (u3           - U_OP[i])
                + rng.normal(0, NOISE_STD)
            )

    return y, u   # both (3, n)


# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
def main() -> None:
    rng = np.random.default_rng(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Warm-up simulation ----
    print("[synth] Simulating warm-up ...")
    y_hist, u_hist = simulate_warmup(WARMUP, rng)
    v_hist = np.full(WARMUP, V_OP) + rng.normal(0, 5.0, WARMUP)

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

    df_x = pd.DataFrame(rows_x)[X_COLS]   # enforce column order
    df_y = pd.DataFrame(rows_y)[Y_COLS]

    # ---- 3. Fit scalers ----
    X_scaler = StandardScaler()
    X_scaler.fit(df_x.values)

    Y_scaler = StandardScaler()
    Y_scaler.fit(df_y.values)

    X_z = X_scaler.transform(df_x.values)
    Y_z = Y_scaler.transform(df_y.values)

    sigma_x = X_scaler.scale_
    sigma_y = Y_scaler.scale_   # shape (3,)

    print(f"[synth] Y means : {Y_scaler.mean_}")
    print(f"[synth] Y stds  : {sigma_y}")

    # ---- 4. Build z-space coefficients (analytical, per electrode) ----
    #
    # For electrode i the true model in physical units is:
    #   y_i(t) = A1_i*y_i(t-1) + A2_i*y_i(t-2) + B1_i*(u_i(t-2)-U_OP_i) + B2_i*(u_i(t-3)-U_OP_i)
    #
    # In standardised space (z = (x - mu) / sigma):
    #   coef_j = physical_coef * sigma_x_j / sigma_y_i
    #
    # The MultiOutputRegressor wraps one Ridge per output, so we set
    # coef_ on each estimator independently.

    base_ridge = Ridge(alpha=1.0)
    # Fit once just to initialise internal sklearn state
    base_ridge.fit(X_z, Y_z[:, 0])

    estimators = []
    for i in range(3):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_z, Y_z[:, i])   # gives correct n_features_in_ etc.

        coef = np.zeros(len(X_COLS))
        for j, col in enumerate(X_COLS):
            u_base = U_BASES[i]
            y_col  = Y_COLS[i]
            if col == f"{u_base}_lag2":
                coef[j] = B1[i] * sigma_x[j] / sigma_y[i]
            elif col == f"{u_base}_lag3":
                coef[j] = B2[i] * sigma_x[j] / sigma_y[i]
            elif col == f"{y_col}_lag1":
                coef[j] = A1[i] * sigma_x[j] / sigma_y[i]
            elif col == f"{y_col}_lag2":
                coef[j] = A2[i] * sigma_x[j] / sigma_y[i]
            # all other lags (other electrodes, voltage) → zero

        ridge.coef_          = coef
        ridge.intercept_     = 0.0
        ridge.n_features_in_ = len(X_COLS)
        estimators.append(ridge)

    # Wrap in MultiOutputRegressor so model.predict() returns (n, 3)
    multi_model = MultiOutputRegressor(base_ridge)
    multi_model.estimators_ = estimators
    multi_model.n_jobs = None

    # ---- 5. Sanity check ----
    for i in range(3):
        ar_poly = np.array([1.0, -A1[i], -A2[i]])
        poles   = np.roots(ar_poly)
        print(f"[synth] El{i+1} poles: {np.round(poles, 3)}  |poles|={np.round(np.abs(poles), 3)}")

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
            "b1": B1, "b2": B2,
            "y_op": Y_OP, "u_op": U_OP,
            "noise_std": NOISE_STD,
            "Ts": TS,
        },
    }
    dump(bundle, BUNDLE_PATH)
    print(f"[synth] Bundle saved → {BUNDLE_PATH}")

    # ---- 7. Save init CSV ----
    #  One row containing the last warm-up state: all lag columns + y_cols
    init_row: dict = {}
    t = WARMUP - 1
    for i, (u_base, y_col) in enumerate(zip(U_BASES, Y_COLS)):
        init_row[f"{u_base}_lag1"] = float(u_hist[i, t])
        init_row[f"{u_base}_lag2"] = float(u_hist[i, t - 1])
        init_row[f"{u_base}_lag3"] = float(u_hist[i, t - 2])
        init_row[f"{y_col}_lag1"]  = float(y_hist[i, t])
        init_row[f"{y_col}_lag2"]  = float(y_hist[i, t - 1])
        init_row[y_col]            = float(y_hist[i, t])
    init_row[f"{V_BASE}_lag1"] = float(v_hist[t])

    init_df = pd.DataFrame([init_row])
    init_df.to_csv(INIT_CSV, index=False)
    print(f"[synth] Init CSV saved → {INIT_CSV}")

    # ---- 8. Open-loop step response check ----
    print("\n[synth] Step response check (u_i += 0.05 m from t=5):")
    y_check = np.array([[Y_OP[i]] * 30 for i in range(3)], dtype=float)
    u_check = np.array([[U_OP[i]] * 30 for i in range(3)], dtype=float)
    for i in range(3):
        u_check[i, 5:] = U_OP[i] + 0.05

    for t in range(2, 30):
        for i in range(3):
            u3 = u_check[i, t - 3] if t >= 3 else U_OP[i]
            y_check[i, t] = (
                A1[i] * y_check[i, t - 1]
                + A2[i] * y_check[i, t - 2]
                + B1[i] * (u_check[i, t - 2] - U_OP[i])
                + B2[i] * (u3                 - U_OP[i])
            )

    for t in range(0, 30, 5):
        vals = "  ".join(f"El{i+1}={y_check[i,t]:.2f} kA" for i in range(3))
        print(f"  t={t:3d}  {vals}")


if __name__ == "__main__":
    main()