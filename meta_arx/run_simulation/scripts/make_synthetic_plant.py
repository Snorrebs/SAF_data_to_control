#!/usr/bin/env python3
"""
make_synthetic_plant.py

Generates a synthetic oscillatory ARX plant bundle that plugs directly into
the existing arx_state.py / closed_loop_sim.py pipeline with zero modifications.

Plant dynamics (discrete-time, Ts = 1 s):
----------------------------------------------
  y(t) = a1*y(t-1) + a2*y(t-2)
        + b1*u(t-2) + b2*u(t-3)
        + noise

Poles chosen to be underdamped (oscillatory) but stable:
  r       = 0.92   (decay per sample)
  omega_d = 0.35   rad/sample (~0.056 Hz at Ts=1s)

  a1 = 2*r*cos(omega_d)  ~  1.686
  a2 = -r^2              ~ -0.846

Input gain is strong so u->y is clearly excitable.

Output
------
  synthetic_plant.meta.joblib   -- drop-in ARX bundle (uses real Ridge object)
  synthetic_plant_init.csv      -- seed history for load_initial_state()

Usage
-----
  cd meta_arx/
  python -m run_simulation.scripts.make_synthetic_plant

Then in run_closed_loop.py change:
  MODEL_PATH = Path("run_simulation/models/synthetic_plant.meta.joblib")
  HIST_CSV   = Path("run_simulation/init_data/synthetic_plant_init.csv")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ #
#  OUTPUT PATHS
# ------------------------------------------------------------------ #
OUT_DIR     = Path("run_simulation/models")
BUNDLE_PATH = OUT_DIR / "synthetic_plant_ocsillatory.meta.joblib"
INIT_CSV    = Path("run_simulation/init_data") / "synthetic_plant_ocsillatory_init.csv"

# ------------------------------------------------------------------ #
#  SIGNAL NAMES  (must match ModelIOConfig used in run_closed_loop)
# ------------------------------------------------------------------ #
Y_COL      = "El1_Resistance_mOhm_filt"
U_BASE     = "El1_pos_m"
Y_LAG_BASE = "y_filt"
I1_BASE    = "El1_kA_filt"
V_BASE     = "RMS_V_transformer_filt"

# ------------------------------------------------------------------ #
#  PLANT PARAMETERS
# ------------------------------------------------------------------ #
R       = 0.97      # was 0.92 — much slower decay, oscillation persists longer
OMEGA_D = 0.5       # was 0.35 — faster oscillation (~0.08 Hz)
B1      = 15.0      # was 8.0
B2      =  6.0      # was 3.0
A1      = 2 * R * np.cos(OMEGA_D)   # ~1.686
A2      = -(R ** 2)                  # ~-0.846

# Operating point (physical units)
Y_OP     = 250.0   # mOhm
U_OP     =   0.5   # m
I1_OP    =  80.0   # kA
V_OP     = 400.0   # V

NOISE_STD = 0.5    # mOhm
WARMUP    = 200
TS        = 1.0

# ------------------------------------------------------------------ #
#  FEATURE COLUMN ORDER
# ------------------------------------------------------------------ #
X_COLS = [
    f"{I1_BASE}_lag1",
    f"{I1_BASE}_lag2",
    f"{U_BASE}_lag1",
    f"{U_BASE}_lag2",
    f"{U_BASE}_lag3",
    f"{V_BASE}_lag1",
    f"{Y_LAG_BASE}_lag1",
    f"{Y_LAG_BASE}_lag2",
]


# ------------------------------------------------------------------ #
#  WARM-UP SIMULATION
# ------------------------------------------------------------------ #
def simulate_warmup(n: int, rng: np.random.Generator):
    y = np.full(n, Y_OP)
    u = np.full(n, U_OP)

    du = rng.normal(0, 0.002, size=n).cumsum()
    u += du
    u = np.clip(u, U_OP - 0.1, U_OP + 0.1)

    for t in range(2, n):
        u3 = u[t - 3] if t >= 3 else U_OP
        y[t] = (
            A1 * y[t - 1]
            + A2 * y[t - 2]
            + B1 * (u[t - 2] - U_OP)
            + B2 * (u3 - U_OP)
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
    y_hist, u_hist = simulate_warmup(WARMUP, rng)
    i1_hist = np.full(WARMUP, I1_OP) + rng.normal(0, 1.0, WARMUP)
    v_hist  = np.full(WARMUP, V_OP)  + rng.normal(0, 5.0, WARMUP)

    # ---- 2. Build dataset for scaler fitting ----
    rows = []
    for t in range(3, WARMUP):
        rows.append({
            Y_COL:                  y_hist[t],
            f"{Y_LAG_BASE}_lag1":   y_hist[t - 1],
            f"{Y_LAG_BASE}_lag2":   y_hist[t - 2],
            f"{U_BASE}_lag1":       u_hist[t - 1],
            f"{U_BASE}_lag2":       u_hist[t - 2],
            f"{U_BASE}_lag3":       u_hist[t - 3],
            f"{I1_BASE}_lag1":      i1_hist[t - 1],
            f"{I1_BASE}_lag2":      i1_hist[t - 2],
            f"{V_BASE}_lag1":       v_hist[t - 1],
        })

    df_fit = pd.DataFrame(rows)

    # ---- 3. Fit scalers ----
    y_scaler = StandardScaler()
    y_scaler.fit(df_fit[[Y_COL]].values)

    x_scaler = StandardScaler()
    x_scaler.fit(df_fit[X_COLS].values)

    sigma_x = x_scaler.scale_
    sigma_y = float(y_scaler.scale_[0])

    print(f"[synth] y  : mean={float(y_scaler.mean_[0]):.2f}  std={sigma_y:.4f}")
    print(f"[synth] u_lag1: mean={x_scaler.mean_[X_COLS.index(f'{U_BASE}_lag1')]:.4f}")

    # ---- 4. Build z-space coefficients ----
    coef = np.zeros(len(X_COLS))
    for j, col in enumerate(X_COLS):
        if col == f"{Y_LAG_BASE}_lag1":
            coef[j] = A1 * sigma_x[j] / sigma_y
        elif col == f"{Y_LAG_BASE}_lag2":
            coef[j] = A2 * sigma_x[j] / sigma_y
        elif col == f"{U_BASE}_lag2":
            coef[j] = B1 * sigma_x[j] / sigma_y
        elif col == f"{U_BASE}_lag3":
            coef[j] = B2 * sigma_x[j] / sigma_y
        # u_lag1, I1 lags, V lag -> zero (no effect)

    # ---- 5. Stuff coefficients into a real Ridge object ----
    model = Ridge(alpha=1.0)
    model.coef_          = coef
    model.intercept_     = 0.0
    model.n_features_in_ = len(X_COLS)

    # ---- 6. Sanity checks ----
    ar_poly = np.array([1.0, -A1, -A2])
    poles   = np.roots(ar_poly)
    print(f"[synth] AR poles        : {poles}")
    print(f"[synth] Pole magnitudes : {np.abs(poles)}")
    print(f"[synth] coef (z-space)  : {dict(zip(X_COLS, np.round(coef, 4)))}")

    # ---- 7. Save bundle ----
    bundle = {
        "model_name": "synthetic_oscillatory_plant",
        "y_col":      Y_COL,
        "X_cols":     X_COLS,
        "model":      model,
        "X_scaler":   x_scaler,
        "y_scaler":   y_scaler,
        "plant_params": {
            "a1": A1, "a2": A2,
            "b1": B1, "b2": B2,
            "poles": poles.tolist(),
            "y_op": Y_OP, "u_op": U_OP,
            "noise_std": NOISE_STD,
            "Ts": TS,
        },
    }
    dump(bundle, BUNDLE_PATH)
    print(f"[synth] Bundle saved -> {BUNDLE_PATH}")

    # ---- 8. Save init CSV ----
    init_row = {col: float(x_scaler.mean_[j]) for j, col in enumerate(X_COLS)}
    init_row[Y_COL] = float(y_scaler.mean_[0])
    init_df = pd.DataFrame([init_row])
    init_df.to_csv(INIT_CSV, index=False)

    # ---- 9. Open-loop step response check ----
    print("\n[synth] Step response check (u += 0.05 m from t=5):")
    y_check = np.full(30, Y_OP)
    u_check = np.full(30, U_OP)
    u_check[5:] = U_OP + 0.05

    for t in range(2, 30):
        u3 = u_check[t - 3] if t >= 3 else U_OP
        y_check[t] = (
            A1 * y_check[t - 1]
            + A2 * y_check[t - 2]
            + B1 * (u_check[t - 2] - U_OP)
            + B2 * (u3 - U_OP)
        )

    for t in range(0, 30, 3):
        print(f"  t={t:3d}  y={y_check[t]:.3f} mOhm  u={u_check[t]:.4f} m")


if __name__ == "__main__":
    main()