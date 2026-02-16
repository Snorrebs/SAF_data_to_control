#!/usr/bin/env python3
"""
compare_step_responses_clean.py

Clean pulse response of ARX models using *consistent equilibrium*:

- Equilibrium is defined as z = 0 for all features => physical value = training mean.
- We therefore initialize all histories at their corresponding physical means.
- Apply a short pulse to electrode speed around its mean operating point.
- Keep exogenous signals (I, V) at their mean operating points (=> z=0).
- Predict in z-space, convert y back to physical, feed back in physical.

This avoids artificial drift/explosion caused by initializing at physical 0.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import load


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
MODEL_CONFIGS = [
    {"label": "AR(2)",  "path": Path("arx/models/model_meta") / "arx_el1res_2321_07.meta.joblib"},
    {"label": "AR(5)",  "path": Path("arx/models/model_meta") / "arx_el1res_5321_07.meta.joblib"},
    {"label": "AR(8)",  "path": Path("arx/models/model_meta") / "arx_el1res_8321_07.meta.joblib"},
    {"label": "AR(15)", "path": Path("arx/models/model_meta") / "arx_el1res_15321_07.meta.joblib"},
]

N_STEPS    = 1000
STEP_SIZE  = 0.00   # Δu in physical units (m/s) as a pulse
PULSE_LEN  = 1       # seconds duration of the pulse

U_BASENAME   = "El1_dpos_mps_filt"
I_BASENAME   = "El1_kA_filt"
V_BASENAME   = "RMS_V_transformer_filt"
Y_LAG_PREFIX = "y_filt_lag"


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _lag_num(name: str, prefix: str) -> int:
    # e.g. "y_filt_lag2" -> 2, "El1_dpos_mps_filt_lag3" -> 3
    return int(name.replace(prefix, ""))


def build_feature_vector_z_from_phys(
    X_cols,
    t: int,
    y_hist_phys: np.ndarray,
    u_hist_phys: np.ndarray,
    I_hist_phys: np.ndarray,
    V_hist_phys: np.ndarray,
    X_scaler,
) -> np.ndarray:
    """
    Build x_z(t) consistent with training: x_z[j] = (x_phys - mean_j) / scale_j.
    For t-L < 0, use equilibrium physical value = mean_j (=> z=0).
    """
    x_z = np.zeros(len(X_cols), dtype=float)

    for j, name in enumerate(X_cols):
        mu = float(X_scaler.mean_[j])
        sig = float(X_scaler.scale_[j])

        # default equilibrium physical value for this feature is its mean
        x_phys = mu

        if name.startswith(Y_LAG_PREFIX):
            L = _lag_num(name, Y_LAG_PREFIX)
            idx = t - L
            if idx >= 0:
                x_phys = float(y_hist_phys[idx])

        elif name.startswith(U_BASENAME + "_lag"):
            L = _lag_num(name, U_BASENAME + "_lag")
            idx = t - L
            if idx >= 0:
                x_phys = float(u_hist_phys[idx])

        elif name.startswith(I_BASENAME + "_lag"):
            L = _lag_num(name, I_BASENAME + "_lag")
            idx = t - L
            if idx >= 0:
                x_phys = float(I_hist_phys[idx])

        elif name.startswith(V_BASENAME + "_lag"):
            L = _lag_num(name, V_BASENAME + "_lag")
            idx = t - L
            if idx >= 0:
                x_phys = float(V_hist_phys[idx])

        else:
            # any other feature -> keep at equilibrium (mean) unless you later want to excite it
            x_phys = mu

        # standardize
        x_z[j] = (x_phys - mu) / sig if sig != 0 else 0.0

    return x_z


def simulate_pulse_response(bundle: Dict, horizon: int, step_u_phys: float, pulse_len: int) -> Tuple[np.ndarray, np.ndarray]:
    model    = bundle["model"]
    X_scaler = bundle["X_scaler"]
    y_scaler = bundle["y_scaler"]
    X_cols   = bundle["X_cols"]

    # Equilibrium (z=0) in physical units for y is y_scaler.mean_
    y_eq = float(y_scaler.mean_[0])

    # Initialize histories at equilibrium
    y_hist_phys = np.full(horizon, y_eq, dtype=float)

    # For u/I/V, equilibrium is "whatever makes their lag-columns z=0" -> use column means.
    # We'll construct u/I/V histories as physical values and let build_feature... standardize per-column.
    u_hist_phys = np.zeros(horizon, dtype=float)
    I_hist_phys = np.zeros(horizon, dtype=float)
    V_hist_phys = np.zeros(horizon, dtype=float)

    # Pick equilibrium physical values for u/I/V by taking the mean of the first matching lag column.
    def first_col_mean(prefix: str) -> float:
        j = next(i for i, n in enumerate(X_cols) if n.startswith(prefix))
        return float(X_scaler.mean_[j])

    u_eq = first_col_mean(U_BASENAME + "_lag")
    I_eq = first_col_mean(I_BASENAME + "_lag")
    V_eq = first_col_mean(V_BASENAME + "_lag")

    u_hist_phys[:] = u_eq
    I_hist_phys[:] = I_eq
    V_hist_phys[:] = V_eq

    # Apply pulse around equilibrium
    u_hist_phys[:pulse_len] = u_eq + step_u_phys
    u_hist_phys[pulse_len:] = u_eq

    # Simulate
    for t in range(horizon):
        x_z = build_feature_vector_z_from_phys(
            X_cols=X_cols,
            t=t,
            y_hist_phys=y_hist_phys,
            u_hist_phys=u_hist_phys,
            I_hist_phys=I_hist_phys,
            V_hist_phys=V_hist_phys,
            X_scaler=X_scaler,
        )

        # model output is in y-space z
        y_pred_z = float(model.predict(x_z.reshape(1, -1))[0])

        # convert to physical and feed back
        y_pred_phys = float(y_scaler.inverse_transform([[y_pred_z]])[0, 0])
        y_hist_phys[t] = y_pred_phys

    t_vec = np.arange(horizon)
    return t_vec, y_hist_phys


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    plt.figure(figsize=(10, 5))

    for cfg in MODEL_CONFIGS:
        bundle = load(cfg["path"])

        t, y = simulate_pulse_response(
            bundle=bundle,
            horizon=N_STEPS,
            step_u_phys=STEP_SIZE,
            pulse_len=PULSE_LEN,
        )

        # Δy relative to equilibrium baseline
        # dy = y - y[0]
        dy=y
        plt.plot(t, dy, label=cfg["label"])

    plt.axhline(y[0], color="k", linewidth=0.6)
    plt.xlabel("Time step (s)")
    plt.ylabel("Resistance [mΩ]")
    plt.title(f"Pulse in holder movement: Δu = {STEP_SIZE} m/s for {PULSE_LEN} s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path("arx/figures/step_pulse_response_clean.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"[fig] Saved -> {out}")


if __name__ == "__main__":
    main()
