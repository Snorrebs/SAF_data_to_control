# run_simulation/vrft/vrft_pid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class PIDParams:
    Kp: float
    Ki: float
    Kd: float


def first_order_alpha(tau_cl: float, Ts: float) -> float:
    """
    Compute alpha for a discrete first-order reference model M(z)
    with time constant tau_cl [s] and sampling time Ts [s]:

        M(z) = (1 - alpha) / (1 - alpha z^-1)

    where alpha = exp(-Ts/tau_cl).
    """
    if tau_cl <= 0:
        raise ValueError("tau_cl must be > 0")
    return float(np.exp(-Ts / tau_cl))


def virtual_reference(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute virtual reference r_v from output y for the model:

        M(z) = (1 - alpha) / (1 - alpha z^-1)

    Its inverse is:

        M^{-1}(z) = (1 - alpha z^-1)/(1 - alpha)

    -> r_v[k] = (1/(1-alpha))*y[k] - (alpha/(1-alpha))*y[k-1]
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N == 0:
        return y.copy()

    rv = np.zeros_like(y)
    denom = (1.0 - alpha)
    if np.isclose(denom, 0.0):
        raise ValueError("alpha too close to 1, choose shorter tau_cl")

    for k in range(N):
        y_k = y[k]
        y_km1 = y[k - 1] if k > 0 else y[0]
        rv[k] = (y_k - alpha * y_km1) / denom
    return rv


def build_vrft_regressors(
    y: np.ndarray,
    u: np.ndarray,
    Ts: float,
    tau_cl: float,
    use_integral: bool = True,
    use_derivative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (Phi, du) for incremental PID VRFT.

    We use:
        du[k] = u[k] - u[k-1]

    Virtual signals:
        r_v = M^{-1} y
        e_v[k] = r_v[k] - y[k]
        I_v[k] = sum_{j=0}^k e_v[j]*Ts
        d_v[k] = (y[k] - y[k-1]) / Ts

    Phi columns (depending on flags):
        [e_v, I_v, d_v]

    Returns:
        Phi: (N-1, n_params)
        du:  (N-1,)
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if len(y) != len(u):
        raise ValueError("y and u must have same length")

    N = len(y)
    if N < 3:
        raise ValueError("Need at least 3 samples for VRFT")

    alpha = first_order_alpha(tau_cl=tau_cl, Ts=Ts)
    r_v = virtual_reference(y, alpha=alpha)
    e_v = r_v - y

    # integrator
    I_v = np.cumsum(e_v) * Ts

    # derivative (on output, similar to your closed_loop_sim)
    d_v = np.zeros_like(y)
    d_v[1:] = (y[1:] - y[:-1]) / Ts

    # incremental control
    du = u[1:] - u[:-1]          # length N-1

    cols = [e_v[1:]]            # skip k=0 to align with du
    if use_integral:
        cols.append(I_v[1:])
    if use_derivative:
        cols.append(d_v[1:])

    Phi = np.vstack(cols).T     # shape (N-1, n_params)
    return Phi, du


def vrft_pid_from_csv(
    csv_path: str,
    y_col: str = "y_pred_mOhm",
    u_col: str = "u_El1_pos_m",
    Ts: float = 1.0,
    tau_cl: float = 300.0,
    use_integral: bool = True,
    use_derivative: bool = False,
) -> PIDParams:
    """
    High-level VRFT routine:

    1) Load CSV with columns y_col (output), u_col (input).
    2) Build regressor matrix Phi and du.
    3) Solve least squares: du ≈ Phi @ rho.
    4) Map rho -> PIDParams(Kp, Ki, Kd).

    tau_cl: desired closed-loop time constant in seconds (e.g. 300 s).

    Return:
        PIDParams with Kp, Ki, Kd (Kd=0 if use_derivative=False).
    """
    df = pd.read_csv(csv_path)
    if y_col not in df or u_col not in df:
        raise ValueError(f"CSV must contain '{y_col}' and '{u_col}'")

    y = df[y_col].to_numpy(dtype=float)
    u = df[u_col].to_numpy(dtype=float)

    Phi, du = build_vrft_regressors(
        y=y,
        u=u,
        Ts=Ts,
        tau_cl=tau_cl,
        use_integral=use_integral,
        use_derivative=use_derivative,
    )

    # Least-squares solution
    rho, *_ = np.linalg.lstsq(Phi, du, rcond=None)

    # Map to PIDParams
    if use_integral and use_derivative:
        Kp, Ki, Kd = rho
    elif use_integral and not use_derivative:
        Kp, Ki = rho
        Kd = 0.0
    elif not use_integral and use_derivative:
        Kp, Kd = rho
        Ki = 0.0
    else:
        Kp = rho[0]
        Ki = 0.0
        Kd = 0.0

    return PIDParams(Kp=float(Kp), Ki=float(Ki), Kd=float(Kd))
