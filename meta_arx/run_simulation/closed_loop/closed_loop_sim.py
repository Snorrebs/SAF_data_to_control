from __future__ import annotations

import numpy as np

from .arx_state import ArxState
from .controller_api import Controller


# Per-electrode rate limit (shared across all electrodes)
_DU_MAX = 0.01   # max movement per step [m/step]


def apply_rate_limit(du_des: float) -> float:
    """Clip requested movement to physical rate limit."""
    return float(np.clip(du_des, -_DU_MAX, _DU_MAX))


def run_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    controllers: list[Controller],
    exog_traj: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run closed-loop simulation for a 3-output VARX model.

    The control signal is Δu (holder movement per step [m/step]), which is
    what the VARX model was trained on. There is no absolute position tracking.

    Args:
        model:       trained bundle (y_cols, X_cols, scalers, models)
        state:       ArxState initialised from history CSV
        reference:   shape (n,) broadcast or (n, 3) per-electrode [mΩ]
        controllers: list of 3 Controller instances (one per electrode)
        exog_traj:   optional exogenous signal trajectories of length n

    Returns:
        y:  (n+1, 3) predicted outputs      [mΩ]
        du: (n,   3) applied movements      [m/step]
        e:  (n,   3) tracking errors        [mΩ]
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.shape[1] != 3:
        raise ValueError(f"reference must have 3 columns, got {reference.shape}")

    n = reference.shape[0]

    y  = np.zeros((n + 1, 3), dtype=float)
    du = np.zeros((n,     3), dtype=float)
    e  = np.zeros((n,     3), dtype=float)

    y_cols = model.get("y_cols") or [model.get("y_col")]
    y[0]   = state.current_y(y_cols=y_cols)

    if len(controllers) != 3:
        raise ValueError(f"Expected 3 controllers, got {len(controllers)}")

    for c in controllers:
        c.reset()

    for k in range(n):
        y_pred = state.predict_next_y(model)

        du_cmd = np.zeros(3, dtype=float)
        for i in range(3):
            du_des, e[k, i] = controllers[i].step(
                reference=float(reference[k, i]),
                y_pred=float(y_pred[i]),
                u_prev=0.0,     # controllers operate on error, not position
            )
            du_cmd[i] = apply_rate_limit(du_des)

        du[k]     = du_cmd
        y[k + 1]  = y_pred

        exog_k: dict[str, float] | None = None
        if exog_traj:
            exog_k = {sig: float(traj[k]) for sig, traj in exog_traj.items()}

        # Pass Δu directly — this is what the model's lag columns store
        state.advance(u_new=du_cmd, y_new=y_pred, exog_new=exog_k)

    return y, du, e


def run_mpc_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    mpc,
    exog_traj: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run closed-loop simulation with a MIMO MPC controller.

    The MPC optimises over Δu (movements) and returns Δu directly.
    state.advance() receives Δu — consistent with how the VARX model
    was identified (Δpos as control input, not absolute position).

    Args:
        model:     trained bundle (y_cols, X_cols, scalers, models)
        state:     ArxState initialised from history CSV
        reference: (n,) broadcast or (n, 3) per-electrode reference [mΩ]
        mpc:       LinearMPC instance
        exog_traj: optional exogenous signal trajectories of length n

    Returns:
        y:  (n+1, 3) predicted outputs   [mΩ]
        du: (n,   3) applied movements   [m/step]
        e:  (n,   3) tracking errors     [mΩ]
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.shape[1] != 3:
        raise ValueError(f"reference must have 3 columns, got {reference.shape}")

    n = reference.shape[0]
    N = mpc.params.N

    y  = np.zeros((n + 1, 3), dtype=float)
    du = np.zeros((n,     3), dtype=float)
    e  = np.zeros((n,     3), dtype=float)

    y_cols = model.get("y_cols") or [model.get("y_col")]
    y[0]   = state.current_y(y_cols=y_cols)

    # du_prev for R-penalty warm-starting in MPC (not used for position)
    du_prev = state.current_u()   # last Δu from init state (lag1 values)

    mpc.reset()

    for k in range(n):
        y_pred = state.predict_next_y(model)

        ref_window = reference[k : k + N]   # (≤N, 3); MPC pads internally

        # MPC returns Δu (movement) directly — no position reconstruction
        du_des, _ = mpc.step(ref_window, state, du_prev)

        # Apply rate limit (MPC bounds already enforce this, but clip for safety)
        du_cmd = np.clip(du_des, mpc.params.du_min, mpc.params.du_max)

        du[k]     = du_cmd
        y[k + 1]  = y_pred
        e[k]      = reference[k] - y_pred

        exog_k: dict[str, float] | None = None
        if exog_traj:
            exog_k = {sig: float(traj[k]) for sig, traj in exog_traj.items()}

        # Pass Δu directly to advance — correct for Δpos-trained model
        state.advance(u_new=du_cmd, y_new=y_pred, exog_new=exog_k)
        du_prev = du_cmd

    return y, du, e