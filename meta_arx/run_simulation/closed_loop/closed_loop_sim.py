from __future__ import annotations

import numpy as np

from .arx_state import ArxState
from .controller_api import Controller


# Per-electrode actuator limits (shared across all electrodes for now)
_DU_MAX = 0.01   # max position change per step [m]
_U_MIN  = 0.1    # minimum electrode position   [m]
_U_MAX  = 2.0    # maximum electrode position   [m]


def apply_actuator_limits(u_des: float, u_prev: float) -> float:
    """Simple actuator model: rate-limit then position-saturate."""
    du = np.clip(u_des - u_prev, -_DU_MAX, _DU_MAX)
    u = np.clip(u_prev + du, _U_MIN, _U_MAX)
    return float(u)


def run_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    controllers: list[Controller],
    exog_traj: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run closed-loop simulation for a 3-output VARX model.

    Args:
        model:       trained bundle (must contain y_cols, X_cols, scalers, model)
        state:       ArxState initialised from history CSV
        reference:   shape ``(n,)`` (broadcast to all 3 electrodes) or ``(n, 3)``
        controllers: list of 3 Controller instances (one per electrode)
        exog_traj:   optional dict of exogenous signal trajectories, e.g.
                     ``{"RMS_V_transformer_filt": np.array([...])}`` of length n.
                     Each signal is passed into advance() at every timestep so the
                     disturbance is live during simulation rather than frozen.

    Returns:
        y: ``(n+1, 3)`` predicted outputs   [mΩ]
        u: ``(n,   3)`` commanded positions [m]
        e: ``(n,   3)`` control errors      [mΩ]
    """

    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.shape[1] != 3:
        raise ValueError(f"reference must have 3 columns, got shape {reference.shape}")

    n = reference.shape[0]

    y = np.zeros((n + 1, 3), dtype=float)
    u = np.zeros((n,     3), dtype=float)
    e = np.zeros((n,     3), dtype=float)

    y_cols = model.get("y_cols") or [model.get("y_col")]
    y[0] = state.current_y(y_cols=y_cols)
    u_prev = state.current_u()

    if len(controllers) != 3:
        raise ValueError(f"Expected 3 controllers, got {len(controllers)}")

    for c in controllers:
        c.reset()

    for k in range(n):
        y_pred = state.predict_next_y(model)

        u_cmd = np.zeros(3, dtype=float)
        for i in range(3):
            u_des, e[k, i] = controllers[i].step(
                reference=float(reference[k, i]),
                y_pred=float(y_pred[i]),
                u_prev=float(u_prev[i]),
            )
            u_cmd[i] = apply_actuator_limits(u_des, float(u_prev[i]))

        u[k] = u_cmd
        y[k + 1] = y_pred

        # Build exog snapshot for this timestep if a trajectory was provided
        exog_k: dict[str, float] | None = None
        if exog_traj:
            exog_k = {sig: float(traj[k]) for sig, traj in exog_traj.items()}

        state.advance(u_new=u_cmd, y_new=y_pred, exog_new=exog_k)
        u_prev = u_cmd

    return y, u, e