from __future__ import annotations

import numpy as np

from .arx_state import ArxState
from .controller import PIDController


def run_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    controllers: list[PIDController],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run closed-loop simulation for 3-output VARX-current model.

    Args:
        model: trained bundle (must contain y_cols, X_cols, scalers, model)
        state: ArxState initialized from history CSV
        reference: array shape (n, 3) or (n,) (will broadcast to 3)
        controllers: list of 3 PIDController instances

    Returns:
        y: (n+1, 3) predicted currents
        u: (n, 3) commanded positions
        e: (n, 3) errors
    """

    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.shape[1] != 3:
        raise ValueError(f"reference must have 3 columns, got shape {reference.shape}")

    n = reference.shape[0]

    y = np.zeros((n + 1, 3), dtype=float)
    u = np.zeros((n, 3), dtype=float)
    e = np.zeros((n, 3), dtype=float)

    y_cols = model.get("y_cols") or [model.get("y_col")]
    y[0] = state.current_y(y_cols=y_cols)

    u_prev = state.current_u()  # defaults to 3 positions

    if len(controllers) != 3:
        raise ValueError("controllers must be a list of 3 PIDController instances")

    for c in controllers:
        c.reset()

    for k in range(n):
        y_pred = state.predict_next_y(model)  # (3,)

        u_cmd = np.zeros(3, dtype=float)
        for i in range(3):
            u_cmd[i], e[k, i] = controllers[i].step(
                reference=float(reference[k, i]),
                y_pred=float(y_pred[i]),
                u_prev=float(u_prev[i]),
            )

        u[k] = u_cmd
        y[k + 1] = y_pred

        state.advance(u_new=u_cmd, y_new=y_pred)
        u_prev = u_cmd

    return y, u, e
