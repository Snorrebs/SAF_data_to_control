from __future__ import annotations

import numpy as np

from .arx_state import ArxState
from .controller_api import Controller


def run_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    controller: Controller,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float)
    n = len(reference)

    y = np.zeros(n + 1, dtype=float)
    u = np.zeros(n, dtype=float)
    e = np.zeros(n, dtype=float)

    y[0] = state.current_y()
    u_prev = state.current_u()

    controller.reset()
    for k in range(n):
        y_pred = state.predict_next_y(model)
        u_cmd, error = controller.step(reference=reference[k], y_pred=y_pred, u_prev=u_prev)

        u[k] = u_cmd
        e[k] = error

        state.advance(u_new=u_cmd, y_new=y_pred)
        y[k + 1] = y_pred
        u_prev = u_cmd

    return y, u, e
