from __future__ import annotations

import numpy as np

from .arx_state import ArxState
from .controller_api import Controller


def apply_actuator_limits(u_des: float, u_prev: float) -> float:
    """
    Simple actuator model.

    Limits:
        - max movement per timestep
        - absolute position bounds
    """

    du_max = 0.01   # max position change per step
    u_min = 0    # minimum electrode position
    u_max = 2.0    # maximum electrode position

    # rate limit
    du = u_des - u_prev
    du = np.clip(du, -du_max, du_max)

    u = u_prev + du

    # position saturation
    u = np.clip(u, u_min, u_max)

    return float(u)


def run_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    controller: Controller,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    reference = np.asarray(reference, dtype=float)
    n = len(reference)

    y = np.zeros(n + 1)
    u = np.zeros(n)
    e = np.zeros(n)

    y[0] = state.current_y()
    u_prev = state.current_u()

    controller.reset()

    for k in range(n):

        # plant prediction
        y_pred = state.predict_next_y(model)

        # controller proposes position
        u_des, e_k = controller.step(
            reference=reference[k],
            y_pred=y_pred,
            u_prev=u_prev,
        )

        # actuator limits applied here
        u_k = apply_actuator_limits(u_des, u_prev)

        # advance plant
        state.advance(u_new=u_k, y_new=y_pred)

        y[k + 1] = y_pred
        u[k] = u_k
        e[k] = e_k

        u_prev = u_k

    return y, u, e
