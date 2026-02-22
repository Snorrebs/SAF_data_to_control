from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .arx_state import ArxState


@dataclass
class PIDParams:
    Kp: float
    Ki: float = 0.0
    Kd: float = 0.0


def run_closed_loop(
    model,  # this is the ARX bundle dict from joblib
    state: ArxState,
    r: np.ndarray,
    pid: PIDParams,
    Ts: float = 1.0,
    u_min: Optional[float] = None,
    u_max: Optional[float] = None,
    du_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    r = np.asarray(r, dtype=float)
    N = len(r)

    y = np.zeros(N + 1, dtype=float)
    u = np.zeros(N, dtype=float)
    e = np.zeros(N, dtype=float)

    # initial conditions from state
    y[0] = state.current_y()
    u_prev = state.current_u_el1()

    integral = 0.0
    prev_pred = y[0]

    for k in range(N):
        # 1) predict next output with current state + model bundle
        y_pred = state.predict_next_y(model)
        e[k] = r[k] - y_pred

        # 2) PID on predicted output
        integral += e[k] * Ts
        derivative = (y_pred - prev_pred) / Ts if k > 0 else 0.0

        # incremental PID around last position
        u_cmd = (
            pid.Kp * e[k]
            + pid.Ki * integral
            + pid.Kd * derivative
            + u_prev
        )

        # rate limit
        if du_max is not None:
            u_cmd = np.clip(u_cmd, u_prev - du_max, u_prev + du_max)

        # absolute saturation
        if u_min is not None or u_max is not None:
            u_cmd = np.clip(
                u_cmd,
                u_min if u_min is not None else -np.inf,
                u_max if u_max is not None else np.inf,
            )

        u[k] = u_cmd
        prev_pred = y_pred
        u_prev = u_cmd

        # 3) advance ARX state using new control + predicted y
        #    NOTE: new ArxState.advance no longer takes the bundle
        state.advance(u_El2_new=u_cmd, y_new=y_pred)
        #state.advance(u_El2_new=u_cmd, y_new=y_pred)

        # 4) log "next" output
        y[k + 1] = y_pred

    return y, u, e
