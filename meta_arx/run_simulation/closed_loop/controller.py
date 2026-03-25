from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDParams:
    Kp: float
    Ki: float = 0.0
    Kd: float = 0.0


@dataclass
class PIDController:
    params: PIDParams
    Ts: float = 1.0
    u_min: float = 10
    u_max: float = 200
    du_max: float = 1

    def __post_init__(self) -> None:
        self._integral = 0.0
        self._prev_y: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_y = None

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        error = reference - y_pred
        self._integral += error * self.Ts
        derivative = 0.0 if self._prev_y is None else (y_pred - self._prev_y) / self.Ts

        u_cmd = (
            u_prev
            + self.params.Kp * error
            + self.params.Ki * self._integral
            + self.params.Kd * derivative
        )

        if self.du_max is not None:
            u_cmd = np.clip(u_cmd, u_prev - self.du_max, u_prev + self.du_max)
        if self.u_min is not None or self.u_max is not None:
            u_cmd = np.clip(
                u_cmd,
                -np.inf if self.u_min is None else self.u_min,
                np.inf if self.u_max is None else self.u_max,
            )

        self._prev_y = y_pred
        return float(u_cmd), float(error)
