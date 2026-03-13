from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PIDParams:
    kp: float
    ki: float
    kd: float


def load_pid_params_csv(path: str | Path) -> PIDParams:
    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError("PID params CSV must contain exactly one row")

    row = df.iloc[0]

    for name in ["kp", "ki", "kd"]:
        if name not in row or pd.isna(row[name]):
            raise ValueError(f"Missing required PID parameter '{name}'")

    return PIDParams(
        kp=float(row["kp"]),
        ki=float(row["ki"]),
        kd=float(row["kd"]),
    )


class PIDController:
    def __init__(self, params: PIDParams, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be > 0")

        self.params = params
        self.dt = float(dt)
        self._i_term = 0.0
        self._prev_e = None

    def reset(self) -> None:
        self._i_term = 0.0
        self._prev_e = None

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        e = float(reference) - float(y_pred)

        d_term = 0.0 if self._prev_e is None else (e - self._prev_e) / self.dt
        self._prev_e = e

        self._i_term += e * self.dt

        p = self.params
        u_des = p.kp * e + p.ki * self._i_term + p.kd * d_term

        return float(u_des), float(e)

    def update_integral(self, accept: bool) -> None:
        # no anti-windup handling for now
        pass
