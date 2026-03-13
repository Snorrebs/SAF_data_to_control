from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class OpenLoopParams:
    u_constant: float


def load_open_loop_params_csv(path: str | Path) -> OpenLoopParams:
    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError("Open-loop params CSV must contain exactly one row")

    row = df.iloc[0]

    if "u_constant" not in row or pd.isna(row["u_constant"]):
        raise ValueError("Missing required open-loop parameter 'u_constant'")

    return OpenLoopParams(
        u_constant=float(row["u_constant"]),
    )


class OpenLoopController:
    def __init__(self, params: OpenLoopParams, dt: float) -> None:
        self.params = params
        self.dt = float(dt)

    def reset(self) -> None:
        pass

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        e = float(reference) - float(y_pred)
        u_des = float(self.params.u_constant)
        return u_des, e

    def update_integral(self, accept: bool) -> None:
        pass
