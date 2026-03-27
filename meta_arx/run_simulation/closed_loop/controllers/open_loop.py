from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class OpenLoopParams:
    u_constant: float


def load_open_loop_params_csv(path: str | Path) -> list[OpenLoopParams]:
    """Load open-loop parameters from CSV.

    The CSV must contain column ``u_constant`` and either:

    * **1 row** – the same value is broadcast to all 3 electrodes, or
    * **3 rows** – one row per electrode (El1, El2, El3 in order).

    Returns:
        A list of 3 ``OpenLoopParams`` instances (one per electrode).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "u_constant" not in df.columns:
        raise ValueError("Open-loop params CSV is missing required column 'u_constant'")

    if len(df) == 1:
        p = OpenLoopParams(u_constant=float(df["u_constant"].iloc[0]))
        return [p, p, p]

    if len(df) >= 3:
        return [OpenLoopParams(u_constant=float(df["u_constant"].iloc[i])) for i in range(3)]

    raise ValueError("Open-loop params CSV must have either 1 row (broadcast) or at least 3 rows")


class OpenLoopController:
    """Open-loop controller: outputs ``u_constant * reference``.

    Useful as a feedforward baseline or for open-loop step tests.
    """

    def __init__(self, params: OpenLoopParams, dt: float) -> None:
        self.params = params
        self.dt = float(dt)

    def reset(self) -> None:
        pass

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        e = float(reference) - float(y_pred)
        u_des = self.params.u_constant * float(reference)
        return u_des, e