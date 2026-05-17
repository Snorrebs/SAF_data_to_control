from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy.typing as npt
import numpy as np
@dataclass(frozen=True)
class OpenLoopParams:
    u_constant: npt.array


def load_open_loop_params_csv(path: str | Path) -> OpenLoopParams:
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
        p = float(df["u_constant"].iloc[0])
        return OpenLoopParams(u_constant = np.array([p,p,p]).transpose())

    if len(df) >= 3:
        p = df.to_numpy()
        return OpenLoopParams(u_constant = p)

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

    def step(self, reference: npt.ndarray, y_pred: dict, u_prev: npt.ndarray) -> tuple[list, list]:
        e1 = float(reference[0]) - float(u_prev[0])
        e2 = float(reference[1]) - float(u_prev[1])
        e3 = float(reference[2]) - float(u_prev[2])
        
        e = np.array([e1,e2,e3])

        K = self.params.u_constant
        u_des = np.diag(e) @ K + np.atleast_2d(u_prev).T
        return u_des.tolist(), e.tolist()
