"""
fusion/controllers/relay.py

Step-and-wait relay controller that matches the real SAF electrode logic.

Logic per electrode, called once per timestep:
  - If |error| <= deadband: hold position, reset counters.
  - Else if still waiting after the last move: hold position, count down.
  - Else: move step_size in the direction of the error, start a new wait.
    After escalation_count consecutive moves still outside the deadband,
    switch to the slower wait period until the deadband is re-entered.

Uses the same step(reference, y_pred, u_prev) API as PIDController so it
slots directly into run_closed_loop_from_config with controller_name="relay".

Config CSV columns (1 row broadcast to all electrodes, or 3 rows per electrode):
  deadband, step_size, wait_normal, wait_escalated, escalation_count
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

class RelayController:

    def __init__(
        self,
        deadband:         float,
        step_size:        float = 0.01,
        wait_normal:      int   = 4,
        wait_escalated:   int   = 20,
        escalation_count: int   = 10,
    ) -> None:
        self.deadband         = float(deadband)
        self.step_size        = float(step_size)
        self.wait_normal      = int(wait_normal)
        self.wait_escalated   = int(wait_escalated)
        self.escalation_count = int(escalation_count)

        self._wait_left: int  = 0
        self._consec:    int  = 0
        self._slow:      bool = False

    def reset(self) -> None:
        self._wait_left = 0
        self._consec    = 0
        self._slow      = False

    def step(
        self,
        reference: float,
        y_pred:    float,
        u_prev:    float,
    ) -> tuple[float, float]:
        """
        Returns (u_desired, error).
        Position and rate limits are applied downstream by apply_actuator_limits.
        """
        e = float(reference) - float(y_pred)

        if self._wait_left > 0:
            self._wait_left -= 1
            return float(u_prev), e

        if abs(e) <= self.deadband:
            self._consec = 0
            self._slow   = False
            return float(u_prev), e

        self._consec += 1
        self._slow = self._consec >= self.escalation_count
        wait = self.wait_escalated if self._slow else self.wait_normal
        self._wait_left = wait - 1

        u_des = float(u_prev) + float(np.sign(e)) * self.step_size
        return u_des, e

def load_relay_params(path: str | Path) -> list[dict]:
    """
    Load relay controller parameters from CSV.

    Returns a list of 3 dicts (one per electrode).
    1-row CSV: same parameters broadcast to all three electrodes.
    3-row CSV: one row per electrode (El1, El2, El3 in order).

    Required column: deadband
    Optional columns (all have defaults): step_size, wait_normal,
                                          wait_escalated, escalation_count
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "deadband" not in df.columns:
        raise ValueError("Relay config CSV must contain a 'deadband' column.")

    defaults = {
        "step_size":        0.01,
        "wait_normal":      4,
        "wait_escalated":   20,
        "escalation_count": 10,
    }

    def _row_to_dict(row: pd.Series) -> dict:
        params = {"deadband": float(row["deadband"])}
        for key, default in defaults.items():
            params[key] = float(row[key]) if key in row.index else default
        return params

    if len(df) == 1:
        p = _row_to_dict(df.iloc[0])
        return [p, p, p]

    if len(df) >= 3:
        return [_row_to_dict(df.iloc[i]) for i in range(3)]

    raise ValueError(
        "Relay config CSV must have 1 row (broadcast) or 3 rows (per electrode)."
    )
