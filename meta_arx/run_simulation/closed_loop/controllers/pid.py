from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
@dataclass(frozen=True)
class PIDParams:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
 
 
def load_pid_params_csv(path: str | Path) -> list[PIDParams]:
    """Load PID parameters from CSV.
 
    The CSV must contain columns ``kp``, ``ki``, ``kd`` and either:
 
    * **1 row** – the same parameters are broadcast to all 3 electrodes, or
    * **3 rows** – one row per electrode (El1, El2, El3 in order).
 
    Returns:
        A list of 3 ``PIDParams`` instances (one per electrode).
    """
    df = pd.read_csv(path)
 
    # Accept both capitalised and lower-case column names
    df.columns = [c.strip().lower() for c in df.columns]
 
    for name in ("kp", "ki", "kd"):
        if name not in df.columns:
            raise ValueError(f"PID params CSV is missing required column '{name}'")
 
    if len(df) == 1:
        p = PIDParams(
            kp=float(df["kp"].iloc[0]),
            ki=float(df["ki"].iloc[0]),
            kd=float(df["kd"].iloc[0]),
        )
        return [p, p, p]
 
    if len(df) >= 3:
        return [
            PIDParams(
                kp=float(df["kp"].iloc[i]),
                ki=float(df["ki"].iloc[i]),
                kd=float(df["kd"].iloc[i]),
            )
            for i in range(3)
        ]
 
    raise ValueError("PID params CSV must have either 1 row (broadcast) or at least 3 rows (one per electrode)")
 
 
class PIDController:
    """Velocity-form PID controller.
 
    - Derivative acts on the measured output (not the error) to avoid
      derivative kick on reference steps.
    - Velocity form: ``u = u_prev + Kp*e + Ki*integral - Kd*d_output``
 
    Actuator rate and position limits are applied externally by
    ``closed_loop_sim.apply_actuator_limits``, not here.
    """
 
    def __init__(self, params: PIDParams, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be > 0")
 
        self.params = params
        self.dt = float(dt)
 
        self._i_term: float = 0.0
        self._prev_y_pred: float | None = None
 
    def reset(self) -> None:
        self._i_term = 0.0
        self._prev_y_pred = None
 
    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        e = float(reference) - float(y_pred)
 
        # Derivative on output (not error) to avoid derivative kick
        d_term = (
            0.0
            if self._prev_y_pred is None
            else (float(y_pred) - self._prev_y_pred) / self.dt
        )
 
        self._i_term += e * self.dt
        self._prev_y_pred = float(y_pred)
 
        p = self.params
        u_des = float(u_prev) + p.kp * e + p.ki * self._i_term - p.kd * d_term
 
        return u_des, e