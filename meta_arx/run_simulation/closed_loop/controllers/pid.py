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
    du_min: float = -0.01   # rate limit [m/step] — used for anti-windup clamp
    du_max: float =  0.01


def load_pid_params_csv(path: str | Path) -> list[PIDParams]:
    """Load PID parameters from CSV.

    The CSV must contain columns ``kp``, ``ki``, ``kd`` and either:

    * **1 row** – the same parameters are broadcast to all 3 electrodes, or
    * **3 rows** – one row per electrode (El1, El2, El3 in order).

    Optional columns ``du_min`` / ``du_max`` set the anti-windup clamp
    (defaults: -0.01 / +0.01 m/step, matching the physical rate limit).

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
        row = df.iloc[0]
        p = PIDParams(
            kp=float(row["kp"]),
            ki=float(row["ki"]),
            kd=float(row["kd"]),
            du_min=float(row["du_min"]) if "du_min" in row.index else -0.01,
            du_max=float(row["du_max"]) if "du_max" in row.index else  0.01,
        )
        return [p, p, p]

    if len(df) >= 3:
        return [
            PIDParams(
                kp=float(df["kp"].iloc[i]),
                ki=float(df["ki"].iloc[i]),
                kd=float(df["kd"].iloc[i]),
                du_min=float(df["du_min"].iloc[i]) if "du_min" in df.columns else -0.01,
                du_max=float(df["du_max"].iloc[i]) if "du_max" in df.columns else  0.01,
            )
            for i in range(3)
        ]

    raise ValueError(
        "PID params CSV must have either 1 row (broadcast) or at least 3 rows (one per electrode)"
    )


class PIDController:
    """Direct-form PID controller producing Delta-u (electrode movement per step).

    The output u_des is the desired movement [m/step] for this step,
    consistent with the VARX model's delta-pos control input. u_prev is
    accepted for interface compatibility but is not used — the PID output
    is not an increment over u_prev; it IS the delta-u directly:

        delta-u = Kp * e + Ki * integral(e) - Kd * (dy/dt)

    Design choices:
    - Derivative acts on the measured output y (not the error) to avoid
      derivative kick on reference steps.
    - Clamping anti-windup: when the output saturates at du_min/du_max,
      the last integral increment is undone so the integrator stops
      accumulating. This prevents large overshoots on the near-integrator
      furnace AR dynamics.

    The physical rate limit is also enforced externally by
    closed_loop_sim.apply_rate_limit.
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

    def step(self, reference: float, y_pred: float, dpos_prev: float) -> tuple[float, float]:
        """Compute desired holder movement for this step.

        Args:
            reference: setpoint in R-tilde space [mOhm]
            y_pred:    current predicted output in R-tilde space [mOhm]
            dpos_prev: previous holder movement [m/step], unused here

        Returns:
            dpos_des: desired holder movement [m/step]
            e:        tracking error (reference - y_pred) [mOhm]
        """
        e = float(reference) - float(y_pred)

        d_term = (
            0.0
            if self._prev_y_pred is None
            else (float(y_pred) - self._prev_y_pred) / self.dt
        )

        self._i_term += e * self.dt

        p = self.params
        dpos_des = p.kp * e + p.ki * self._i_term - p.kd * d_term

        dpos_clipped = float(np.clip(dpos_des, p.du_min, p.du_max))
        if dpos_des != dpos_clipped and p.ki != 0.0:
            self._i_term -= e * self.dt

        self._prev_y_pred = float(y_pred)

        return dpos_des, e