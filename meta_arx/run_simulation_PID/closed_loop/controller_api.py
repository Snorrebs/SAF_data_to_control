# run_simulation_PID/closed_loop/controller_api.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Controller(Protocol):
    """
    Minimal controller interface for the simulator.

    Contract:
      - reset(): clears internal state (integrator, filters, etc.)
      - step(reference, y_pred, u_prev): returns (u_cmd, error)

    Notes:
      - y_pred is the current simulated/measured output (e.g., current or resistance)
      - reference is the desired output at this timestep
      - u_prev is the previous applied control command (e.g., holder position command)
    """

    def reset(self) -> None:
        ...

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        ...