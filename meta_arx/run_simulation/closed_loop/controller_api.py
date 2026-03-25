from __future__ import annotations

from typing import Protocol


class Controller(Protocol):
    """Single-electrode controller interface.

    Each electrode gets its own Controller instance.  The simulation loop
    calls ``step`` once per timestep per electrode.
    """

    def reset(self) -> None:
        ...

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        """Compute the desired actuator position for one timestep.

        Args:
            reference: setpoint for this timestep
            y_pred:    model-predicted output at this timestep
            u_prev:    actuator position at the previous timestep

        Returns:
            u_des : desired actuator position (before actuator limits)
            e     : control error  (reference - y_pred)
        """
        ...