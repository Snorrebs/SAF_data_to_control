from __future__ import annotations

from typing import Protocol


class Controller(Protocol):
    """Single-electrode controller interface.

    Each electrode gets its own Controller instance. The simulation loop calls
    ``step`` once per timestep per electrode.
    """

    def reset(self) -> None:
        ...

    def step(self, reference: float, y_pred: float, dpos_prev: float,) -> tuple[float, float]:
        """Compute the desired holder movement for one timestep.

        Args:
            reference: setpoint for this timestep
            y_pred:    model-predicted output at this timestep
            u_prev:    previous movement input, if used by the controller

        Returns:
            dpos_des: desired holder movement for this timestep [m/step]
            e:        control error (reference - y_pred)
        """
        ...
