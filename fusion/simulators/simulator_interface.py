"""
simulator_interface.py
----------------------
Abstract base class that defines the interface every simulator must implement.

You do not need to change or call this file directly.
SaFSimulator (in saf_simulator.py) already implements all of these methods.

The interface exists so that Plant (in plant.py) can work with any simulator
without knowing the internal details -- it just calls predict_next() and advance().

HOW THE INTERFACE IS USED
--------------------------
Each simulation step works like this:

    y_pred = sim.predict_next()        # ask the simulator what R will be next step
    # ... controller decides new position u_k ...
    sim.advance(u_new=u_k, y_new=y_pred)  # update internal state

Plant wraps this to add GP correction:

    y_fused = plant.predict_next_y()   # predict_next() + GP delta
    plant.advance(u_new=u_k, y_new=y_fused)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class SimulatorInterface(ABC):
    """
    Minimal stateful interface that every simulator must implement.

    SaFSimulator subclasses this. You should not subclass it yourself unless
    you are implementing an entirely different plant model.
    """

    @property
    @abstractmethod
    def output_col(self) -> str:
        """
        Name of the signal this simulator predicts.
        Example: 'El1_Resistance_mOhm_filt' for electrode 1 resistance.
        """

    @property
    @abstractmethod
    def input_col(self) -> str:
        """
        Name of the control input signal.
        Example: 'El1_pos_m' for electrode 1 position.
        """

    @property
    def default_u0(self) -> float:
        """
        Starting electrode position in metres.
        SaFSimulator overrides this with the value read from init_row.
        """
        return 0.0

    @abstractmethod
    def current_y(self) -> float:
        """Return the most recently stored R value for the primary electrode (mOhm)."""

    @abstractmethod
    def current_u(self) -> float:
        """Return the most recent electrode position command applied (metres)."""

    @abstractmethod
    def predict_next(self) -> float:
        """
        Compute the one-step-ahead R prediction without changing internal state.

        Called by Plant.predict_next_y() to get the ARX baseline before
        adding the GP correction. Does NOT advance the lag registers.
        """

    @abstractmethod
    def advance(self, u_new: float, y_new: float) -> None:
        """
        Advance internal state one time step.

        u_new : new electrode position command (metres).
        y_new : the R value to write into the lag registers.
                IMPORTANT: always pass the raw ARX prediction here (not the
                GP-corrected value) so GP errors do not feed back into the ARX.
        """

    @abstractmethod
    def get_gp_features(self) -> dict[str, float]:
        """
        Return the current simulator state as a feature dict for the GP.

        Plant calls this inside predict_next_y() to build the GP input vector.
        Features that Plant manages itself (step_in_window, y_sim, y_real_lag1/2)
        do not need to be included here -- Plant adds them externally.
        """

    def set_feature(self, col: str, val: float) -> None:
        """
        Overwrite a column in the simulator state with a real measurement.

        Use this to inject sensor readings (e.g. from PI) into the state
        instead of relying on ARX predictions for that signal.
        Default is a no-op; SaFSimulator overrides it.
        """

    def reset(self) -> None:
        """Reset the simulator to its initial conditions. Default is a no-op."""
