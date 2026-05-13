"""
reference_converter.py
======================
Converts a resistance reference signal from absolute mΩ into detrended
R̃ space (the space the VARX model operates in).

Background
----------
The VARX model was trained on R̃(t) = R(t) - median(R, 30-min window).
A controller reference given as an absolute resistance value (e.g. 1.0 mΩ)
must be converted to R̃ before being passed to the simulator, otherwise the
controller will regulate toward the wrong target.

The conversion is simply:
    R̃_ref(t) = R_ref(t) - trend(t)

where trend(t) is the current rolling median estimate of the slow baseline.

Trend estimation strategy
-------------------------
1. INITIALISATION — seed the rolling window from the ArxState init row.
   The lag columns El{i}_Resistance_mOhm_filt_lag1 ... lag_N in the state
   row contain the most recent N resistance values (newest at lag1, oldest
   at lag_N). These are used to populate the initial window without any
   additional data source.

2. ONLINE UPDATE — at each simulation step, append the latest predicted
   (or measured) resistance to the window and drop the oldest sample.
   The median is recomputed from the current window contents.

This mirrors the training preprocessing exactly (rolling median, centred,
same window length) while being causally implementable in real time.

Usage
-----
    from reference_converter import ReferenceConverter

    # Initialise once from the ArxState row
    converter = ReferenceConverter.from_state(state, window_s=1800)

    # At each simulation step k:
    r_tilde = converter.convert(r_abs)          # shape (3,) → (3,)
    ...
    converter.update(y_pred)                    # advance window with new R̃

    # Or convert an entire pre-planned trajectory up front (open-loop trend):
    r_tilde_traj = converter.convert_trajectory(r_abs_traj)  # (n, 3) → (n, 3)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Sequence

# Column base names used in the init CSV / ArxState row
_RES_LAG_BASES = [
    "El1_Resistance_mOhm_filt",
    "El2_Resistance_mOhm_filt",
    "El3_Resistance_mOhm_filt",
]


class ReferenceConverter:
    """
    Converts absolute resistance references to R̃ (detrended) space.

    Parameters
    ----------
    initial_windows : list of array-like, length 3
        Seed values for each electrode's rolling window.
        Newest sample first (index 0 = most recent).
        Typically extracted from ArxState lag columns.
    window_s : int
        Rolling median window length in seconds (= samples at 1 Hz).
        Must match the detrend_window used during VARX training (default 1800).
    """

    def __init__(
        self,
        initial_windows: list[Sequence[float]],
        window_s: int = 1800,
    ) -> None:
        if len(initial_windows) != 3:
            raise ValueError(
                f"Expected 3 electrode windows, got {len(initial_windows)}"
            )

        self._window_s = window_s
        # deque with maxlen automatically drops oldest when full
        self._windows: list[deque] = []

        for ei, win in enumerate(initial_windows):
            win_arr = np.asarray(win, dtype=float)
            # Remove NaNs — can occur if lag depth < window_s
            win_arr = win_arr[~np.isnan(win_arr)]
            if len(win_arr) == 0:
                raise ValueError(
                    f"El{ei+1}: no valid seed values after removing NaNs. "
                    f"Check that the init CSV contains lag columns."
                )
            d = deque(win_arr, maxlen=window_s)
            self._windows.append(d)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_state(
        cls,
        state,                  # ArxState instance
        window_s: int = 1800,
    ) -> "ReferenceConverter":
        """
        Initialise from an ArxState row.

        Reads El{i}_Resistance_mOhm_filt_lag1 ... lag_N to seed the
        rolling window. Lag1 is the most recent value, lag_N is oldest —
        so we reverse to get chronological order (oldest first) for the
        deque, which is the natural append direction.

        Parameters
        ----------
        state : ArxState
            Initialised state object (from load_initial_state).
        window_s : int
            Rolling median window length in seconds. Should match the
            value used during VARX training (default 1800).
        """
        row = state.row
        initial_windows = []

        for base in _RES_LAG_BASES:
            # Collect all available lag depths for this electrode
            lag_values = []
            k = 1
            while True:
                col = f"{base}_lag{k}"
                if col not in row.index:
                    break
                lag_values.append(float(row[col]))
                k += 1

            if len(lag_values) == 0:
                raise KeyError(
                    f"No lag columns found for '{base}' in ArxState row. "
                    f"Check that the init CSV was generated with generate_init_csv()."
                )

            # lag_values[0] = lag1 = most recent
            # lag_values[-1] = lag_N = oldest
            # Reverse to chronological order: oldest → newest
            chronological = list(reversed(lag_values))
            initial_windows.append(chronological)

        n_lags = len(initial_windows[0])
        print(
            f"  ReferenceConverter: seeded from {n_lags} lag samples per electrode "
            f"(window_s={window_s}). "
            f"Window fill = {100*n_lags/window_s:.1f}%."
        )
        if n_lags < window_s // 2:
            print(
                f"  WARNING: Window is less than 50% full ({n_lags}/{window_s} samples). "
                f"Trend estimate may be unreliable at simulation start. "
                f"Consider using a longer init history or a warm-up period."
            )

        return cls(initial_windows, window_s=window_s)

    @classmethod
    def from_operating_point(
        cls,
        op_point: Sequence[float],
        window_s: int = 1800,
    ) -> "ReferenceConverter":
        """
        Initialise with a fixed operating point per electrode.

        Fills each window entirely with the given value, which gives a
        median equal to that value. Useful for quick testing or when no
        history is available.

        Parameters
        ----------
        op_point : array-like, shape (3,)
            Absolute resistance operating point [mΩ] per electrode.
        window_s : int
            Window length in seconds.
        """
        op = np.asarray(op_point, dtype=float)
        if op.shape != (3,):
            raise ValueError(f"op_point must have shape (3,), got {op.shape}")
        initial_windows = [
            np.full(window_s, op[i]).tolist() for i in range(3)
        ]
        return cls(initial_windows, window_s=window_s)

    # ── Core conversion ───────────────────────────────────────────────────────

    def current_trend(self) -> np.ndarray:
        """
        Current rolling median estimate of the slow resistance baseline.

        Returns
        -------
        trend : np.ndarray, shape (3,)
            Median of the current window per electrode [mΩ].
        """
        return np.array([np.median(w) for w in self._windows], dtype=float)

    def convert(self, r_abs: Sequence[float]) -> np.ndarray:
        """
        Convert absolute resistance reference to R̃ space.

            R̃_ref = R_ref - trend

        Parameters
        ----------
        r_abs : array-like, shape (3,)
            Absolute resistance reference per electrode [mΩ].

        Returns
        -------
        r_tilde : np.ndarray, shape (3,)
            Detrended reference in R̃ space [mΩ deviation].
        """
        r_abs = np.asarray(r_abs, dtype=float)
        if r_abs.shape != (3,):
            raise ValueError(f"r_abs must have shape (3,), got {r_abs.shape}")
        return r_abs - self.current_trend()

    def update(self, r_new: Sequence[float]) -> None:
        """
        Advance the rolling window with the latest resistance values.

        Call this once per simulation step, after predict_next_y(), with
        the new predicted (or measured) resistance values. This keeps the
        trend estimate in sync with the evolving simulation state.

        Parameters
        ----------
        r_new : array-like, shape (3,)
            Latest absolute resistance values [mΩ] — use the raw R values,
            not R̃. If your simulator only tracks R̃, add back the current
            trend first: r_abs = r_tilde + self.current_trend()
        """
        r_new = np.asarray(r_new, dtype=float)
        if r_new.shape != (3,):
            raise ValueError(f"r_new must have shape (3,), got {r_new.shape}")
        for i, w in enumerate(self._windows):
            w.append(float(r_new[i]))

    def convert_trajectory(
        self,
        r_abs_traj: np.ndarray,
        freeze_trend: bool = False,
    ) -> np.ndarray:
        """
        Convert an entire reference trajectory to R̃ space.

        Two modes:
        - freeze_trend=False (default): evolves the trend step by step using
          the reference itself as a proxy for the resistance. Useful when you
          want to account for setpoint changes shifting the trend over time.
          NOTE: this modifies the internal window state — call only once, or
          reinitialise the converter afterwards.
        - freeze_trend=True: uses the current trend for all timesteps.
          Simpler, appropriate when the reference is nearly constant or the
          simulation is short relative to the window length.

        Parameters
        ----------
        r_abs_traj : np.ndarray, shape (n,) or (n, 3)
            Absolute resistance reference trajectory [mΩ].
        freeze_trend : bool
            If True, hold the trend constant at its current value.

        Returns
        -------
        r_tilde_traj : np.ndarray, shape (n, 3)
            Detrended reference trajectory.
        """
        r_abs_traj = np.asarray(r_abs_traj, dtype=float)
        if r_abs_traj.ndim == 1:
            r_abs_traj = np.repeat(r_abs_traj.reshape(-1, 1), 3, axis=1)
        if r_abs_traj.shape[1] != 3:
            raise ValueError(
                f"r_abs_traj must have shape (n, 3), got {r_abs_traj.shape}"
            )

        n = r_abs_traj.shape[0]
        r_tilde_traj = np.zeros_like(r_abs_traj)

        if freeze_trend:
            trend = self.current_trend()
            r_tilde_traj = r_abs_traj - trend[np.newaxis, :]
        else:
            for k in range(n):
                r_tilde_traj[k] = self.convert(r_abs_traj[k])
                self.update(r_abs_traj[k])

        return r_tilde_traj

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def window_fill_pct(self) -> np.ndarray:
        """Fraction of the window that is filled (0–1) per electrode."""
        return np.array(
            [len(w) / self._window_s for w in self._windows], dtype=float
        )

    def __repr__(self) -> str:
        trend = self.current_trend()
        fill  = self.window_fill_pct()
        return (
            f"ReferenceConverter("
            f"window_s={self._window_s}, "
            f"trend=[{trend[0]:.4f}, {trend[1]:.4f}, {trend[2]:.4f}] mΩ, "
            f"fill=[{fill[0]:.0%}, {fill[1]:.0%}, {fill[2]:.0%}])"
        )