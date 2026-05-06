"""
plant.py
--------
Plant wraps a SaFSimulator and adds an optional additive GP correction layer.

The GP predicts the residual between the ARX simulator and the real plant:
  delta = y_real - y_arx

At each step:
  y_fused = y_arx + GP.predict(features)

The simulator's lag registers always receive y_arx (not y_fused) so that
GP errors cannot compound step to step through the ARX inputs.

Example
-----
    from handoff.simulators.plant import Plant
    from handoff.training.gp_loader import load_gp_bundle
    import joblib

    arx   = joblib.load("models/arx_joint.joblib")
    gp    = load_gp_bundle("models/gp_el1_correction.pt")
    row   = build_init_row_from_scalars(pos=1.04, r=1.0, ka=65.0, rx=0.82, v=165.0,
                                        arx_bundle=arx)
    sim   = SaFSimulator(arx, row, electrode=1)
    plant = Plant(sim, gp)

    for step in range(N):
        y_pred = plant.predict_next_y()    # ARX + GP correction
        plant.advance(u_new=u[step], y_new=y_pred)
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .simulator_interface import SimulatorInterface
from ..training.gp_loader import predict_single as _predict_single

# Rolling window length for variability features.
# Must match the value used when training the GP (train_gp.py, default is 30).
_ROLLING_WINDOW = 30


class Plant:
    """
    Wraps a SaFSimulator and adds an optional additive GP correction.

    Parameters
    ----------
    sim        : SaFSimulator instance (already initialised)
    gp_bundle  : loaded GP bundle dict, or None to run without GP correction
    H          : training window length; step_in_window is clamped to [0, H-1].
                 Should match the value used during GP training (default 20).
    clip_delta : if set, Sets a maximum GP corrections, Corrections
                 larger than this (mOhm) are clipped.
                 This limits extrapolation when the GP is far from its
                 training distribution. Recommended: 0.10-0.15 mOhm.
    """

    def __init__(
        self,
        sim:        SimulatorInterface,
        gp_bundle:  dict[str, Any] | None,
        H:          int   = 20,
        clip_delta: float | None = 0.15,
    ) -> None:
        self._sim        = sim
        self._bundle     = gp_bundle
        self._H          = H
        self._clip_delta = clip_delta
        self._step       = 0
        self._y_sim_last: float = sim.current_y()

        # Most recent GP correction stats. Read these after predict_next_y().
        # gp_mean is the additive correction applied (mOhm). 0.0 when no GP.
        # gp_std is one-std predictive uncertainty (mOhm). 0.0 when no GP.
        self.gp_mean: float = 0.0
        self.gp_std:  float = 0.0

        # Rolling output history for GP features y_real_lag1 and y_real_lag2.
        initial_y = sim.current_y()
        self._y_real_hist: deque[float] = deque([initial_y, initial_y], maxlen=3)

        if gp_bundle is not None:
            self._feature_names: list[str] = gp_bundle["feature_names"]
            sigma = gp_bundle.get("sigma_ref")
            self._sigma_ref: float = float(sigma) if sigma is not None else 0.01
            self._feed_means: dict[str, float] = gp_bundle.get("feed_feature_means", {})
        else:
            self._feature_names = []
            self._sigma_ref     = 0.01
            self._feed_means    = {}

        # Rolling variation buffers used by the GP feature set.
        # Only included if the GP bundle actually uses these features.
        self._has_rolling: bool = any("_rolling_std_" in f for f in self._feature_names)
        if self._has_rolling:
            self._rolling_reac: dict[int, deque] = {
                j: deque([0.0] * _ROLLING_WINDOW, maxlen=_ROLLING_WINDOW)
                for j in (1, 2, 3)
            }
            self._rolling_r: dict[int, deque] = {
                j: deque([0.0] * _ROLLING_WINDOW, maxlen=_ROLLING_WINDOW)
                for j in (1, 2, 3)
            }

   
    #                         Public interface 
    # (matches FusedPlant from the original codebase) 
    # ------------------------------------------------------------------ #

    def current_y(self) -> float:
        """Return the current stored R value for the El1 electrode (mOhm)."""
        return self._sim.current_y()

    def current_u(self) -> float:
        """Return the current position of the El1 electrode (m)."""
        return self._sim.current_u()

    def predict_next_y(self) -> float:
        """
        Find next R prediction with GP correction.

        Calls the ARX simulator first, then applies the additive GP correction.
        The raw ARX value is saved internally so advance() can pass it to the
        simulator's lag registers (not the GP-corrected value).

        Returns the GP-corrected R prediction (mOhm).
        """
        # Step 1: Get the ARX simulator's R prediction
        y_arx = self._sim.predict_next()
        self._y_sim_last = y_arx   # save ARX value. used in advance()

        # If no GP bundle is loaded, return the plain ARX prediction
        if self._bundle is None:
            self.gp_mean = 0.0
            self.gp_std  = 0.0
            return y_arx

        # Step 2: build the GP input feature vector
        x        = self._build_feature_vector(y_sim=y_arx)
        # Step 3: run the GP to get (mean correction, variance)
        mu, var  = _predict_single(self._bundle, x)

        # Clip large corrections to prevent the GP from exploding the prediciton
        if self._clip_delta is not None:
            mu = float(np.clip(mu, -self._clip_delta, self._clip_delta))

        # Expose correction stats so the controller can read them
        self.gp_mean = float(mu)
        self.gp_std  = float(max(var, 0.0) ** 0.5)
        # Step 4: return ARX + GP correction
        return float(y_arx + mu)

    def advance(self, u_new: float, y_new: float) -> None:
        """
        Advance the simulation one step.

        u_new : new electrode position command (m)
        y_new : the fused output from predict_next_y()
        """
        # Give the simulator the raw ARX prediction (NOT the GP corrected value).
        # step by step through the lag registers.
        self._sim.advance(u_new=u_new, y_new=self._y_sim_last)
        # Store the GP corrected output so the GP can use it as a lagged feature
        self._y_real_hist.append(y_new)
        self._step += 1

        if self._has_rolling:
            feats = self._sim.get_gp_features()
            self._update_rolling(
                new_reac={j: feats.get(f"El{j}_CalcReac_filt_lag1", 0.0) for j in (1, 2, 3)},
                new_r   ={j: feats.get(f"El{j}_y_filt_lag1",        0.0) for j in (1, 2, 3)},
            )

    def advance_teacher_forced(self, u_new: float, y_real: float) -> None:
        """
        Advance with a real measured y.

        Use this when replaying recorded data to prevent the
        simulator from drifting away from the real plant trajectory.
        """
        self._sim.advance(u_new=u_new, y_new=y_real)
        self._y_real_hist.append(y_real)
        self._step += 1

    def get_full_state(self) -> dict[str, float]:
        """
        Return all current signal values plus one-step-ahead ARX predictions.

        Call this after advance() to get the state the controller should act on at the next step.

        Keys: R{i}, kA{i}, X{i}, pos{i}, V (current lag values)
              R_next{i}, kA_next{i}, X_next{i}, V_next (ARX predictions)
        where i is 1, 2, or 3 for each electrode.
        """
        return self._sim.get_full_state()

    def set_feature(self, col: str, val: float) -> None:
        """Inject a measured signal into the simulator state.
        
           if you want to keep a feature constant or controll it externally
        """
        self._sim.set_feature(col, val)

    def seed_rolling(self, seg: Any, t0: int) -> None:
        """
        Seed rolling variability buffers from real data at a reset point.

        When replaying historical data
        call this at the start of each replay window so the rolling_std features
        match the state the GP saw during training.

        seg : pd.DataFrame with El{j}_y_filt columns
        t0  : reset index within seg
        """
        if not self._has_rolling:
            return
        r_start = max(0, t0 - _ROLLING_WINDOW + 1)
        for j in (1, 2, 3):
            r_col = f"El{j}_y_filt"
            if hasattr(seg, "columns") and r_col in seg.columns:
                vals = list(seg[r_col].values[r_start:t0 + 1])
                self._rolling_r[j] = deque(vals, maxlen=_ROLLING_WINDOW)
            else:
                self._rolling_r[j] = deque([0.0] * _ROLLING_WINDOW, maxlen=_ROLLING_WINDOW)
            reac_val = 0.0
            if hasattr(self._sim, "get_gp_features"):
                reac_val = float(self._sim.get_gp_features().get(
                    f"El{j}_CalcReac_filt_lag1", 0.0))
            n_hist = t0 - r_start + 1
            self._rolling_reac[j] = deque([reac_val] * n_hist, maxlen=_ROLLING_WINDOW)

    
    #                            Private 
    # ------------------------------------------------------------------ #

    def _update_rolling(
        self,
        new_reac: dict[int, float],
        new_r:    dict[int, float],
    ) -> None:
        """Append latest reactance and R values to the rolling buffers."""
        if not self._has_rolling:
            return
        for j in (1, 2, 3):
            if j in new_reac:
                self._rolling_reac[j].append(float(new_reac[j]))
            if j in new_r:
                self._rolling_r[j].append(float(new_r[j]))

    def _build_feature_vector(self, y_sim: float | None = None) -> np.ndarray:
        """
        Assemble the feature array that the GP expects as input.

        Reads base features from the simulator, then adds:
        step_in_window is the clamped step counter (stays at H-1 beyond training window)
        y_real_lag1/2 are the previous GP-corrected R values
        y_sim, y_sim_sq is the ARX prediction and its square
        rolling_std features is a 30-step rolling mean of R and reactance per electrode
        """
        features = self._sim.get_gp_features()

        # Clamp to [0, H-1] rather than cycling so the GP stays in
        # "end-of-window" mode for steps beyond the training window.
        features["step_in_window"] = float(min(self._step, self._H - 1))

        hist = self._y_real_hist
        features["y_real_lag1"] = hist[-1]
        features["y_real_lag2"] = hist[-2] if len(hist) >= 2 else hist[-1]

        if y_sim is not None:
            features["y_sim"]    = y_sim
            features["y_sim_sq"] = y_sim * y_sim

        if self._has_rolling:
            for j in (1, 2, 3):
                buf_reac = self._rolling_reac[j]
                buf_r    = self._rolling_r[j]
                features[f"El{j}_rolling_std_CalcReac_30s"] = (
                    float(np.std(buf_reac)) if len(buf_reac) > 1 else 0.0
                )
                features[f"El{j}_rolling_std_R_30s"] = (
                    float(np.std(buf_r)) if len(buf_r) > 1 else 0.0
                )

        # Use training-set feature means for any features absent during simulation
        # (typically measurement-only features unavailable in open-loop). This keeps
        # the GP inside its training distribution instead of seeing out-of-range zeros.
        return np.array(
            [features.get(f, self._feed_means.get(f, 0.0)) for f in self._feature_names],
            dtype=np.float32,
        )
