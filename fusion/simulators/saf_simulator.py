"""
saf_simulator.py
Three-electrode ARX simulator for SAF furnace.

The simulator propagates 10 coupled signals at each step
  R  x 3  arc resistance, mOhm, one per electrode
  kA x 3  arc current, kA
  X  x 3  arc reactance, mOhm
  V  x 1  shared transformer RMS voltage

All 10 signals are predicted by a single joint ARX model in one matrix
multiply per step. Cross-electrode coupling (e.g. El2 current affecting
El1 resistance) is captured because all
electrode states are included as inputs to the joint model.

The simulator supports both single-electrode and full three-electrode control:
  - single-electrode: call advance(u_new, y_new). This will keep non-primary electrode
    positions constant.
  - three-electrode:  call advance_multi(u_new_vec, y_new_vec).

GP correction is applied externally by the "Plant" wrapper (plant.py).
The simulator's lag registers store ARX-only predictions so that GP correction
does not affect step-to-step state through the ARX.

Public API
  SaFSimulator              : the simulator class
  build_init_row            : build starting state from a data segment (from historic data)
  build_init_row_from_scalars : build starting state from scalar values (from custom values)

Example (single-electrode)
    import joblib
    from handoff.simulators.saf_simulator import SaFSimulator, build_init_row_from_scalars
    from handoff.simulators.plant import Plant
    from handoff.training.gp_loader import load_gp_bundle

    arx   = joblib.load("models/arx_joint.joblib")
    gp    = load_gp_bundle("models/gp_el1_correction.pt")
    row   = build_init_row_from_scalars(pos=1.04, r=1.006, ka=65.0, rx=0.82, v=165.0,
                                        arx_bundle=arx)
    sim   = SaFSimulator(arx, row, electrode=1)
    plant = Plant(sim, gp)

    for step in range(N):
        y_pred = plant.predict_next_y()
        plant.advance(u_new=u[step], y_new=y_pred)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator_interface import SimulatorInterface

# Default electrode position when no real measurement is available
_DEFAULT_POS = 1.04   # (typical SAF operating position)

# Maximum lag depth tracked in sim._row for GP features.
# The ARX model only uses lags 1-3, but GP models (V12+) use up to this depth.
# Increasing this value adds columns to sim._row automatically. No other changes needed.
_MAX_LAG = 10

def _shift_lags(row: "pd.Series", prefix: str, new_val: float) -> None:
    """Shift lag registers for a given signal prefix in-place.

    Finds all columns of the form ``prefix{k}`` present in row.index,
    shifts them old-to-new (lagN <- lagN-1 ... lag2 <- lag1 <- new_val),
    then writes new_val into lag1.  Works for any lag depth. The simulator
    never needs to know in advance how many lags a GP variant will use.
    """
    k = 1
    while f"{prefix}{k}" in row.index:
        k += 1
    # k is now one past the highest lag present; shift from top down
    for j in range(k - 1, 1, -1):
        row[f"{prefix}{j}"] = row[f"{prefix}{j-1}"]
    if f"{prefix}1" in row.index:
        row[f"{prefix}1"] = new_val

# Maps electrode index to the tap-changer column name used as a GP feature
_TC_COL = {1: "TCA", 2: "TCB", 3: "TCC"}

# Hard upper limit on predicted reactance (mOhm). 
_REAC_CLIP_MAX = 3.0

#                            Internal helper: fake scaler for clip-bound computation
class _FakeScaler:
    """
    Minimal scaler stub with mean_ and scale_ arrays.

    The base simulator's _bounds() method reads bundle["y_scaler"].mean_[0]
    and .scale_[0] to estimate clipping thresholds. This lets us extract
    per-signal statistics from the joint Y_scaler without needing 10 separate
    single-output scalers.
    """
    def __init__(self, mean: float, scale: float) -> None:
        self.mean_  = np.array([mean],  dtype=np.float64)
        self.scale_ = np.array([scale], dtype=np.float64)

def _extract_clip_bounds(y_sc, y_index: dict) -> tuple[dict, dict]:
    """
    Build fake per-bundle dicts that carry clip bounds for each signal.

    The joint Y_scaler stores mean and scale for all 10 outputs as a array.
    This function extracts the per-signal stats and wraps them in _FakeScaler
    objects so the parent class can compute clipping bounds from them.

    Returns (fake_bundles, fake_reac) where:
      fake_bundles: {1: {"r": ..., "ka": ...}, 2: ..., 3: ..., "v": ...}
      fake_reac:    {1: {"y_scaler": ...}, 2: ..., 3: ...}
    """
    fake_bundles: dict = {}
    for i in (1, 2, 3):
        r_idx  = y_index["R"][i]
        ka_idx = y_index["kA"][i]
        fake_bundles[i] = {
            "r":  {"y_scaler": _FakeScaler(float(y_sc.mean_[r_idx]),  float(y_sc.scale_[r_idx]))},
            "ka": {"y_scaler": _FakeScaler(float(y_sc.mean_[ka_idx]), float(y_sc.scale_[ka_idx]))},
        }
    v_idx = y_index["v"]
    fake_bundles["v"] = {
        "y_scaler": _FakeScaler(float(y_sc.mean_[v_idx]), float(y_sc.scale_[v_idx]))
    }

    fake_reac: dict = {}
    for i in (1, 2, 3):
        rx_idx = y_index["reac"][i]
        fake_reac[i] = {
            "y_scaler": _FakeScaler(float(y_sc.mean_[rx_idx]), float(y_sc.scale_[rx_idx]))
        }

    return fake_bundles, fake_reac

                   # SaFSimulator (three-electrode ARX)
class SaFSimulator(SimulatorInterface):
    """
    Three-electrode SAF furnace simulator using a joint ARX model.

    All 10 signals (R, kA, X per electrode, plus shared V) are predicted in a
    single forward pass at each time step. Cross-electrode effects (e.g. El2
    current draw affecting El1) are captured
    because the joint model sees all electrode states as inputs.

    The simulator maintains a state row (a pd.Series of lagged signal values).
    At each step, predictions are made from the current state, then lag registers
    are shifted to to add new the new values.

    Parameters
    arx_bundle : dict
        Loaded joint ARX bundle from arx_joint.joblib. Required keys:
        model, X_scaler, Y_scaler, X_cols, y_index.
    init_row : pd.Series
        Initial state row containing all lag columns. Build with
        build_init_row() or build_init_row_from_scalars().
    electrode : int
        Primary electrode index (1, 2, or 3). This electrode's R is the
        output exposed to Plant. The other two run in parallel to keep
        cross-electrode states accurate. Default 1.
    initial_position : float or None
        Override initial position of the primary electrode (m).
        If None, reads from init_row.
    """

    def __init__(
        self,
        arx_bundle:       dict,
        init_row:         pd.Series,
        electrode:        int = 1,
        initial_position: float | None = None,
    ) -> None:
        if electrode not in (1, 2, 3):
            raise ValueError(f"electrode must be 1, 2, or 3, got {electrode}")

        self._electrode    = electrode
        self._row          = init_row.copy()
        self._model        = arx_bundle["model"]
        self._x_sc         = arx_bundle["X_scaler"]
        self._y_sc         = arx_bundle["Y_scaler"]
        self._xcols        = arx_bundle["X_cols"]
        self._yidx         = arx_bundle["y_index"]
        self._cached_preds: np.ndarray | None = None

        # Find clip bounds (mean +/- 6*std) from Y_scaler.
        # Predictions outside this range are physically implausible and indicate
        # the model is extrapolating. Clipping prevents the kA -> X -> R -> kA
        # feedback loop from diverging in long free-running simulations.
        def _bounds(fake_bun, lo_floor=None):
            ys = fake_bun["y_scaler"]
            mu, sigma = float(ys.mean_[0]), float(ys.scale_[0])
            lo = mu - 6.0 * sigma
            if lo_floor is not None:
                lo = max(lo, lo_floor)
            return (lo, mu + 6.0 * sigma)

        _co = arx_bundle.get("clip_overrides")
        if _co:
            self._r_clip    = _co["r_clip"]
            self._ka_clip   = _co["ka_clip"]
            self._reac_clip = _co["reac_clip"]
            self._v_clip    = (0.0, 400.0)
        else:
            fake_bundles, fake_reac = _extract_clip_bounds(arx_bundle["Y_scaler"],
                                                            arx_bundle["y_index"])
            self._r_clip    = {i: _bounds(fake_bundles[i]["r"],  lo_floor=0.0) for i in (1, 2, 3)}
            self._ka_clip   = {i: _bounds(fake_bundles[i]["ka"], lo_floor=0.0) for i in (1, 2, 3)}
            self._reac_clip = {i: _bounds(fake_reac[i],          lo_floor=0.0) for i in (1, 2, 3)}
            self._v_clip    = _bounds(fake_bundles["v"],          lo_floor=0.0)

        # Primary electrode position
        pos_col = f"El{electrode}_pos_m_lag1"
        raw_pos = float(self._row.get(pos_col, 0.0))
        if initial_position is not None:
            self._u_pos = float(initial_position)
        elif raw_pos != 0.0:
            self._u_pos = raw_pos
        else:
            self._u_pos = _DEFAULT_POS

        # Per-electrode position register for dpos = u_new - u_prev
        self._pos = {i: float(self._row.get(f"El{i}_pos_m_lag1", _DEFAULT_POS))
                     for i in (1, 2, 3)}

    # SimulatorInterface identity properties
    @property
    def output_col(self) -> str:
        return f"El{self._electrode}_Resistance_mOhm_filt"

    @property
    def input_col(self) -> str:
        return f"El{self._electrode}_pos_m"

    @property
    def default_u0(self) -> float:
        return self._u_pos

    
    #                             State access
    def current_y(self) -> float:
        """Return the most recent stored R value for the primary electrode (mOhm)."""
        col = f"El{self._electrode}_y_filt_lag1"
        return float(self._row.get(col, 0.0))

    def current_u(self) -> float:
        """Return the current position of the primary electrode (m)."""
        return self._u_pos

    
    #             Joint ARX inference with step-level caching
    def _predict_all(self) -> np.ndarray:
        """
        Run one joint ARX forward pass and return all 10 signal predictions.

        The result is cached until advance() or advance_multi() shifts the lag
        registers. More calls within the same time step (e.g. from Plant
        computing ARX-only R, then again for the lag register update) hit the
        cache and skip the matrix multiply.
        """
        if self._cached_preds is not None:
            return self._cached_preds
        row = self._row
        x = np.array(
            [float(row[c]) if c in row.index else 0.0 for c in self._xcols],
            dtype=np.float64,
        ).reshape(1, -1)
        x_z = self._x_sc.transform(x)
        y_z = self._model.predict(x_z) # shape (1, 10)
        y   = self._y_sc.inverse_transform(y_z).ravel() # shape (10,) in physical units
        self._cached_preds = y
        return y

    def _predict_r(self, i: int) -> float:
        """Arc resistance (mOhm) for electrode i, clipped to valid range."""
        raw = float(self._predict_all()[self._yidx["R"][i]])
        return float(np.clip(max(raw, 0.0), *self._r_clip[i]))

    def _predict_ka(self, i: int) -> float:
        """Arc current (kA) for electrode i, clipped to valid range."""
        raw = float(self._predict_all()[self._yidx["kA"][i]])
        return float(np.clip(max(raw, 0.0), *self._ka_clip[i]))

    def _predict_reac(self, i: int) -> float:
        """Arc reactance (mOhm) for electrode i, clipped to valid range."""
        raw = float(self._predict_all()[self._yidx["reac"][i]])
        val = float(np.clip(max(raw, 0.0), *self._reac_clip[i]))
        return float(np.clip(val, 0.0, _REAC_CLIP_MAX))

    def _predict_v(self) -> float:
        """Transformer RMS voltage, clipped to valid range."""
        raw = float(self._predict_all()[self._yidx["v"]])
        return float(np.clip(max(raw, 0.0), *self._v_clip))

    def predict_next(self) -> float:
        """One-step-ahead R prediction for the primary electrode."""
        return self._predict_r(self._electrode)

   
    #                        Advance (single-electrode)
    def advance(self, u_new: float, y_new: float) -> None:
        """
        Advance all electrode lag registers one time step.

        u_new : new position command for the primary electrode (m).
        y_new : ARX R prediction to store in the primary electrode's R lag.
                Pass the simulator's own prediction here, not the GP-corrected
                value.

        Non-primary electrode positions are held constant. For full
        three-electrode control use advance_multi() instead.
        """
        row = self._row

        # Compute all new signal values from the current (old) state first,
        # before adjusting lag registers. All three electrodes read the same
        # state so the order of updates below does not affect predictions.
        new_ka   = {i: float(np.clip(max(self._predict_ka(i),   0.0), *self._ka_clip[i]))
                    for i in (1, 2, 3)}
        v_new_v  = float(np.clip(max(self._predict_v(), 0.0), *self._v_clip))
        new_reac = {i: self._predict_reac(i) for i in (1, 2, 3)}

        # Non-primary electrodes hold their current position constant.
        # Only the primary electrode moves.
        u_by_el = {i: self._pos[i] for i in (1, 2, 3)}
        u_by_el[self._electrode] = u_new

        # Primary electrode use y_new (passed in from Plant, the ARX-only R).
        # Other electrodes use the ARX prediction so they stay physically consistent.
        new_r = {
            el: y_new if el == self._electrode else self._predict_r(el)
            for el in (1, 2, 3)
        }

        # Shift all lag registers for all three electrodes.
        # The pattern is: lag3 <- lag2 <- lag1 <- new_value  (oldest value dropped)
        for i in (1, 2, 3):
            dpos_k = u_by_el[i] - self._pos[i]
            _shift_lags(row, f"El{i}_dpos_mps_filt_lag", dpos_k)
            _shift_lags(row, f"El{i}_pos_m_lag",         self._pos[i])
            self._pos[i] = u_by_el[i]
            _shift_lags(row, f"El{i}_y_filt_lag",        new_r[i])
            _shift_lags(row, f"El{i}_kA_filt_lag",       new_ka[i])
            _shift_lags(row, f"El{i}_CalcReac_filt_lag",
                        float(np.clip(new_reac[i], *self._reac_clip[i])))

        if "RMS_V_transformer_filt_lag1" in row.index:
            row["RMS_V_transformer_filt_lag1"] = v_new_v

        self._u_pos = u_new
        self._cached_preds = None

    
    #                    Advance (three-electrode MIMO)
    def advance_multi(
        self,
        u_new_vec: dict[int, float],
        y_new_vec: dict[int, float],
    ) -> None:
        """
        Full three-electrode advance provides positions and R values for all electrodes.

        Use this when running a three-electrode control loop where all electrode
        positions are commanded simultaneously.

        u_new_vec : {1: pos1, 2: pos2, 3: pos3}  (position commands in m)
        y_new_vec : {1: r1,   2: r2,   3: r3}    (ARX R predictions in mOhm)
                    Electrodes missing from y_new_vec are predicted internally.
        """
        row = self._row

        new_ka   = {i: float(np.clip(max(self._predict_ka(i),   0.0), *self._ka_clip[i]))
                    for i in (1, 2, 3)}
        v_new_v  = float(np.clip(max(self._predict_v(), 0.0), *self._v_clip))
        new_reac = {i: self._predict_reac(i) for i in (1, 2, 3)}
        new_r    = {i: y_new_vec.get(i, self._predict_r(i)) for i in (1, 2, 3)}

        for i in (1, 2, 3):
            u_i    = u_new_vec.get(i, self._pos[i])
            dpos_k = u_i - self._pos[i]
            _shift_lags(row, f"El{i}_dpos_mps_filt_lag", dpos_k)
            _shift_lags(row, f"El{i}_pos_m_lag",         self._pos[i])
            self._pos[i] = u_i
            _shift_lags(row, f"El{i}_y_filt_lag",        new_r[i])
            _shift_lags(row, f"El{i}_kA_filt_lag",       new_ka[i])
            _shift_lags(row, f"El{i}_CalcReac_filt_lag",
                        float(np.clip(new_reac[i], *self._reac_clip[i])))

        if "RMS_V_transformer_filt_lag1" in row.index:
            row["RMS_V_transformer_filt_lag1"] = v_new_v

        self._u_pos = u_new_vec.get(self._electrode, self._u_pos)
        self._cached_preds = None

    
    #                          GP feature extraction
    def get_gp_features(self) -> dict[str, float]:
        """Return the GP feature dict for the primary electrode."""
        return self.get_gp_features_electrode(self._electrode)

    def get_gp_features_electrode(self, i: int) -> dict[str, float]:
        """
        Build the GP feature vector for electrode i.

        Own features (12):  step_in_window and y_sim are placeholder zeros;
                            Plant fills them in before calling the GP.
        Cross features (9 x 2 = 18): state of the other two electrodes.

        The full set captures both the specific electrode dynamics and cross-electrode
        interactions.
        """
        row   = self._row
        other = [j for j in (1, 2, 3) if j != i]

        def _g(col: str, fallback: float = 0.0) -> float:
            return float(row[col]) if col in row.index else fallback

        feats: dict[str, float] = {
            "step_in_window":               0.0,   # filled by Plant
            "y_sim":                        0.0,   # filled by Plant
            f"El{i}_dpos_mps_filt_lag1":    _g(f"El{i}_dpos_mps_filt_lag1"),
            f"El{i}_dpos_mps_filt_lag2":    _g(f"El{i}_dpos_mps_filt_lag2"),
            f"El{i}_dpos_mps_filt_lag3":    _g(f"El{i}_dpos_mps_filt_lag3"),
            f"El{i}_pos_m_lag1":            _g(f"El{i}_pos_m_lag1", self._u_pos),
            f"El{i}_kA_filt_lag1":          _g(f"El{i}_kA_filt_lag1"),
            f"El{i}_kA_filt_lag2":          _g(f"El{i}_kA_filt_lag2"),
            f"El{i}_CalcReac_filt_lag1":    _g(f"El{i}_CalcReac_filt_lag1"),
            f"El{i}_CalcReac_filt_lag2":    _g(f"El{i}_CalcReac_filt_lag2"),
            _TC_COL[i]:                     _g(_TC_COL[i]),
            "RMS_V_transformer_filt_lag1":  _g("RMS_V_transformer_filt_lag1"),
        }
        for j in other:
            feats[f"El{j}_dpos_mps_filt_lag1"] = _g(f"El{j}_dpos_mps_filt_lag1")
            feats[f"El{j}_dpos_mps_filt_lag2"] = _g(f"El{j}_dpos_mps_filt_lag2")
            feats[f"El{j}_pos_m_lag1"]         = _g(f"El{j}_pos_m_lag1")
            feats[f"El{j}_y_filt_lag1"]        = _g(f"El{j}_y_filt_lag1")
            feats[f"El{j}_kA_filt_lag1"]       = _g(f"El{j}_kA_filt_lag1")
            feats[f"El{j}_kA_filt_lag2"]       = _g(f"El{j}_kA_filt_lag2")
            feats[f"El{j}_CalcReac_filt_lag1"] = _g(f"El{j}_CalcReac_filt_lag1")
            feats[f"El{j}_CalcReac_filt_lag2"] = _g(f"El{j}_CalcReac_filt_lag2")
            feats[_TC_COL[j]]                  = _g(_TC_COL[j])
        return feats

    def get_full_state(self) -> dict[str, float]:
        """
        Return all current signal values plus one-step-ahead predictions.

        Current values are read from the lag registers (what the simulator
        currently knows as "last measured"). Predicted values are the ARX
        one-step-ahead estimates for the next time step.

        Keys
        R{i}       : arc resistance lag1 for electrode i (mOhm)
        kA{i}      : arc current lag1 for electrode i (kA)
        X{i}       : arc reactance lag1 for electrode i (mOhm)
        pos{i}     : current electrode position for electrode i (m)
        V          : transformer RMS voltage (V)
        R_next{i}  : one-step-ahead ARX prediction of R for electrode i (mOhm)
        kA_next{i} : one-step-ahead ARX prediction of kA for electrode i (kA)
        X_next{i}  : one-step-ahead ARX prediction of reactance for electrode i (mOhm)
        V_next     : one-step-ahead ARX prediction of transformer voltage (V)
        """
        row   = self._row
        state: dict[str, float] = {}

        for i in (1, 2, 3):
            state[f"R{i}"]   = float(row.get(f"El{i}_y_filt_lag1",        0.0))
            state[f"kA{i}"]  = float(row.get(f"El{i}_kA_filt_lag1",       0.0))
            state[f"X{i}"]   = float(row.get(f"El{i}_CalcReac_filt_lag1", 0.0))
            state[f"pos{i}"] = self._pos[i]

        state["V"] = float(row.get("RMS_V_transformer_filt_lag1", 0.0))

        for i in (1, 2, 3):
            state[f"R_next{i}"]  = self._predict_r(i)
            state[f"kA_next{i}"] = self._predict_ka(i)
            state[f"X_next{i}"]  = self._predict_reac(i)

        state["V_next"] = self._predict_v()
        return state

    def set_feature(self, col: str, val: float) -> None:
        """Overwrite a column in the state row (for injecting real measurements)."""
        self._row[col] = val

#                 Helper: build init row from a data segment
def build_init_row(
    seg:        pd.DataFrame,
    t:          int,
    arx_bundle: dict,
) -> pd.Series:
    """
    Build the initial state row for SaFSimulator from a real data segment.

    Reads lagged signal values from the segment at time t so the
    simulator starts from a physically realistic state. Use this when you
    have a history segment to replay, or you want to start from a specific point in history.

    The segment must contain these columns (one set per electrode i in 1, 2, 3):
        El{i}_dpos_f   : filtered electrode velocity (m/s)
        El{i}_pos_m    : electrode position (m)
        El{i}_y_filt   : filtered arc resistance (mOhm)
        El{i}_kA_f     : filtered arc current (kA)
        El{i}_reac_f   : filtered arc reactance (mOhm)
    And shared columns:
        rms_v_f        : filtered transformer RMS voltage
        tca, tcb, tcc  : tap-changer positions

    Parameters
    seg        : pd.DataFrame with at least t+1 rows
    t          : row index to treat as the current time step
    arx_bundle : joint ARX bundle (only X_cols is used here)
    """

    def _lag(arr: np.ndarray, k: int) -> float:
        return float(arr[max(0, t - k)])

    tca = seg["tca"].values    if "tca"    in seg.columns else np.zeros(len(seg))
    tcb = seg["tcb"].values    if "tcb"    in seg.columns else np.zeros(len(seg))
    tcc = seg["tcc"].values    if "tcc"    in seg.columns else np.zeros(len(seg))
    rv  = seg["rms_v_f"].values if "rms_v_f" in seg.columns else np.zeros(len(seg))

    lookup: dict[str, float] = {
        "RMS_V_transformer_filt_lag1": _lag(rv, 1),
        "TCA": float(tca[t]),
        "TCB": float(tcb[t]),
        "TCC": float(tcc[t]),
    }

    for i in (1, 2, 3):
        d  = seg[f"El{i}_dpos_f"].values  if f"El{i}_dpos_f"  in seg.columns else np.zeros(len(seg))
        pm = seg[f"El{i}_pos_m"].values   if f"El{i}_pos_m"   in seg.columns else np.zeros(len(seg))
        yf = seg[f"El{i}_y_filt"].values  if f"El{i}_y_filt"  in seg.columns else np.zeros(len(seg))
        ka = seg[f"El{i}_kA_f"].values    if f"El{i}_kA_f"    in seg.columns else np.zeros(len(seg))
        rx = seg[f"El{i}_reac_f"].values  if f"El{i}_reac_f"  in seg.columns else np.zeros(len(seg))

        lookup[f"El{i}_dpos_mps_filt_lag1"] = _lag(d,  1)
        lookup[f"El{i}_dpos_mps_filt_lag2"] = _lag(d,  2)
        lookup[f"El{i}_dpos_mps_filt_lag3"] = _lag(d,  3)
        lookup[f"El{i}_pos_m_lag1"]         = _lag(pm, 1)
        lookup[f"El{i}_y_filt_lag1"]        = _lag(yf, 1)
        lookup[f"El{i}_y_filt_lag2"]        = _lag(yf, 2)
        lookup[f"El{i}_y_filt_lag3"]        = _lag(yf, 3)
        lookup[f"El{i}_kA_filt_lag1"]       = _lag(ka, 1)
        lookup[f"El{i}_kA_filt_lag2"]       = _lag(ka, 2)
        lookup[f"El{i}_kA_filt_lag3"]       = _lag(ka, 3)
        lookup[f"El{i}_CalcReac_filt_lag1"] = _lag(rx, 1)
        lookup[f"El{i}_CalcReac_filt_lag2"] = _lag(rx, 2)
        lookup[f"El{i}_CalcReac_filt_lag3"] = _lag(rx, 3)

    all_cols = sorted(set(arx_bundle["X_cols"]) | set(lookup.keys()))
    return pd.Series({c: lookup.get(c, 0.0) for c in all_cols})

#           Helper: build init row from scalar operating point values
def build_init_row_from_scalars(
    pos:        float,
    r:          float,
    ka:         float,
    rx:         float,
    v:          float,
    arx_bundle: dict,
    tca:        float = 4.0,
    tcb:        float = 4.0,
    tcc:        float = 4.0,
    electrode:  int   = 1,
) -> pd.Series:
    """
    Build the initial state row from scalar operating point values.

    All lag registers are filled with these constant values (Furnace was in this state for 0<t). The simulator
    will warm up over the first ~3 steps as lag history builds naturally.
    Use this when starting simulation from a known operating point without a
    recorded data segment.

    Parameters
    pos  : electrode position (m), applied to the primary electrode.
           Other electrodes are also initialised to this position.
    r    : arc resistance (mOhm)
    ka   : arc current (kA)
    rx   : arc reactance (mOhm)
    v    : transformer RMS voltage
    arx_bundle : joint ARX bundle (only X_cols is used to determine required columns)
    tca, tcb, tcc : tap-changer positions (default 4.0, typical mid-range)
    electrode : primary electrode index (1, 2, or 3)
    """
    lookup: dict[str, float] = {
        "RMS_V_transformer_filt_lag1": v,
        "TCA": tca, "TCB": tcb, "TCC": tcc,
    }
    for i in (1, 2, 3):
        for lag in range(1, _MAX_LAG + 1):
            lookup[f"El{i}_dpos_mps_filt_lag{lag}"] = 0.0    # no movement at init
            lookup[f"El{i}_pos_m_lag{lag}"]          = pos
            lookup[f"El{i}_y_filt_lag{lag}"]         = r
            lookup[f"El{i}_kA_filt_lag{lag}"]        = ka
            lookup[f"El{i}_CalcReac_filt_lag{lag}"]  = rx

    all_cols = sorted(set(arx_bundle["X_cols"]) | set(lookup.keys()))
    return pd.Series({c: lookup.get(c, 0.0) for c in all_cols})
