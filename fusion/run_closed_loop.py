"""
run_closed_loop.py
Drop-in replacement for meta_arx/run_simulation/scripts/run_closed_loop.py.

Uses the joint ARX simulator and per-electrode GP correction instead of
the original single-electrode ARX model. Everything else is identical, same
function signature, same CSV output format, same controller types.

HOW TO USE
In VRFT v5.py change one import line:

    # Old:
    from run_simulation.scripts.run_closed_loop_fusion import run_closed_loop_from_config

    # New (joint ARX + GP):
    from fusion.run_closed_loop import run_closed_loop_from_config

Everything else in VRFT v5.py stays exactly the same.

Output CSV columns
  t_s           : time in seconds
  y1, y2, y3   : predicted arc resistance per electrode (mOhm)
  r1, r2, r3   : reference signal per electrode (mOhm)
  u1, u2, u3   : electrode position commands (m)
  e1, e2, e3   : controller error per electrode (reference - y_pred)
  v_transformer : transformer RMS voltage (V)

Reference CSV format
  Columns r1, r2, r3  (per-electrode references, preferred)
  Column  reference   (single reference broadcast to all three electrodes)

Typical operating point used as initial state
  position   : 1.04 m
  resistance : 1.006 mOhm
  current    : 65 kA
  reactance  : 0.82 mOhm
  voltage    : 165 V
"""
from __future__ import annotations

import os
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE         = Path(__file__).resolve().parent          # fusion/ (this package)
_PROJECT_ROOT = _HERE.parent                             # SAF_data_to_control/
_META_ARX     = _PROJECT_ROOT / "meta_arx"

for _p in [str(_PROJECT_ROOT), str(_META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
from .training.delta_arx import DeltaARXWrapper  # noqa: F401  needed for joblib unpickling of v14/v15

from .simulators.saf_simulator import SaFSimulator, build_init_row_from_scalars
from .training.gp_loader import load_gp_bundle, predict_single, predict_single_certainty

# MODEL SELECTION: change _GP_VARIANT to switch between plant models.
# Each variant is a matched ARX + GP pair trained on the same dataset.
_GP_VARIANT = "v18"

# ARX model paired with each GP variant (do not change unless you retrain):
_ARX_FOR_VARIANT = {
    "txt2026_512":       "arx_joint_txt2026.joblib",     # ARX trained on 2026 txt data
    "pi_512":            "arx_joint_pi_v3.joblib",       # ARX trained on PI data
    "combined_deep_512": "arx_joint_combined_v3.joblib", # ARX trained on PI + txt combined, deep kernel
    "combined_512":      "arx_joint_combined_v3.joblib", # ARX trained on PI + txt combined, Matern32 kernel
    "v6":                "arx_joint_v6.joblib",           # V6 joint ARX, debiased GP
    "v7":                "arx_joint_v6.joblib",           # V7 two-stage: linear correction + GP
    "v8":                "arx_joint_v8.joblib",           # V8 full-dataset retrain (10/80/10 split)
    "v9":                "arx_joint_v9.joblib",           # V9 step-episode filtered ARX + GP
    "rollout":           "arx_joint_v9.joblib",           # SVGP trained on H=1000 rollout windows (PI data)
    "v11":               "arx_joint_pi_v3.joblib",        # V11 one-step, no step_in_window (MPC-focused)
    "v12":               "arx_joint_v12.joblib",          # V12 SEM rollout, retrained ARX + GP
    "v13":               "arx_joint_v13.joblib",          # V13 = V12 without own-R lags (fixes drift)
    "v14":               "arx_joint_v14.joblib",          # V14 per-electrode Ridge, delta-R target
    "v15":               "arx_joint_v15.joblib",          # V15 delta-R, 5 lags, all infra fixes
    "v15b":              "arx_joint_v15.joblib",          # V15b = V15 ARX + one-step-anchored GP
    "v15s":              "arx_joint_v15_stable.joblib",   # V15s = V15 with R-lag coefs zeroed (stable open-loop)
    "v16":               "arx_joint_v16.joblib",          # V16 delta-R, no R-lag features (architecturally stable)
    "v16a":              "arx_joint_v16.joblib",          # V16a backup, 10-epoch GP (before 60-epoch retrain)
    "v18":               "arx_joint_v17.joblib",          # V18 rollout SVGP, V17 joint ARX (rank=10), 42 features
}

# V7 uses a lightweight linear residual model before the GP.
# The linear model removes most of the ARX residual variance; the GP then
# learns only the remaining nonlinear component.
_HAS_LINEAR_STAGE = {"v7"}

# OOD gate: when the mean norm_var across all electrodes exceeds this threshold,
# the controller holds the previous position and the integrators are frozen.
# Set to 1.0 to disable the gate entirely.
_OOD_GATE_THRESHOLD = 0.5

_ARX_MODEL = _ARX_FOR_VARIANT[_GP_VARIANT]
# Typical SAF operating point used to seed the initial simulator state.
# Per-electrode R values come from a stable rule-controller run. The three
# electrodes sit at different resistances in steady state.
_TYPICAL_POS    = 1.04
_TYPICAL_R      = 1.006   # fallback scalar (not used for multi-electrode init)
_TYPICAL_KA     = 65.0
_TYPICAL_REAC   = 0.82
_TYPICAL_V      = 165.0
_TYPICAL_POS_BY_EL = {1: 1.04,  2: 1.03,  3: 1.04}
_TYPICAL_R_BY_EL   = {1: 1.20,  2: 0.77,  3: 1.14}

# V6 and later were trained on PI furnace data at a different operating point.
# Using the correct kA and R values prevents the ARX from starting far outside
# its training distribution and immediately diverging.
_TYPICAL_KA_FOR    = {"v6": 118.0, "v7": 118.0, "v8": 118.0, "v9": 118.0,
                      "rollout": 118.0, "v11": 118.0, "v12": 118.0, "v13": 118.0,
                      "v14": 118.0, "v15": 118.0, "v15b": 118.0, "v15s": 118.0,
                      "v16": 118.0, "v16a": 118.0, "v18": 118.0}
_TYPICAL_REAC_FOR  = {"v6": 0.88,  "v7": 0.88,  "v8": 0.88,  "v9": 0.88,
                      "rollout": 0.88, "v11": 0.88, "v12": 0.88, "v13": 0.88,
                      "v14": 0.88, "v15": 0.88, "v15b": 0.88, "v15s": 0.88,
                      "v16": 0.88, "v16a": 0.88, "v18": 0.88}
_TYPICAL_R_BY_EL_FOR = {
    "v6":      {1: 1.08, 2: 1.07, 3: 1.07},
    "v7":      {1: 1.08, 2: 1.07, 3: 1.07},
    "v8":      {1: 1.08, 2: 1.07, 3: 1.07},
    "v9":      {1: 1.08, 2: 1.07, 3: 1.07},
    "rollout": {1: 1.08, 2: 1.07, 3: 1.07},
    "v11":     {1: 1.08, 2: 1.07, 3: 1.07},
    "v12":     {1: 1.08, 2: 1.07, 3: 1.07},
    "v13":     {1: 1.08, 2: 1.07, 3: 1.07},
    "v14":     {1: 1.08, 2: 1.07, 3: 1.07},
    "v15":     {1: 1.08, 2: 1.07, 3: 1.07},
    "v15s":    {1: 1.08, 2: 1.07, 3: 1.07},
    "v16":     {1: 1.08, 2: 1.07, 3: 1.07},
    "v16a":    {1: 1.08, 2: 1.07, 3: 1.07},
    "v18":     {1: 1.08, 2: 1.07, 3: 1.07},
}

# The "rollout" SVGP was trained in R_tilde space (R - initial_R).
# At inference, the initial R from the simulator is used as the mean so that
# R_tilde_approx = R_abs - R_initial ≈ 0 at the operating point.
_NEEDS_RTILDE_APPROX = {"rollout"}

# Variants where cross-electrode R lags are frozen at training-set means before
# each ARX forward pass. This breaks the coupling cascade that makes PID unstable.
# The means are stored in the ARX bundle under "r_cross_mean".
_DECOUPLED_R_VARIANTS = {"v13"}

def _load_reference(path: str | Path) -> np.ndarray:
    """
    Read reference signal(s) from a CSV file.

    Returns an (n, 3) array.  If the CSV has columns r1/r2/r3 they are used
    directly.  If it has a single 'reference' column it is broadcast to all
    three electrodes.  The first numeric column is used as a last resort.
    """
    df = pd.read_csv(path)

    if all(f"r{i}" in df.columns for i in (1, 2, 3)):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)

    if "reference" in df.columns:
        ref = df["reference"].to_numpy(dtype=float)
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            raise ValueError(f"Reference CSV {path} has no numeric column.")
        ref = df[num_cols[0]].to_numpy(dtype=float)

    return np.column_stack([ref, ref, ref])   # broadcast scalar ref to all 3 electrodes

def _build_sim_and_gps(
    gp_variant: str = _GP_VARIANT,
) -> "tuple[SaFSimulator, dict, dict]":
    """
    Load the joint ARX model, per-electrode GP bundles, and (for V7) linear
    residual correction models.

    Returns (sim, gp_bundles, linear_models).
    gp_bundles    : {1: bundle, 2: bundle, 3: bundle}, missing keys use ARX only
    linear_models : {1: model, 2: model, 3: model}, empty unless variant is V7
    """
    import __main__
    from .training.arx_model import ReducedRankRidge

    # The included ARX bundle was saved when the training script was run
    # directly, so Python stored the class as __main__.ReducedRankRidge.
    # We inject the class here so joblib can find it regardless of what
    # __main__ is now.
    # Bundles retrained with train_arx.py use the proper module path and
    # load without this step.
    if not hasattr(__main__, "ReducedRankRidge"):
        __main__.ReducedRankRidge = ReducedRankRidge

    if gp_variant not in _ARX_FOR_VARIANT:
        raise ValueError(
            f"Unknown gp_variant {gp_variant!r}. "
            f"Choose from: {list(_ARX_FOR_VARIANT)}"
        )

    models_dir = _HERE / "models"
    arx_name   = _ARX_FOR_VARIANT[gp_variant]
    arx_path   = models_dir / arx_name

    if not arx_path.exists():
        raise FileNotFoundError(
            f"ARX model not found: {arx_path}\n"
            f"Expected: fusion/models/{arx_name}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arx = joblib.load(arx_path)

    ka      = _TYPICAL_KA_FOR.get(gp_variant, _TYPICAL_KA)
    rx      = _TYPICAL_REAC_FOR.get(gp_variant, _TYPICAL_REAC)
    r_by_el = _TYPICAL_R_BY_EL_FOR.get(gp_variant, _TYPICAL_R_BY_EL)

    init_row = build_init_row_from_scalars(
        pos        = _TYPICAL_POS,
        r          = r_by_el[1],
        ka         = ka,
        rx         = rx,
        v          = _TYPICAL_V,
        arx_bundle = arx,
    )

    # Override per-electrode R and position lags so each electrode starts at
    # its natural equilibrium rather than an identical symmetric value.
    for _i in (1, 2, 3):
        for _lag in (1, 2, 3):
            init_row[f"El{_i}_y_filt_lag{_lag}"]  = r_by_el[_i]
            init_row[f"El{_i}_pos_m_lag{_lag}"]    = _TYPICAL_POS_BY_EL[_i]

    sim = SaFSimulator(arx, init_row, electrode=1)
    print(f"[fusion] ARX loaded: {arx_name}  (variant={gp_variant})")

    gp_bundles: dict = {}
    for i in (1, 2, 3):
        gp_path = models_dir / f"gp_el{i}_{gp_variant}.pt"
        if gp_path.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_bundles[i] = load_gp_bundle(str(gp_path))
            print(f"[fusion] El{i} GP loaded: {gp_path.name}  "
                  f"features={len(gp_bundles[i]['feature_names'])}")
        else:
            print(f"[fusion] El{i}: GP not found at {gp_path.name}, running ARX only.")

    linear_models: dict = {}
    if gp_variant in _HAS_LINEAR_STAGE:
        for i in (1, 2, 3):
            lin_path = models_dir / f"linear_residual_el{i}.joblib"
            if lin_path.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    linear_models[i] = joblib.load(lin_path)
                print(f"[fusion] El{i} linear residual loaded: {lin_path.name}")
            else:
                print(f"[fusion] El{i}: linear residual not found at {lin_path.name}, skipping.")

    return sim, gp_bundles, linear_models

class _RollingFeatures:
    """
    Maintains rolling-window statistics that the ARX simulator does not track.

    The V6/V7 GP bundles use 30-step rolling std of R and CalcReac per electrode,
    extra dpos velocity lags (lag4/5 for El1), R imbalance, and TCA_diff.
    The simulator only keeps lag1-3 for dpos and has no rolling history, so we
    maintain the buffers here and inject the computed values into sim._row before
    each GP inference call.
    """

    _W = 30  # rolling window length in steps

    def __init__(self):
        self._r    = {i: deque(maxlen=self._W) for i in (1, 2, 3)}
        self._reac = {i: deque(maxlen=self._W) for i in (1, 2, 3)}
        self._dpos1 = deque(maxlen=5)   # El1 dpos history for lag4/5

    def update(self, row: dict, y_override: "dict | None" = None) -> None:
        """Record current simulator state into the rolling buffers.

        y_override: per-electrode {1: r1, 2: r2, 3: r3} to use as the R value
        instead of y_filt_lag1.  Pass the GP-corrected y_pred here when y-lags
        are frozen so rolling_std_R stays alive and does not collapse to zero.
        """
        for i in (1, 2, 3):
            r_val = (float(y_override[i]) if y_override and i in y_override
                     else row.get(f"El{i}_y_filt_lag1", 0.0))
            self._r[i].append(r_val)
            self._reac[i].append(row.get(f"El{i}_CalcReac_filt_lag1", 0.0))
        self._dpos1.append(row.get("El1_dpos_mps_filt_lag1", 0.0))

    def inject(self, row: dict) -> None:
        """Write computed rolling features into sim._row before GP inference."""
        for i in (1, 2, 3):
            r_arr    = np.array(self._r[i])
            reac_arr = np.array(self._reac[i])
            row[f"El{i}_rolling_std_R_30s"]        = float(np.std(r_arr))    if len(r_arr)    > 1 else 0.0
            row[f"El{i}_rolling_std_CalcReac_30s"] = float(np.std(reac_arr)) if len(reac_arr) > 1 else 0.0

        # R imbalance per electrode: deviation from the three-electrode mean.
        # Computed for all three so V11/V12 GP models (which use El{i}_R_imbalance
        # for i=1,2,3) see the correct value rather than defaulting to 0.0.
        r_now = [float(self._r[i][-1]) if self._r[i] else row.get(f"El{i}_y_filt_lag1", 0.0)
                 for i in (1, 2, 3)]
        r_mean_now = np.mean(r_now)
        for _i_el, _r_val in enumerate(r_now, start=1):
            row[f"El{_i_el}_R_imbalance"] = float(_r_val - r_mean_now)

        # No tap changes in simulation, so TCA_diff is always 0
        row["TCA_diff"] = 0.0

        # El1 dpos lags 4 and 5 (simulator only keeps lag1-3)
        d = list(self._dpos1)
        row["El1_dpos_mps_filt_lag4"] = d[-4] if len(d) >= 4 else 0.0
        row["El1_dpos_mps_filt_lag5"] = d[-5] if len(d) >= 5 else 0.0

        # CosPhi = R / |Z| = R / sqrt(R^2 + X^2) per electrode
        for _j in (1, 2, 3):
            _r  = float(row.get(f"El{_j}_y_filt_lag1", 0.0))
            _x  = float(row.get(f"El{_j}_CalcReac_filt_lag1", 0.0))
            _z2 = _r * _r + _x * _x
            row[f"El{_j}_CosPhi"] = float(_r / _z2 ** 0.5) if _z2 > 1e-6 else 0.0

def _gp_corrected_r(
    sim:             SaFSimulator,
    gp_bundles:      dict,
    electrode:       int,
    plant_cache:     dict,
    step:            int,
    linear_models:   "dict | None" = None,
    gp_variant:      str = "",
    r_initial:       "dict | None" = None,
    gp_ramp_offset:  int = 0,
    gp_scale:        float = 1.0,
    gp_bias:         float = 0.0,
) -> "tuple[float, float, float, float]":
    """
    Return (R_corrected, gp_variance, norm_var, ind_dist) for one electrode.

    norm_var : epistemic variance normalised to [0, 1], 0 = confident, 1 = OOD
    ind_dist : min L2 distance to nearest inducing point (standardised space)

    For V7, a linear residual correction is applied first, then the GP corrects
    the remaining nonlinear component.  The total correction (linear + GP) is
    clipped to +/-0.15 mOhm to limit extrapolation damage.
    """
    sim._electrode = electrode
    y_arx = sim._predict_r(electrode)
    bun   = gp_bundles.get(electrode)
    if bun is None:
        return y_arx, 0.0, 0.0, 0.0

    # V6/V7 use velocity-based features (dpos_mps lags) that live directly in
    # sim._row; the V6 joint ARX tracks them automatically. Older variants
    # need the legacy feature-builder which adds y_sim, y_real_lag etc.
    if any("dpos_mps" in f for f in bun["feature_names"]):
        # Clamp step_in_window to the GP's training range so steps beyond
        # ROLLOUT_H see the learned steady-state correction, not OOD extrapolation.
        _siw_max = float(bun.get("metadata", {}).get("rollout_H", 19))
        sim._row["step_in_window"] = float(min(step, _siw_max - 1))
        sim._row["y_sim"]          = y_arx
        sim._row["y_sim_sq"]       = y_arx * y_arx
        # TCA/TCB/TCC: use the value already in sim._row (set by tap auto-correction).
        # The previous override to training mean prevented tap changes from taking
        # effect during simulation.  The tap lookup now seeds the correct value
        # at initialisation and updates it each step in run_locked_closed_loop,
        # so sim._row already holds the right tap setting.
        # Only fall back to training mean if the tap lookup is not present
        # (i.e. the tap has not been set from outside).
        _tap_lookup_present = (_HERE / "models" / "tap_lookup.json").exists()
        if not _tap_lookup_present:
            for _j, _fname in enumerate(bun["feature_names"]):
                if _fname in ("TCA", "TCB", "TCC"):
                    sim._row[_fname] = float(bun["x_mean"][_j])

        # Apply per-electrode R-lag demeaning if the bundle was trained with it
        _r_op = bun.get("r_op_offset", {})
        raw_x = []
        for f in bun["feature_names"]:
            val = float(sim._row.get(f, 0.0))
            if _r_op:
                for _i in (1, 2, 3):
                    if f.startswith(f"El{_i}_y_filt_lag"):
                        val -= _r_op.get(_i, 0.0)
                        break
            raw_x.append(val)
        x = np.array(raw_x, dtype=np.float32)
        delta_mean = float(bun.get("metadata", {}).get("delta_mean", 0.0))

        lin_delta = 0.0
        if linear_models and electrode in linear_models:
            lin = linear_models[electrode]
            x_norm = (x.astype(np.float64) - lin["x_mean"]) / lin["x_std"]
            lin_delta = float(np.dot(x_norm, lin["coef"]) + lin["delta_mean"])

        mu, var, norm_var, ind_dist = predict_single_certainty(bun, x)
        correction = float(np.clip(lin_delta + mu + delta_mean, -0.15, 0.15))
        # Subtract operating-point bias; scale by gp_scale (0 = ARX-only).
        correction = (correction - gp_bias) * gp_scale

        # Ramp GP correction in over the first _RAMP steps after the hold
        # phase ends. The zero-dpos cold start is slightly out of distribution
        # for rollout-trained GPs; ramping avoids a large initial transient.
        _RAMP      = 30
        _ramp_step = step - gp_ramp_offset
        if gp_variant in ("v14", "v15", "v15b") and _ramp_step < _RAMP:
            correction *= max(0.0, float(_ramp_step)) / _RAMP

        return float(y_arx + correction), float(var), float(norm_var), float(ind_dist)

    else:
        feats = sim.get_gp_features_electrode(electrode)
        feats["step_in_window"] = float(min(step, 19))
        feats["y_sim"]          = y_arx
        feats["y_sim_sq"]       = y_arx * y_arx
        feats["y_real_lag1"]    = plant_cache.get(f"r{electrode}",      y_arx)
        feats["y_real_lag2"]    = plant_cache.get(f"r{electrode}_lag2", y_arx)

        x = np.array([feats.get(f, 0.0) for f in bun["feature_names"]], dtype=np.float32)

        lin_delta = 0.0
        if linear_models and electrode in linear_models:
            lin = linear_models[electrode]
            x_norm = (x.astype(np.float64) - lin["x_mean"]) / lin["x_std"]
            lin_delta = float(np.dot(x_norm, lin["coef"]) + lin["delta_mean"])

        mu, var, norm_var, ind_dist = predict_single_certainty(bun, x)
        mu = float(np.clip(lin_delta + mu, -0.15, 0.15))
        return float(y_arx + mu), float(var), float(norm_var), float(ind_dist)

def _apply_tap_from_reference(sim, reference: "np.ndarray") -> None:
    """Set TCA/TCB/TCC in sim._row and initialise R lags to the expected
    equilibrium for the chosen tap setting.

    The tap changer shifts the transformer voltage which sets a new R
    operating point.  Without also updating the initial R lags the ARX
    starts from the hardcoded thesis OP (1.08/1.07/1.07) even when a
    very different tap is requested, and the simulation never reaches the
    correct starting equilibrium.

    Called from both run_closed_loop_from_config and
    run_locked_closed_loop_from_config.
    """
    _tap_lookup_path = _HERE / "models" / "tap_lookup.json"
    if not _tap_lookup_path.exists():
        return
    try:
        from fusion.training.tap_lookup import TapLookup
        import numpy as _np
        _tl   = TapLookup.load(_tap_lookup_path)
        _tc   = {1: "TCA", 2: "TCB", 3: "TCC"}
        _taps = {}
        for _i in (1, 2, 3):
            # Use the first reference value (t=0) not the mean so the
            # initial condition matches the starting setpoint, not the average
            _r_start  = float(reference[0, _i - 1])
            _tap_val  = _tl.get_tap(_i, _r_start)
            _r_eq     = _tl._tables[_i].get(_tap_val, _r_start)
            _taps[_i] = _tap_val
            _col = _tc[_i]
            if _col in sim._row.index:
                sim._row[_col] = _tap_val
            # Seed R lag registers to the tap's expected equilibrium so the
            # ARX starts from the correct operating point
            for _lag in range(1, 11):
                _rc = f"El{_i}_y_filt_lag{_lag}"
                if _rc in sim._row.index:
                    sim._row[_rc] = _r_eq
        print(f"[fusion] Auto-tap (t=0 ref):  "
              f"TCA={_taps[1]:.0f} (R={_tl._tables[1][_taps[1]]:.3f})  "
              f"TCB={_taps[2]:.0f} (R={_tl._tables[2][_taps[2]]:.3f})  "
              f"TCC={_taps[3]:.0f} (R={_tl._tables[3][_taps[3]]:.3f})")
    except Exception as _e:
        print(f"[fusion] Tap auto-set skipped: {_e}")


def _seed_dpos_lags(sim, gp_variant: str) -> None:
    """Placeholder for variant-specific dpos initialisation. Currently no-op."""
    pass

def run_closed_loop_from_config(
    ref_csv:           str | Path,
    controller_name:   str,
    controller_config: str | Path,
    out_csv:           str | Path,
    dt:                float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Run a three-electrode closed-loop simulation with the joint ARX + GP plant.

    Parameters
    ref_csv           : CSV with reference signal(s).
                        Columns r1/r2/r3 for per-electrode refs, or a single
                        'reference' column broadcast to all three electrodes.
    controller_name   : Controller type, e.g. "pid", "relay", or "open_loop".
    controller_config : CSV with controller parameters (kp, ki, kd for PID).
    out_csv           : Where to write the simulation output CSV.
    dt                : Sample time in seconds (default 1.0).
    gp_variant        : Optional. Override the module-level _GP_VARIANT for this call.
                        Accepted values: "txt2026_512", "pi_512", "combined_512",
                        "combined_deep_512", "v6", "v7", "v8", "v9".
    **kwargs          : Other keyword arguments are ignored (compatibility).

    Returns
    pd.DataFrame with columns:
        t_s, y1, y2, y3, r1, r2, r3, u1, u2, u3, e1, e2, e3, v_transformer
    """
    from run_simulation.closed_loop.closed_loop_sim import apply_actuator_limits

    # Relay uses a per-electrode step(float, float, float) API and is loaded here
    # directly so it does not need to be registered in controller_registry.
    # Everything else uses the unified step(ndarray, dict, ndarray) API from
    # make_controllers().
    if controller_name == "relay":
        from .controllers.relay import RelayController, load_relay_params
        controllers = [RelayController(**p) for p in load_relay_params(controller_config)]
    elif controller_name == "decoupled_pid":
        # Standard PID but electrode position changes are pre-multiplied by the
        # static decoupling matrix D = G^{-1} computed from the ARX gain at OP.
        # Each PID sees an independent channel; cross-coupling cancellation is
        # absorbed into the actual electrode commands.
        controller_name = "pid"   # use standard PID internally
        from run_simulation.closed_loop.controller_registry import make_controllers
        controllers = make_controllers(
            name="pid", config_path=controller_config, dt=dt
        )
        # Decoupler will be applied below; flag it
        controller_name = "decoupled_pid"
    else:
        try:
            from run_simulation.closed_loop.controller_registry import make_controllers
            controllers = make_controllers(
                name=controller_name, config_path=controller_config, dt=dt
            )
        except (ImportError, AttributeError):
            from run_simulation.closed_loop.controller_registry import make_controller
            controllers = [
                make_controller(name=controller_name, config_path=controller_config, dt=dt)
                for _ in (1, 2, 3)
            ]

    # Relay and the legacy fallback return a list; everything from make_controllers()
    # returns a single unified controller object.
    _unified = not isinstance(controllers, list)

    gp_variant = kwargs.pop("gp_variant", _GP_VARIANT)
    sim, gp_bundles, linear_models = _build_sim_and_gps(gp_variant)
    reference  = _load_reference(ref_csv)

    if kwargs.pop("auto_tap", True):
        _apply_tap_from_reference(sim, reference)
        _seed_dpos_lags(sim, gp_variant)

    # Compute static decoupling matrix for decoupled_pid mode.
    # G[i,j] = dR_i / d(dpos_j) at the operating point (one relay step perturbation).
    # D = G^{-1} transforms independent PID outputs into coordinated electrode commands.
    _decoupler: "np.ndarray | None" = None
    if controller_name == "decoupled_pid":
        _DELTA = 0.01
        R_nom  = np.array([sim._predict_r(i) for i in (1, 2, 3)])
        G      = np.zeros((3, 3))
        for _j in (1, 2, 3):
            _row_pert = sim._row.copy()
            _row_pert[f"El{_j}_dpos_mps_filt_lag1"] = _DELTA
            _sim_pert = type(sim)(sim._model.__class__.__new__(sim._model.__class__),
                                  _row_pert, electrode=1) if False else None
            # Simpler: temporarily set the lag in sim, predict, restore
            _orig = float(sim._row.get(f"El{_j}_dpos_mps_filt_lag1", 0.0))
            sim._row[f"El{_j}_dpos_mps_filt_lag1"] = _DELTA
            sim._cached_preds = None
            R_pert = np.array([sim._predict_r(i) for i in (1, 2, 3)])
            G[:, _j - 1] = (R_pert - R_nom) / _DELTA
            sim._row[f"El{_j}_dpos_mps_filt_lag1"] = _orig
            sim._cached_preds = None
        try:
            _decoupler = np.linalg.inv(G)
            print(f"[fusion] Decoupler D = G^-1 computed at OP.")
            print(f"[fusion]   G diag (mOhm/m): {np.diag(G).round(3)}")
        except np.linalg.LinAlgError:
            print("[fusion] WARNING: gain matrix singular, using pseudoinverse.")
            _decoupler = np.linalg.pinv(G)

    # Load cross-R freeze means once for decoupled variants (not per step).
    _r_cross_mean: dict[int, float] = {}
    if gp_variant in _DECOUPLED_R_VARIANTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _arx_bun_dec = joblib.load(
                _HERE / "models" / _ARX_FOR_VARIANT[gp_variant])
        _r_cross_mean = _arx_bun_dec.get("r_cross_mean", {})
        print(f"[fusion] V13 decoupled: cross-R means = {_r_cross_mean}")
    # reference already loaded above for tap auto-set; n is needed below.

    # For the "rollout" SVGP, capture the initial state per electrode.
    # All lag slots that the simulator doesn't track (lags 4-10 for R and pos,
    # lags 4-10 for kA) are filled with their initial values so the GP sees
    # a properly warm-started state at t=0 rather than a zero-filled history.
    r_initial: "dict | None" = None
    if gp_variant in _NEEDS_RTILDE_APPROX:
        r_initial = {
            "R":   {i: float(sim._row.get(f"El{i}_y_filt_lag1",    0.0)) for i in (1,2,3)},
            "pos": {i: float(sim._row.get(f"El{i}_pos_m_lag1",      1.04)) for i in (1,2,3)},
            "kA":  {i: float(sim._row.get(f"El{i}_kA_filt_lag1", 118.0)) for i in (1,2,3)},
        }
        print(f"[fusion] rollout SVGP: R_initial={r_initial['R']}  "
              f"pos={r_initial['pos']}  kA={r_initial['kA']}")

    if _unified:
        controllers.reset()
    else:
        for c in controllers:
            c.reset()

    n              = len(reference)
    y              = np.zeros((n + 1, 3))   # predicted R per electrode (mOhm)
    gp_var_arr     = np.zeros((n + 1, 3))   # GP predictive variance per electrode
    norm_var_arr   = np.zeros((n + 1, 3))   # normalised epistemic uncertainty [0, 1]
    ind_dist_arr   = np.zeros((n + 1, 3))   # min L2 distance to nearest inducing point
    u              = np.zeros((n,     3))   # position commands (m)
    e              = np.zeros((n,     3))   # controller errors
    state_list: list[dict] = []             # sim._row snapshot per timestep

    plant_cache: dict = {}
    u_prev = np.array([_TYPICAL_POS_BY_EL[i] for i in (1, 2, 3)])

    # Equilibrium values and damping factor used after each advance step
    _ka_eq_sim   = float(_TYPICAL_KA_FOR.get(gp_variant, _TYPICAL_KA))
    _reac_eq_sim = float(_TYPICAL_REAC_FOR.get(gp_variant, _TYPICAL_REAC))
    _V_eq_sim    = _TYPICAL_V
    _DAMP_SIM    = 0.10   # fraction of ARX prediction retained per step

    # Rolling feature tracker, seeds from initial state and updated each step
    rolling_feats = _RollingFeatures()
    rolling_feats.update(sim._row)
    rolling_feats.inject(sim._row)

    # Seed y[0] from current simulator state
    sim._electrode = 1
    state_list.append(dict(sim._row))
    for i in (1, 2, 3):
        y[0, i - 1], gp_var_arr[0, i - 1], norm_var_arr[0, i - 1], ind_dist_arr[0, i - 1] = _gp_corrected_r(
            sim, gp_bundles, i, plant_cache, step=0, linear_models=linear_models,
            gp_variant=gp_variant, r_initial=r_initial,
        )
        plant_cache[f"r{i}"]      = y[0, i - 1]
        plant_cache[f"r{i}_lag2"] = y[0, i - 1]
    sim._electrode = 1

    for k in range(n):
        plant_cache["step"] = k
        rolling_feats.inject(sim._row)

        # Decoupled variants: freeze cross-electrode R lags at training-set means
        # before ARX predictions so cross-electrode R coupling is removed.
        if _r_cross_mean:
            for _j in (1, 2, 3):
                _mj = _r_cross_mean.get(_j, 0.0)
                for _k in (1, 2, 3):
                    _col = f"El{_j}_y_filt_lag{_k}"
                    if _col in sim._row.index:
                        sim._row[_col] = _mj

        # 1. Predict R one step ahead (ARX + GP) for all three electrodes
        y_pred:     dict = {}
        gp_var_k:   dict = {}
        norm_var_k: dict = {}
        ind_dist_k: dict = {}
        for i in (1, 2, 3):
            y_pred[i], gp_var_k[i], norm_var_k[i], ind_dist_k[i] = _gp_corrected_r(
                sim, gp_bundles, i, plant_cache, step=k, linear_models=linear_models,
                gp_variant=gp_variant, r_initial=r_initial,
            )

        sim._electrode = 1

        # 2. OOD gate: hold position and freeze integrators when GP is uncertain.
        # Disabled for "rollout": the SVGP's norm_var is not calibrated to the
        # same scale as the V9 GP so the threshold would gate continuously.
        mean_nv  = float(np.mean([norm_var_k[i] for i in (1, 2, 3)]))
        ood_hold = (mean_nv > _OOD_GATE_THRESHOLD) and (gp_variant not in _NEEDS_RTILDE_APPROX)

        # 3. Each controller computes its desired electrode position
        u_new: dict = {}
        if ood_hold:
            for i in (1, 2, 3):
                u_new[i]     = float(u_prev[i - 1])
                u[k, i - 1] = u_new[i]
                e[k, i - 1] = reference[k, i - 1] - y_pred[i]
        elif _unified:
            u_des, e_k = controllers.step(
                reference=reference[k],
                y_pred=y_pred,
                u_prev=u_prev,
            )
            if _decoupler is not None:
                # Decoupled PID: transform raw PID du through D = G^{-1} so each
                # controller acts on an independent channel with cross-coupling cancelled.
                _raw_du = np.array([
                    float(u_des[i-1][0] if hasattr(u_des[i-1], '__len__') else u_des[i-1])
                    - float(u_prev[i-1])
                    for i in (1, 2, 3)
                ])
                _dec_du = _decoupler @ _raw_du
                for i in (1, 2, 3):
                    _u_dec = float(u_prev[i-1]) + float(_dec_du[i-1])
                    u_ki   = apply_actuator_limits(_u_dec, u_prev[i - 1])
                    u_new[i]     = u_ki
                    u[k, i - 1] = u_ki
            else:
                for i in (1, 2, 3):
                    u_ki         = apply_actuator_limits(u_des[i - 1], u_prev[i - 1])
                    u_new[i]     = u_ki
                    u[k, i - 1] = u_ki
            e[k] = e_k
        else:
            for i in (1, 2, 3):
                u_des, e_k = controllers[i - 1].step(
                    reference=reference[k, i - 1],
                    y_pred=y_pred[i],
                    u_prev=u_prev[i - 1],
                )
                u_ki         = apply_actuator_limits(u_des, u_prev[i - 1])
                u_new[i]     = u_ki
                u[k, i - 1] = u_ki
                e[k, i - 1] = e_k

        # 4. Advance the simulator to the next time step.
        # Feed the raw ARX prediction (not the GP-corrected y_pred) back into
        # the y-lag registers.  The GP correction is only used for the output
        # and the controller.  Using y_pred here would create a feedback loop
        # where GP sign corrections get amplified each step.
        y_arx_vec = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx_vec)

        # Damp kA and reac lags toward their typical operating values after
        # each advance step.  The joint ARX kA / reactance feedback loops
        # are excited by relay moves in ways not seen during training and can
        # produce large oscillations (kA swinging 30-50 kA in a few steps).
        # A 10% blend per step keeps slow physical drift (kA and reactance do
        # change over a heat) while suppressing the fast oscillation artifact.
        for _i in (1, 2, 3):
            for _lag in (1, 2, 3):
                _kc = f"El{_i}_kA_filt_lag{_lag}"
                _rc = f"El{_i}_CalcReac_filt_lag{_lag}"
                if _kc in sim._row.index:
                    sim._row[_kc] = (1.0 - _DAMP_SIM) * _ka_eq_sim + _DAMP_SIM * float(sim._row[_kc])
                if _rc in sim._row.index:
                    sim._row[_rc] = (1.0 - _DAMP_SIM) * _reac_eq_sim + _DAMP_SIM * float(sim._row[_rc])
        _vc = "RMS_V_transformer_filt_lag1"
        if _vc in sim._row.index:
            sim._row[_vc] = (1.0 - _DAMP_SIM) * _V_eq_sim + _DAMP_SIM * float(sim._row[_vc])

        rolling_feats.update(sim._row, y_override=y_pred)
        state_list.append(dict(sim._row))

        for i in (1, 2, 3):
            y[k + 1, i - 1]             = y_pred[i]
            gp_var_arr[k + 1, i - 1]   = gp_var_k[i]
            norm_var_arr[k + 1, i - 1] = norm_var_k[i]
            ind_dist_arr[k + 1, i - 1] = ind_dist_k[i]
            plant_cache[f"r{i}_lag2"] = plant_cache.get(f"r{i}", y_arx_vec[i])
            plant_cache[f"r{i}"]      = y_arx_vec[i]

        u_prev = np.array([u_new[i] for i in (1, 2, 3)])

    t   = np.arange(n + 1) * dt
    u0  = np.full(3, _TYPICAL_POS)
    ref = np.vstack([np.full((1, 3), np.nan), reference])  # prepend NaN row

    state_df = pd.DataFrame(state_list)

    out = pd.DataFrame({
        "t_s": t,
        "y1":  y[:, 0],  "y2":  y[:, 1],  "y3":  y[:, 2],
        "r1":  ref[:, 0], "r2":  ref[:, 1], "r3":  ref[:, 2],
        "u1":  np.r_[u0[0], u[:, 0]],
        "u2":  np.r_[u0[1], u[:, 1]],
        "u3":  np.r_[u0[2], u[:, 2]],
        "e1":  np.r_[np.nan, e[:, 0]],
        "e2":  np.r_[np.nan, e[:, 1]],
        "e3":  np.r_[np.nan, e[:, 2]],
        "v_transformer": np.full(n + 1, _TYPICAL_V),
        "gp_var1":   gp_var_arr[:, 0],
        "gp_var2":   gp_var_arr[:, 1],
        "gp_var3":   gp_var_arr[:, 2],
        "norm_var1": norm_var_arr[:, 0],
        "norm_var2": norm_var_arr[:, 1],
        "norm_var3": norm_var_arr[:, 2],
        "ind_dist1": ind_dist_arr[:, 0],
        "ind_dist2": ind_dist_arr[:, 1],
        "ind_dist3": ind_dist_arr[:, 2],
    })
    out = pd.concat([out, state_df.reset_index(drop=True)], axis=1)

    out_csv = Path(out_csv)
    if not out_csv.is_absolute():
        out_csv = _HERE / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[fusion] Simulation done ({n} steps). Output: {out_csv}")
    return out
