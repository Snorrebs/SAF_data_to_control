"""
run_closed_loop.py
------------------
Drop-in replacement for meta_arx/run_simulation/scripts/run_closed_loop.py.

Uses the joint ARX simulator and per-electrode GP correction instead of
the original single-electrode ARX model. Everything else is identical, same
function signature, same CSV output format, same controller types.

HOW TO USE
----------
In VRFT v5.py change one import line:

    # Old:
    from run_simulation.scripts.run_closed_loop_fusion import run_closed_loop_from_config

    # New (joint ARX + GP):
    from fusion.run_closed_loop import run_closed_loop_from_config

Everything else in VRFT v5.py stays exactly the same.

Output CSV columns
------------------
  t_s           : time in seconds
  y1, y2, y3   : predicted arc resistance per electrode (mOhm)
  r1, r2, r3   : reference signal per electrode (mOhm)
  u1, u2, u3   : electrode position commands (m)
  e1, e2, e3   : controller error per electrode (reference - y_pred)
  v_transformer : transformer RMS voltage (V)

Reference CSV format
--------------------
  Columns r1, r2, r3  -- per-electrode references (preferred)
  Column  reference   -- single reference broadcast to all three electrodes

Typical operating point used as initial state
---------------------------------------------
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

from .simulators.saf_simulator import SaFSimulator, build_init_row_from_scalars
from .training.gp_loader import load_gp_bundle, predict_single


# MODEL SELECTION -- change _GP_VARIANT to switch between plant models.
# Each variant is a matched ARX + GP pair trained on the same dataset.
# =============================================================================
_GP_VARIANT = "txt2026_512"

# ARX model paired with each GP variant (do not change unless you retrain):
_ARX_FOR_VARIANT = {
    "txt2026_512":       "arx_joint_txt2026.joblib",     # ARX trained on 2026 txt data
    "pi_512":            "arx_joint_pi_v3.joblib",       # ARX trained on PI data
    "combined_deep_512": "arx_joint_combined_v3.joblib", # ARX trained on PI + txt combined, deep kernel
    "combined_512":      "arx_joint_combined_v3.joblib", # ARX trained on PI + txt combined, Matern32 kernel
    "v6":                "arx_joint_v6.joblib",           # V6 joint ARX, debiased GP
    "v7":                "arx_joint_v6.joblib",           # V7 two-stage: linear correction + GP
}

# V7 uses a lightweight linear residual model before the GP.
# The linear model removes ~97% of the ARX residual variance; the GP then
# learns only the remaining nonlinear component.
_HAS_LINEAR_STAGE = {"v7"}

_ARX_MODEL = _ARX_FOR_VARIANT[_GP_VARIANT]
# =============================================================================

# Typical SAF operating point used to seed the initial simulator state.
# Per-electrode R values come from a stable rule-controller run — the three
# electrodes sit at different resistances in steady state.
_TYPICAL_POS    = 1.04
_TYPICAL_R      = 1.006   # fallback scalar (not used for multi-electrode init)
_TYPICAL_KA     = 65.0
_TYPICAL_REAC   = 0.82
_TYPICAL_V      = 165.0
_TYPICAL_POS_BY_EL = {1: 1.04,  2: 1.03,  3: 1.04}
_TYPICAL_R_BY_EL   = {1: 1.20,  2: 0.77,  3: 1.14}

# V6/V7 were trained on PI furnace data at a different operating point.
# Using the correct kA and R values prevents the ARX from starting 5+ sigma
# outside its training distribution and immediately diverging.
_TYPICAL_KA_FOR    = {"v6": 118.0, "v7": 118.0}
_TYPICAL_REAC_FOR  = {"v6": 0.88,  "v7": 0.88}
_TYPICAL_R_BY_EL_FOR = {
    "v6": {1: 1.08, 2: 1.07, 3: 1.07},
    "v7": {1: 1.08, 2: 1.07, 3: 1.07},
}


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
    gp_bundles    : {1: bundle, 2: bundle, 3: bundle}  -- missing keys -> ARX only
    linear_models : {1: model, 2: model, 3: model}     -- empty unless variant is V7
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


def _gp_corrected_r(
    sim:           SaFSimulator,
    gp_bundles:    dict,
    electrode:     int,
    plant_cache:   dict,
    step:          int,
    linear_models: "dict | None" = None,
) -> "tuple[float, float]":
    """
    Return (R_corrected, gp_variance) for one electrode.

    For V7, a linear residual correction is applied first, then the GP corrects
    the remaining nonlinear component.  The total correction (linear + GP) is
    clipped to +/-0.15 mOhm to limit extrapolation damage.
    """
    sim._electrode = electrode
    y_arx = sim._predict_r(electrode)
    bun   = gp_bundles.get(electrode)
    if bun is None:
        return y_arx, 0.0

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

    mu, var = predict_single(bun, x)
    mu      = float(np.clip(lin_delta + mu, -0.15, 0.15))
    return float(y_arx + mu), float(var)


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
    ----------
    ref_csv           : CSV with reference signal(s).
                        Columns r1/r2/r3 for per-electrode refs, or a single
                        'reference' column broadcast to all three electrodes.
    controller_name   : Controller type, e.g. "pid" or "open_loop".
    controller_config : CSV with controller parameters (kp, ki, kd for PID).
    out_csv           : Where to write the simulation output CSV.
    dt                : Sample time in seconds (default 1.0).
    gp_variant        : Optional. Override the module-level _GP_VARIANT for this call.
                        Accepted values: "txt2026_512", "pi_512", "combined_512",
                        "combined_deep_512", "v6", "v7".
    **kwargs          : Other keyword arguments are ignored (compatibility).

    Returns
    -------
    pd.DataFrame with columns:
        t_s, y1, y2, y3, r1, r2, r3, u1, u2, u3, e1, e2, e3, v_transformer
    """
    from run_simulation.closed_loop.closed_loop_sim import apply_actuator_limits

    # Multi-electrode API (make_controllers) is preferred; fall back to
    # calling the single-electrode factory three times for older meta_arx versions.
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

    gp_variant = kwargs.pop("gp_variant", _GP_VARIANT)
    sim, gp_bundles, linear_models = _build_sim_and_gps(gp_variant)
    reference = _load_reference(ref_csv)          # (n, 3)
    
# Small change to allow unified controller: MARKUS_TEST_CODE
    if controller_name != "generalized_controller":
        for c in controllers:
            c.reset()
        else:
            controllers[0].reset()
# End change
    n           = len(reference)
    y           = np.zeros((n + 1, 3))   # predicted R per electrode (mOhm)
    gp_var_arr  = np.zeros((n + 1, 3))   # GP predictive variance per electrode
    u           = np.zeros((n,     3))   # position commands (m)
    e           = np.zeros((n,     3))   # controller errors
    state_list: list[dict] = []          # sim._row snapshot per timestep

    plant_cache: dict = {}
    u_prev = np.array([_TYPICAL_POS_BY_EL[i] for i in (1, 2, 3)])

    # Seed y[0] from current simulator state
    sim._electrode = 1
    state_list.append(dict(sim._row))
    for i in (1, 2, 3):
        y[0, i - 1], gp_var_arr[0, i - 1] = _gp_corrected_r(
            sim, gp_bundles, i, plant_cache, step=0, linear_models=linear_models
        )
        plant_cache[f"r{i}"]      = y[0, i - 1]
        plant_cache[f"r{i}_lag2"] = y[0, i - 1]
    sim._electrode = 1

    for k in range(n):
        plant_cache["step"] = k

        # 1. Predict R one step ahead (ARX + GP) for all three electrodes
        y_pred:   dict = {}
        gp_var_k: dict = {}
        for i in (1, 2, 3):
            y_pred[i], gp_var_k[i] = _gp_corrected_r(
                sim, gp_bundles, i, plant_cache, step=k, linear_models=linear_models
            )

        sim._electrode = 1

        # 2. Each controller computes its desired electrode position

# Test fuunction for unified multielectrode controller. MARKUS_TEST_CODE
        if controller_name == "generalized_controller":
            # 2.1. Unified controller output
            u_new: dict = {}

            u_des, e_k = controllers.step(
                reference = reference[k],
                y_pred    = y_pred,
                u_prev    = u_prev
            )
            #3.1. Clip the position command to actuator limits (speed, range)

            for i in (1,2,3):
                u_ki = apply_actuator_limits(u_des[i-1], u_prev[i - 1])
                u_new[i]      = u_ki
                u[k, i - 1]   = u_ki
            e[k]  = e_k
        else:
            u_new: dict = {}
            for i in (1, 2, 3):
                u_des, e_k = controllers[i - 1].step(
                    reference = reference[k, i - 1],
                    y_pred    = y_pred[i],
                    u_prev    = u_prev[i - 1],
                    )
                # 3. Clip the position command to actuator limits (speed, range)
                u_ki          = apply_actuator_limits(u_des, u_prev[i - 1])
                u_new[i]      = u_ki
                u[k, i - 1]  = u_ki
                e[k, i - 1]  = e_k
# End test.
        # 4. Advance the simulator to the next time step
        y_arx_vec = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx_vec)

        state_list.append(dict(sim._row))

        for i in (1, 2, 3):
            y[k + 1, i - 1]           = y_pred[i]
            gp_var_arr[k + 1, i - 1]  = gp_var_k[i]
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
        "gp_var1": gp_var_arr[:, 0],
        "gp_var2": gp_var_arr[:, 1],
        "gp_var3": gp_var_arr[:, 2],
    })
    out = pd.concat([out, state_df.reset_index(drop=True)], axis=1)

    out_csv = Path(out_csv)
    if not out_csv.is_absolute():
        out_csv = _HERE / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[fusion] Simulation done ({n} steps). Output: {out_csv}")
    return out
