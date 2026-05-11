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


# MODEL SELECTION -- change _GP_VARIANT to switch between the two plant models.
# Each variant bundles a matched ARX + GP pair trained on the same dataset.
# =============================================================================
_GP_VARIANT = "combined_deep_512"   # "pi_512" | "txt2026_512" | "combined_deep_512"

# ARX model paired with each GP variant (do not change unless you retrain):
_ARX_FOR_VARIANT = {
    "txt2026_512":      "arx_joint_txt2026.joblib",     # ARX trained on 2026 txt data
    "pi_512":           "arx_joint_pi_v3.joblib",       # ARX trained on PI data
    "combined_deep_512":"arx_joint_combined_v3.joblib", # ARX trained on PI + txt combined, deep kernel
    "combined_512":     "arx_joint_combined_v3.joblib", # ARX trained on PI + txt combined, Matern32 kernel
}
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


def _build_sim_and_gps() -> tuple[SaFSimulator, dict]:
    """
    Load the joint ARX model and per-electrode GP bundles.

    Returns (sim, gp_bundles) where gp_bundles is a dict {1: bundle, 2:..., 3:...}.
    Missing GP files are skipped; those electrodes run on ARX only.
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

    models_dir = _HERE / "models"
    arx_path   = models_dir / _ARX_MODEL

    if not arx_path.exists():
        raise FileNotFoundError(
            f"ARX model not found: {arx_path}\n"
            f"Expected: fusion/models/{_ARX_MODEL}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arx = joblib.load(arx_path)

    init_row = build_init_row_from_scalars(
        pos        = _TYPICAL_POS,
        r          = _TYPICAL_R,
        ka         = _TYPICAL_KA,
        rx         = _TYPICAL_REAC,
        v          = _TYPICAL_V,
        arx_bundle = arx,
    )

    # Override per-electrode R and position lags so each electrode starts at
    # its natural equilibrium rather than an identical symmetric value.
    for _i in (1, 2, 3):
        for _lag in (1, 2, 3):
            init_row[f"El{_i}_y_filt_lag{_lag}"]  = _TYPICAL_R_BY_EL[_i]
            init_row[f"El{_i}_pos_m_lag{_lag}"]    = _TYPICAL_POS_BY_EL[_i]

    sim = SaFSimulator(arx, init_row, electrode=1)

    gp_bundles: dict = {}
    for i in (1, 2, 3):
        gp_path = models_dir / f"gp_el{i}_{_GP_VARIANT}.pt"
        if gp_path.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_bundles[i] = load_gp_bundle(str(gp_path))
            print(f"[fusion] El{i} GP loaded: {gp_path.name}  "
                  f"features={len(gp_bundles[i]['feature_names'])}")
        else:
            print(f"[fusion] El{i}: GP not found at {gp_path.name}, running ARX only.")

    return sim, gp_bundles


def _gp_corrected_r(
    sim:         SaFSimulator,
    gp_bundles:  dict,
    electrode:   int,
    plant_cache: dict,
    step:        int,
) -> float:
    """Return GP-corrected R prediction for one electrode."""
    sim._electrode = electrode
    y_arx = sim._predict_r(electrode)
    bun   = gp_bundles.get(electrode)
    if bun is None:
        return y_arx

    feats = sim.get_gp_features_electrode(electrode)
    feats["step_in_window"] = float(min(step, 19))
    feats["y_sim"]          = y_arx
    feats["y_sim_sq"]       = y_arx * y_arx
    feats["y_real_lag1"]    = plant_cache.get(f"r{electrode}",      y_arx)
    feats["y_real_lag2"]    = plant_cache.get(f"r{electrode}_lag2", y_arx)

    x      = np.array([feats.get(f, 0.0) for f in bun["feature_names"]], dtype=np.float32)
    mu, _  = predict_single(bun, x)
    mu     = float(np.clip(mu, -0.15, 0.15))
    return float(y_arx + mu)


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
    **kwargs          : Ignored. Accepted for compatibility with older call sites.

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

    sim, gp_bundles = _build_sim_and_gps()
    reference       = _load_reference(ref_csv)          # (n, 3)
    for c in controllers:
        c.reset()

    n      = len(reference)
    y      = np.zeros((n + 1, 3))   # predicted R per electrode (mOhm)
    u      = np.zeros((n,     3))   # position commands (m)
    e      = np.zeros((n,     3))   # controller errors

    plant_cache: dict = {}
    u_prev = np.array([_TYPICAL_POS_BY_EL[i] for i in (1, 2, 3)])

    # Seed y[0] from current simulator state
    sim._electrode = 1
    for i in (1, 2, 3):
        y[0, i - 1] = _gp_corrected_r(sim, gp_bundles, i, plant_cache, step=0)
        plant_cache[f"r{i}"]      = y[0, i - 1]
        plant_cache[f"r{i}_lag2"] = y[0, i - 1]
    sim._electrode = 1

    for k in range(n):
        plant_cache["step"] = k

        # 1. Predict R one step ahead (ARX + GP) for all three electrodes
        y_pred = {}
        for i in (1, 2, 3):
            y_pred[i] = _gp_corrected_r(sim, gp_bundles, i, plant_cache, step=k)

        sim._electrode = 1

        # 2. Each controller computes its desired electrode position
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

        # 4. Advance the simulator to the next time step
        y_arx_vec = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx_vec)

        for i in (1, 2, 3):
            y[k + 1, i - 1]           = y_pred[i]
            plant_cache[f"r{i}_lag2"] = plant_cache.get(f"r{i}", y_arx_vec[i])
            plant_cache[f"r{i}"]      = y_arx_vec[i]

        u_prev = np.array([u_new[i] for i in (1, 2, 3)])

    t   = np.arange(n + 1) * dt
    u0  = np.full(3, _TYPICAL_POS)
    ref = np.vstack([np.full((1, 3), np.nan), reference])  # prepend NaN row

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
    })

    out_csv = Path(out_csv)
    if not out_csv.is_absolute():
        out_csv = _HERE / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[fusion] Simulation done ({n} steps). Output: {out_csv}")
    return out
